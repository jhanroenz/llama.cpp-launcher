#!/usr/bin/env python3
"""OpenAI-compatible agent loop + local tools for llama-launcher."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

CREATE_NO_WINDOW = 0x08000000
MAX_READ_BYTES = 1_000_000
MAX_TOOL_OUTPUT_CHARS = 80_000
MAX_SHELL_OUTPUT_CHARS = 40_000
MAX_GLOB_RESULTS = 500
MAX_GREP_RESULTS = 200
MAX_MEMORY_LIST = 500
SHELL_TIMEOUT_SEC = 60


class AgentCancelled(Exception):
    """Raised when the user stops an in-flight agent turn."""


class AgentError(Exception):
    """Fatal agent / API error."""


@dataclass
class AgentConfig:
    host: str = "127.0.0.1"
    port: str = "11434"
    api_key: str = ""
    workspace: Path = field(default_factory=Path.home)
    vault: Path | None = None
    max_steps: int = 20
    temperature: float = 0.2
    model: str = "local"


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]"


def _safe_resolve(root: Path, user_path: str | Path) -> Path:
    root = root.resolve()
    raw = Path(user_path)
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AgentError(f"Path escapes allowed root ({root}): {user_path}") from exc
    return target


def _rg_available() -> bool:
    return shutil.which("rg") is not None


def _run_rg(
    pattern: str,
    search_root: Path,
    *,
    glob: str | None = None,
    max_results: int = MAX_GREP_RESULTS,
) -> str:
    cmd = ["rg", "--line-number", "--color", "never", "--max-count", str(max_results), pattern]
    if glob:
        cmd.extend(["--glob", glob])
    cmd.append(str(search_root))
    flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SHELL_TIMEOUT_SEC,
            creationflags=flags,
        )
    except FileNotFoundError:
        return _python_grep(pattern, search_root, glob=glob, max_results=max_results)
    except subprocess.TimeoutExpired:
        return "Error: ripgrep timed out."
    # rg exit 1 = no matches
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode not in (0, 1):
        return _truncate(f"rg failed ({proc.returncode}): {err or out}")
    if not out:
        return "(no matches)"
    return _truncate(out)


def _python_grep(
    pattern: str,
    search_root: Path,
    *,
    glob: str | None = None,
    max_results: int = MAX_GREP_RESULTS,
) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Invalid regex: {exc}"
    hits: list[str] = []
    paths: list[Path]
    if search_root.is_file():
        paths = [search_root]
    else:
        pattern_glob = glob or "**/*"
        paths = sorted(p for p in search_root.glob(pattern_glob) if p.is_file())
    for path in paths:
        if len(hits) >= max_results:
            break
        try:
            if path.stat().st_size > MAX_READ_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                rel = path
                try:
                    rel = path.relative_to(search_root if search_root.is_dir() else search_root.parent)
                except ValueError:
                    pass
                hits.append(f"{rel}:{i}:{line}")
                if len(hits) >= max_results:
                    break
    if not hits:
        return "(no matches)"
    return _truncate("\n".join(hits))


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file under the agent workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace (or absolute under workspace)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file under the agent workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a workspace file, or write full content if old_string is empty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string", "description": "Exact text to replace (empty = overwrite whole file)."},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_glob_search",
            "description": "Find files under the workspace matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents under the workspace (uses ripgrep when available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Optional subpath under workspace."},
                    "glob": {"type": "string", "description": "Optional file glob filter."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_shell_command",
            "description": (
                "Run a shell command with cwd=workspace. Not path-sandboxed beyond cwd. "
                "Prefer specialized tools for file edits when possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "description": "Optional timeout (default 60)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return title/url/snippet results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Default 5, max 15."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "List markdown notes in the Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Optional vault-relative folder prefix."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a markdown note from the Obsidian vault by vault-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative path, e.g. inbox/note.md"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Create or overwrite a markdown note in the Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative path ending in .md"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search markdown notes in the Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "folder": {"type": "string", "description": "Optional vault-relative folder."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the local date/time in ISO format.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolExecutor:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace = config.workspace.resolve()
        self.vault = config.vault.resolve() if config.vault else None

    def _require_vault(self) -> Path:
        if not self.vault or not self.vault.is_dir():
            raise AgentError("Obsidian vault is not configured or does not exist.")
        return self.vault

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "file_glob_search": self._file_glob_search,
            "grep_search": self._grep_search,
            "exec_shell_command": self._exec_shell,
            "web_search": self._web_search,
            "memory_list": self._memory_list,
            "memory_read": self._memory_read,
            "memory_write": self._memory_write,
            "memory_search": self._memory_search,
            "get_datetime": self._get_datetime,
        }
        fn = handlers.get(name)
        if not fn:
            return f"Unknown tool: {name}"
        try:
            return fn(arguments)
        except AgentError as exc:
            return f"Error: {exc}"
        except OSError as exc:
            return f"OS error: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Tool error: {exc}"

    def _read_file(self, args: dict[str, Any]) -> str:
        path = _safe_resolve(self.workspace, str(args.get("path") or ""))
        if not path.is_file():
            raise AgentError(f"Not a file: {path}")
        if path.stat().st_size > MAX_READ_BYTES:
            raise AgentError(f"File too large (>{MAX_READ_BYTES} bytes)")
        return _truncate(path.read_text(encoding="utf-8", errors="replace"))

    def _write_file(self, args: dict[str, Any]) -> str:
        path = _safe_resolve(self.workspace, str(args.get("path") or ""))
        content = str(args.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def _edit_file(self, args: dict[str, Any]) -> str:
        path = _safe_resolve(self.workspace, str(args.get("path") or ""))
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") if args.get("new_string") is not None else "")
        if not path.is_file() and old:
            raise AgentError(f"File not found: {path}")
        if not old:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, encoding="utf-8")
            return f"Wrote full file ({len(new)} chars) to {path}"
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            raise AgentError("old_string not found in file")
        if count > 1:
            raise AgentError(f"old_string matched {count} times; make it unique")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"Edited {path} (1 replacement)"

    def _file_glob_search(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "").strip()
        if not pattern:
            raise AgentError("pattern is required")
        matches = sorted(p for p in self.workspace.glob(pattern) if p.is_file())
        if not matches:
            return "(no matches)"
        lines = []
        for p in matches[:MAX_GLOB_RESULTS]:
            try:
                lines.append(str(p.relative_to(self.workspace)))
            except ValueError:
                lines.append(str(p))
        extra = len(matches) - len(lines)
        text = "\n".join(lines)
        if extra > 0:
            text += f"\n…and {extra} more"
        return _truncate(text)

    def _grep_search(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "")
        if not pattern:
            raise AgentError("pattern is required")
        sub = str(args.get("path") or "").strip()
        root = _safe_resolve(self.workspace, sub) if sub else self.workspace
        if not root.exists():
            raise AgentError(f"Path not found: {root}")
        glob = str(args.get("glob") or "").strip() or None
        if _rg_available():
            return _run_rg(pattern, root, glob=glob)
        return _python_grep(pattern, root, glob=glob)

    def _exec_shell(self, args: dict[str, Any]) -> str:
        command = str(args.get("command") or "").strip()
        if not command:
            raise AgentError("command is required")
        timeout = args.get("timeout_sec", SHELL_TIMEOUT_SEC)
        try:
            timeout_f = float(timeout)
        except (TypeError, ValueError):
            timeout_f = float(SHELL_TIMEOUT_SEC)
        timeout_f = max(1.0, min(300.0, timeout_f))
        self.workspace.mkdir(parents=True, exist_ok=True)
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        shell = True
        try:
            proc = subprocess.run(
                command,
                shell=shell,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_f,
                creationflags=flags,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout_f}s"
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        out = _truncate(out.strip() or "(no output)", MAX_SHELL_OUTPUT_CHARS)
        return f"exit={proc.returncode}\n{out}"

    def _web_search(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise AgentError("query is required")
        max_results = args.get("max_results", 5)
        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(15, n))
        try:
            from ddgs import DDGS
        except ImportError:
            return "Error: ddgs package not installed. Run: pip install ddgs"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
        except Exception as exc:  # noqa: BLE001
            return f"Web search failed: {exc}"
        if not results:
            return "(no results)"
        lines = []
        for i, row in enumerate(results, start=1):
            title = row.get("title") or ""
            href = row.get("href") or row.get("link") or ""
            body = row.get("body") or row.get("snippet") or ""
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return _truncate("\n".join(lines))

    def _memory_list(self, args: dict[str, Any]) -> str:
        vault = self._require_vault()
        folder = str(args.get("folder") or "").strip()
        root = _safe_resolve(vault, folder) if folder else vault
        if not root.is_dir():
            raise AgentError(f"Not a directory: {root}")
        notes = sorted(p for p in root.rglob("*.md") if p.is_file())
        # Skip Obsidian internals
        notes = [p for p in notes if ".obsidian" not in p.parts]
        if not notes:
            return "(no notes)"
        lines = []
        for p in notes[:MAX_MEMORY_LIST]:
            try:
                lines.append(str(p.relative_to(vault)).replace("\\", "/"))
            except ValueError:
                lines.append(str(p))
        extra = len(notes) - len(lines)
        text = "\n".join(lines)
        if extra > 0:
            text += f"\n…and {extra} more"
        return _truncate(text)

    def _memory_read(self, args: dict[str, Any]) -> str:
        vault = self._require_vault()
        path = _safe_resolve(vault, str(args.get("path") or ""))
        if not path.is_file():
            raise AgentError(f"Note not found: {path}")
        if path.stat().st_size > MAX_READ_BYTES:
            raise AgentError(f"Note too large (>{MAX_READ_BYTES} bytes)")
        return _truncate(path.read_text(encoding="utf-8", errors="replace"))

    def _memory_write(self, args: dict[str, Any]) -> str:
        vault = self._require_vault()
        rel = str(args.get("path") or "").strip()
        if not rel:
            raise AgentError("path is required")
        if not rel.lower().endswith(".md"):
            rel = rel + ".md"
        path = _safe_resolve(vault, rel)
        content = str(args.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        try:
            shown = str(path.relative_to(vault)).replace("\\", "/")
        except ValueError:
            shown = str(path)
        return f"Wrote note {shown} ({len(content)} chars)"

    def _memory_search(self, args: dict[str, Any]) -> str:
        vault = self._require_vault()
        pattern = str(args.get("pattern") or "")
        if not pattern:
            raise AgentError("pattern is required")
        folder = str(args.get("folder") or "").strip()
        root = _safe_resolve(vault, folder) if folder else vault
        if _rg_available():
            return _run_rg(pattern, root, glob="*.md")
        return _python_grep(pattern, root, glob="**/*.md")

    def _get_datetime(self, _args: dict[str, Any]) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def build_tools_prompt_section() -> str:
    lines = ["Available tools (call via <tool_call> blocks — see format below):"]
    for spec in TOOL_SCHEMAS:
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else {}
        name = fn.get("name") or "unknown"
        desc = fn.get("description") or ""
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        lines.append(f"- {name}: {desc}")
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        if props:
            param_bits = []
            for pname, pschema in props.items():
                if isinstance(pschema, dict):
                    param_bits.append(f"{pname} ({pschema.get('type', 'string')})")
                else:
                    param_bits.append(str(pname))
            lines.append(f"  parameters: {', '.join(param_bits)}")
    lines.extend(
        [
            "",
            "To call a tool, emit one or more blocks exactly like:",
            '<tool_call>{"name": "TOOL_NAME", "arguments": {...}}</tool_call>',
            "",
            "After tool results are returned, continue or give the final answer.",
            "Do not wrap the JSON in markdown fences inside <tool_call>.",
            "When done, reply normally without any <tool_call> block.",
        ]
    )
    return "\n".join(lines)


def build_system_prompt(config: AgentConfig) -> str:
    vault_line = (
        f"Obsidian vault: {config.vault.resolve()}"
        if config.vault
        else "Obsidian vault: (not configured — memory_* tools will fail until set)"
    )
    return (
        "You are a local coding and research agent running inside llama-launcher.\n"
        f"Workspace root (file/shell tools): {config.workspace.resolve()}\n"
        f"{vault_line}\n"
        "Use memory_* tools for vault notes; use read/write/edit/grep/glob for code and files in the workspace.\n"
        "Prefer specialized tools over shell when possible. Be concise.\n"
        "Shell commands are not path-sandboxed beyond cwd=workspace — be careful.\n\n"
        + build_tools_prompt_section()
    )


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_tool_call(obj: dict[str, Any]) -> ParsedToolCall | None:
    name = obj.get("name") or obj.get("tool")
    fn = obj.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or name
        args_raw = fn.get("arguments")
    else:
        args_raw = None
    if not name:
        return None
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or args_raw or {}
    if isinstance(args, str):
        args = _parse_tool_args(args)
    if not isinstance(args, dict):
        args = {}
    return ParsedToolCall(str(name), args, json.dumps(obj, ensure_ascii=False)[:500])


def _parse_qwen_xml_tool_call(block: str) -> ParsedToolCall | None:
    fn_match = re.search(r"<function=([^>\s]+)>", block)
    if not fn_match:
        return None
    name = fn_match.group(1)
    args: dict[str, Any] = {}
    for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", block, re.DOTALL):
        args[pm.group(1)] = pm.group(2).strip()
    return ParsedToolCall(name, args, block[:500])


def _parse_tool_call_inner(inner: str) -> list[ParsedToolCall]:
    inner_clean = _strip_code_fences(inner.strip())
    if not inner_clean:
        return []
    try:
        obj = json.loads(inner_clean)
    except json.JSONDecodeError:
        tc = _parse_qwen_xml_tool_call(inner)
        return [tc] if tc else []
    if isinstance(obj, list):
        out: list[ParsedToolCall] = []
        for item in obj:
            if isinstance(item, dict):
                tc = _parse_json_tool_call(item)
                if tc:
                    out.append(tc)
        return out
    if isinstance(obj, dict):
        tc = _parse_json_tool_call(obj)
        return [tc] if tc else []
    return []


def parse_tool_calls_from_text(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Split assistant text into user-visible prose and parsed tool calls."""
    if not text:
        return "", []

    calls: list[ParsedToolCall] = []
    block_pat = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
    for match in block_pat.finditer(text):
        calls.extend(_parse_tool_call_inner(match.group(1)))

    visible = block_pat.sub("", text).strip()

    if not calls:
        for match in re.finditer(
            r'\{\s*"(?:name|tool)"\s*:\s*"([^"]+)"\s*,\s*"(?:arguments|parameters|args)"\s*:\s*(\{[\s\S]*?\})\s*\}',
            text,
        ):
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                tc = _parse_json_tool_call(obj)
                if tc:
                    calls.append(tc)
        if calls:
            for tc in calls:
                visible = visible.replace(tc.raw, "").strip()

    return visible, calls


def _tool_calls_from_api_message(assistant: dict[str, Any]) -> list[ParsedToolCall]:
    """Fallback if the server returns structured tool_calls without us requesting tools."""
    out: list[ParsedToolCall] = []
    for call in assistant.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "unknown")
        args = _parse_tool_args(fn.get("arguments"))
        out.append(ParsedToolCall(name, args, json.dumps(call, ensure_ascii=False)[:500]))
    return out


def _assistant_visible_text(assistant: dict[str, Any]) -> str:
    content = str(assistant.get("content") or "")
    visible, _calls = parse_tool_calls_from_text(content)
    if visible:
        return visible
    return content.strip()


def _format_tool_results(name: str, result: str) -> str:
    return f"[Tool result for {name}]\n{result}"


def chat_completions(
    config: AgentConfig,
    messages: list[dict[str, Any]],
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if cancel_check and cancel_check():
        raise AgentCancelled()

    host = (config.host or "127.0.0.1").strip() or "127.0.0.1"
    if host in ("0.0.0.0", "::", "[::]"):
        host = "127.0.0.1"
    port = str(config.port or "11434").strip() or "11434"
    url = f"http://{host}:{port}/v1/chat/completions"

    # Do not send OpenAI "tools" — llama-server switches to peg-native parsing and
    # returns HTTP 500 when model output does not match. Tools run in Python instead.
    body: dict[str, Any] = {
        "model": config.model or "local",
        "messages": messages,
        "temperature": config.temperature,
        "stream": False,
    }

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise AgentError(f"HTTP {exc.code} from {url}: {_truncate(detail, 2000)}") from exc
    except urllib.error.URLError as exc:
        raise AgentError(f"Cannot reach llama-server at {url}: {exc.reason}") from exc

    if cancel_check and cancel_check():
        raise AgentCancelled()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentError(f"Invalid JSON from server: {_truncate(raw, 500)}") from exc
    return payload


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": raw}
    return {"value": raw}


def _assistant_message_from_choice(choice: dict[str, Any]) -> dict[str, Any]:
    msg = choice.get("message")
    if isinstance(msg, dict):
        out: dict[str, Any] = {"role": "assistant"}
        if "content" in msg:
            out["content"] = msg.get("content")
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning_content"):
            out["reasoning_content"] = msg["reasoning_content"]
        return out
    # Fallback older shape
    return {"role": "assistant", "content": choice.get("text") or ""}


@dataclass
class AgentCallbacks:
    on_assistant: Callable[[str], None] | None = None
    on_tool_start: Callable[[str, str, dict[str, Any]], None] | None = None
    on_tool_end: Callable[[str, str, str], None] | None = None
    on_status: Callable[[str], None] | None = None


def run_agent_turn(
    config: AgentConfig,
    history: list[dict[str, Any]],
    user_text: str,
    *,
    callbacks: AgentCallbacks | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Append user message, run tool loop, return updated history (including assistant/tool msgs)."""
    cb = callbacks or AgentCallbacks()
    workspace = Path(config.workspace).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    config.workspace = workspace
    if config.vault:
        config.vault = Path(config.vault).expanduser()

    messages = list(history)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": build_system_prompt(config)})
    else:
        messages[0] = {"role": "system", "content": build_system_prompt(config)}

    messages.append({"role": "user", "content": user_text})
    executor = ToolExecutor(config)
    max_steps = max(1, min(100, int(config.max_steps)))

    for step in range(max_steps):
        if cancel_check and cancel_check():
            raise AgentCancelled()
        if cb.on_status:
            cb.on_status(f"Thinking (step {step + 1}/{max_steps})…")

        payload = chat_completions(config, messages, cancel_check=cancel_check)
        choices = payload.get("choices") or []
        if not choices:
            raise AgentError(f"Empty choices from server: {_truncate(json.dumps(payload), 800)}")

        assistant = _assistant_message_from_choice(choices[0] if isinstance(choices[0], dict) else {})
        content = str(assistant.get("content") or "")
        reasoning = assistant.get("reasoning_content")
        history_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reasoning:
            history_msg["reasoning_content"] = reasoning
        messages.append(history_msg)

        parsed_calls = parse_tool_calls_from_text(content)
        if not parsed_calls:
            parsed_calls = _tool_calls_from_api_message(assistant)

        visible = _assistant_visible_text(assistant)
        if visible and cb.on_assistant:
            cb.on_assistant(visible)

        if not parsed_calls:
            if cb.on_status:
                cb.on_status("Done")
            return messages

        result_chunks: list[str] = []
        for i, call in enumerate(parsed_calls):
            if cancel_check and cancel_check():
                raise AgentCancelled()
            call_id = f"call_{step}_{i}"
            if cb.on_tool_start:
                cb.on_tool_start(call_id, call.name, call.arguments)
            if cb.on_status:
                cb.on_status(f"Tool: {call.name}")
            result = executor.execute(call.name, call.arguments)
            if cb.on_tool_end:
                cb.on_tool_end(call_id, call.name, result)
            result_chunks.append(_format_tool_results(call.name, result))

        messages.append({"role": "user", "content": "\n\n".join(result_chunks)})

    # Max steps reached — ask once more for a final answer
    if cb.on_status:
        cb.on_status("Max tool steps reached; requesting final answer…")
    messages.append(
        {
            "role": "user",
            "content": "You have reached the maximum number of tool steps. Give your best final answer now without calling tools.",
        }
    )
    payload = chat_completions(config, messages, cancel_check=cancel_check)
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        assistant = _assistant_message_from_choice(choices[0])
        messages.append({"role": "assistant", "content": assistant.get("content")})
        visible = _assistant_visible_text(assistant)
        if visible and cb.on_assistant:
            cb.on_assistant(visible)
    if cb.on_status:
        cb.on_status("Done")
    return messages


def profile_api_key(profile: dict[str, Any]) -> str:
    if profile.get("use_api_key"):
        return str(profile.get("api_key") or "").strip()
    return ""


def normalize_endpoint_host(host: str) -> str:
    h = (host or "127.0.0.1").strip() or "127.0.0.1"
    if h in ("0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return h
