#!/usr/bin/env python3
"""llama-launcher — advanced visual profile editor and llama-server / llama-bench wrapper."""

from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import sys
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap, QPolygon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

APP_NAME = "llama-launcher"
CREATE_NO_WINDOW = 0x08000000
PROFILES_SCHEMA = 3

# Powers-of-2 defaults + common llama.cpp sizes from your profiles
CTX_PRESETS: tuple[str, ...] = tuple(
    str(n)
    for n in sorted(
        {
            2048,
            4096,
            8192,
            16384,
            32768,
            57344,
            65536,
            131072,
            262144,
        }
    )
)

FLASH_ATTN_PRESETS = ("on", "off", "auto")
REASONING_PRESETS = ("off", "on", "auto")
FIT_PRESETS = ("on", "off")
TOOL_PRESETS = (
    "all",
    "read_file,file_glob_search,grep_search,get_datetime",
    "read_file,write_file,edit_file,file_glob_search,grep_search",
    "read_file,file_glob_search,grep_search,exec_shell_command,write_file,edit_file,get_datetime",
)


@dataclass(frozen=True)
class Knob:
    """Declarative optional llama-server knob (checkbox-gated)."""

    group: str
    use_key: str
    value_key: str
    flag: str
    kind: str  # value | bool | combo
    default: str = ""
    choices: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    label: str = ""

    @property
    def ui_label(self) -> str:
        return self.label or self.flag


# Common runtime + tool-use + sampling + server knobs (not every CLI flag).
KNOBS: tuple[Knob, ...] = (
    # --- Runtime ---
    Knob("Runtime", "use_ot", "ot", "-ot", "value", ".ffn_.*_exps.=CPU", (), ("--override-tensor",), "-ot (tensor override)"),
    Knob("Runtime", "use_batch", "batch", "-b", "value", "512", (), ("--batch-size",), "-b (batch-size)"),
    Knob("Runtime", "use_ubatch", "ubatch", "-ub", "value", "512", (), ("--ubatch-size",), "-ub (ubatch-size)"),
    Knob("Runtime", "use_flash_attn", "flash_attn", "--flash-attn", "combo", "on", FLASH_ATTN_PRESETS, ("-fa",), "--flash-attn"),
    Knob("Runtime", "use_np", "np", "-np", "value", "1", (), ("--parallel",), "-np (parallel slots)"),
    Knob("Runtime", "use_cache_reuse", "cache_reuse", "--cache-reuse", "value", "256", (), (), "--cache-reuse"),
    Knob("Runtime", "use_n_predict", "n_predict", "-n", "value", "-1", (), ("--predict", "--n-predict"), "-n (n-predict)"),
    Knob("Runtime", "use_threads_batch", "threads_batch", "-tb", "value", "8", (), ("--threads-batch",), "-tb (threads-batch)"),
    Knob("Runtime", "use_n_cpu_moe", "n_cpu_moe", "--n-cpu-moe", "value", "35", (), ("-ncmoe",), "--n-cpu-moe"),
    Knob("Runtime", "use_cpu_moe", "cpu_moe", "--cpu-moe", "bool", "", (), ("-cmoe",), "--cpu-moe"),
    Knob("Runtime", "use_fit", "fit", "--fit", "combo", "on", FIT_PRESETS, ("-fit",), "--fit"),
    Knob("Runtime", "use_no_warmup", "no_warmup", "--no-warmup", "bool", "", (), (), "--no-warmup"),
    Knob("Runtime", "use_kv_unified", "kv_unified", "--kv-unified", "bool", "", (), ("-kvu",), "--kv-unified"),
    Knob("Runtime", "use_no_kv_offload", "no_kv_offload", "--no-kv-offload", "bool", "", (), ("-nkvo",), "--no-kv-offload"),
    Knob("Runtime", "use_slots", "slots", "--slots", "bool", "", (), (), "--slots"),
    Knob("Runtime", "use_slot_save_path", "slot_save_path", "--slot-save-path", "value", str(Path(r"D:\slots")), (), (), "--slot-save-path"),
    Knob("Runtime", "use_no_context_shift", "no_context_shift", "--no-context-shift", "bool", "", (), (), "--no-context-shift"),
    Knob("Runtime", "use_swa_full", "swa_full", "--swa-full", "bool", "", (), (), "--swa-full"),
    Knob("Runtime", "use_mmproj", "mmproj", "--mmproj", "value", "", (), ("-mm",), "--mmproj"),
    Knob("Runtime", "use_alias", "alias", "--alias", "value", "", (), ("-a",), "--alias"),
    Knob("Runtime", "use_chat_template_kwargs", "chat_template_kwargs", "--chat-template-kwargs", "value", '{"enable_thinking":true}', (), (), "--chat-template-kwargs"),
    Knob("Runtime", "use_chat_template", "chat_template", "--chat-template", "value", "", (), (), "--chat-template"),
    Knob("Runtime", "use_chat_template_file", "chat_template_file", "--chat-template-file", "value", "", (), (), "--chat-template-file"),
    # --- Reasoning ---
    Knob("Reasoning", "use_reasoning", "reasoning", "--reasoning", "combo", "off", REASONING_PRESETS, ("-rea",), "--reasoning"),
    Knob("Reasoning", "use_reasoning_budget", "reasoning_budget", "--reasoning-budget", "value", "0", (), (), "--reasoning-budget"),
    Knob("Reasoning", "use_reasoning_format", "reasoning_format", "--reasoning-format", "value", "deepseek", (), (), "--reasoning-format"),
    Knob("Reasoning", "use_reasoning_budget_message", "reasoning_budget_message", "--reasoning-budget-message", "value", "", (), (), "--reasoning-budget-message"),
    # --- Tool use / agent ---
    Knob("Tools", "use_tools", "tools", "--tools", "combo", "all", TOOL_PRESETS, (), "--tools"),
    Knob("Tools", "use_agent", "agent", "--agent", "bool", "", (), ("-ag",), "--agent (all tools + CORS proxy)"),
    # --- Sampling ---
    Knob("Sampling", "use_temp", "temp", "--temp", "value", "0.8", (), ("--temperature",), "--temp"),
    Knob("Sampling", "use_top_k", "top_k", "--top-k", "value", "40", (), (), "--top-k"),
    Knob("Sampling", "use_top_p", "top_p", "--top-p", "value", "0.95", (), (), "--top-p"),
    Knob("Sampling", "use_min_p", "min_p", "--min-p", "value", "0.05", (), (), "--min-p"),
    Knob("Sampling", "use_seed", "seed", "--seed", "value", "-1", (), ("-s",), "--seed"),
    Knob("Sampling", "use_repeat_penalty", "repeat_penalty", "--repeat-penalty", "value", "1.0", (), (), "--repeat-penalty"),
    Knob("Sampling", "use_presence_penalty", "presence_penalty", "--presence-penalty", "value", "0.0", (), (), "--presence-penalty"),
    Knob("Sampling", "use_frequency_penalty", "frequency_penalty", "--frequency-penalty", "value", "0.0", (), (), "--frequency-penalty"),
    Knob("Sampling", "use_ignore_eos", "ignore_eos", "--ignore-eos", "bool", "", (), (), "--ignore-eos"),
    Knob("Sampling", "use_grammar", "grammar", "--grammar", "value", "", (), (), "--grammar"),
    Knob("Sampling", "use_json_schema", "json_schema", "--json-schema", "value", "", (), ("-j",), "--json-schema"),
    # --- Server ---
    Knob("Server", "use_api_key", "api_key", "--api-key", "value", "", (), (), "--api-key"),
    Knob("Server", "use_metrics", "metrics", "--metrics", "bool", "", (), (), "--metrics"),
    Knob("Server", "use_props", "props", "--props", "bool", "", (), (), "--props"),
    Knob("Server", "use_cont_batching", "cont_batching", "--cont-batching", "bool", "", (), ("-cb",), "--cont-batching"),
    Knob("Server", "use_embedding", "embedding", "--embedding", "bool", "", (), ("--embeddings",), "--embedding"),
    Knob("Server", "use_reuse_port", "reuse_port", "--reuse-port", "bool", "", (), (), "--reuse-port"),
)

# Derived maps
RAW_FLAG_MAP: dict[str, Knob] = {}
for _knob in KNOBS:
    RAW_FLAG_MAP[_knob.flag] = _knob
    for _alias in _knob.aliases:
        RAW_FLAG_MAP[_alias] = _knob

LLAMA_CPP_SERVER = Path(r"D:\llama.cpp\build\bin\llama-server.exe")
TURBOQUANT_SERVER = Path(r"D:\llama-cpp-turboquant\build\bin\llama-server.exe")
AI_STUFF = Path(r"D:\AI stuff")
SLOTS_DIR = Path(r"D:\slots")
GEMMA4_CHAT_TEMPLATE = Path(r"D:\llama.cpp\models\templates\google-gemma-4-31B-it.jinja")


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / "llama_launcher"
    path.mkdir(parents=True, exist_ok=True)
    return path


def profiles_path() -> Path:
    return config_dir() / "profiles.json"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    defaults: dict[str, Any] = {
        "hf_token": "",
        "download_dir": str(AI_STUFF),
    }
    if not path.is_file():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(data, dict):
        return defaults
    out = dict(defaults)
    if isinstance(data.get("hf_token"), str):
        out["hf_token"] = data["hf_token"]
    if isinstance(data.get("download_dir"), str) and data["download_dir"].strip():
        out["download_dir"] = data["download_dir"].strip()
    return out


def save_settings(settings: dict[str, Any]) -> None:
    path = settings_path()
    payload = {
        "hf_token": str(settings.get("hf_token") or ""),
        "download_dir": str(settings.get("download_dir") or AI_STUFF),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_stdio() -> None:
    """pythonw sets stdout/stderr to None; tqdm/HF logging then crash on .write."""
    # Prefer classic HTTP downloads (MB/s progress) over Xet reconstruction bars.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_DISABLE_XET = True
    except Exception:  # noqa: BLE001
        pass

    class _NullWriter:
        encoding = "utf-8"

        def write(self, s: object) -> int:
            if s is None:
                return 0
            if isinstance(s, (bytes, bytearray)):
                return len(s)
            return len(str(s))

        def flush(self) -> None:
            return None

        def isatty(self) -> bool:
            return False

        def fileno(self) -> int:
            raise OSError("no fileno")

        def readable(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def close(self) -> None:
            return None

        @property
        def closed(self) -> bool:
            return False

    if sys.stdout is None:
        sys.stdout = _NullWriter()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullWriter()  # type: ignore[assignment]


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{n} B"


# Approximate bytes/element for llama.cpp KV cache types (ggml type_size/blck_size).
CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4.0,
    "fp32": 4.0,
    "f16": 2.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,  # 1.0625
    "q5_1": 22 / 32,
    "q5_0": 18 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,  # 0.5625
    "q6_0": 22 / 32,  # approx
    "iq4_nl": 0.55,
    "iq4_xs": 0.5,
    # turboquant branch (approx — treat as lower-bit compressed KV)
    "turbo4": 0.55,
    "turbo3": 0.42,
    "turbo2": 0.30,
}


@dataclass(frozen=True)
class GgufModelInfo:
    path: str
    architecture: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    n_embd_head_k: int
    n_embd_head_v: int
    train_ctx: int | None = None
    sliding_window: int | None = None
    swa_pattern: tuple[bool, ...] = ()
    n_embd_head_k_swa: int | None = None
    n_embd_head_v_swa: int | None = None
    shared_kv_layers: int = 0


@dataclass(frozen=True)
class KvModelConfig:
    info: GgufModelInfo
    cache_type_k: str
    cache_type_v: str
    n_parallel: int = 1
    swa_full: bool = False

    def _type_bytes(self) -> tuple[float, float]:
        bk = CACHE_TYPE_BYTES.get(self.cache_type_k.lower().strip(), 2.0)
        bv = CACHE_TYPE_BYTES.get(self.cache_type_v.lower().strip(), 2.0)
        return bk, bv

    @property
    def bytes_per_token(self) -> float:
        """Marginal bytes of KV when increasing ctx by 1 (global layers only for SWA models)."""
        n_ctx = max(2, self.info.train_ctx or 8192)
        return max(0.0, self.kv_bytes(n_ctx) - self.kv_bytes(n_ctx - 1))

    def kv_bytes(self, n_ctx: int) -> float:
        """Estimate KV cache bytes at a given context length (ISWA / shared-KV aware)."""
        n_ctx = max(0, int(n_ctx))
        info = self.info
        bk, bv = self._type_bytes()
        n_par = max(1, self.n_parallel)
        n_kv = max(1, info.n_head_kv)

        if info.swa_pattern and info.sliding_window:
            pattern = info.swa_pattern
            if len(pattern) < info.n_layer:
                pattern = pattern + (False,) * (info.n_layer - len(pattern))
            # Final shared_kv_layers reuse KV from earlier same-type layers (no extra alloc).
            n_own = max(0, info.n_layer - max(0, info.shared_kv_layers))
            dk_swa = info.n_embd_head_k_swa or info.n_embd_head_k
            dv_swa = info.n_embd_head_v_swa or info.n_embd_head_v
            total = 0.0
            for i in range(n_own):
                is_swa = bool(pattern[i])
                if is_swa and not self.swa_full:
                    tokens = min(n_ctx, info.sliding_window)
                    dk, dv = dk_swa, dv_swa
                else:
                    tokens = n_ctx
                    dk, dv = info.n_embd_head_k, info.n_embd_head_v
                total += tokens * n_kv * (dk * bk + dv * bv)
            return total * n_par

        # Dense full-context KV (llama / qwen / etc.)
        per_token = n_kv * (info.n_embd_head_k * bk + info.n_embd_head_v * bv) * info.n_layer
        return per_token * n_ctx * n_par

    def swa_summary(self) -> str:
        info = self.info
        if not info.swa_pattern or not info.sliding_window:
            return "full-ctx KV"
        n_own = max(0, info.n_layer - max(0, info.shared_kv_layers))
        pattern = info.swa_pattern[:n_own]
        n_swa = sum(1 for x in pattern if x)
        n_full = n_own - n_swa
        shared = info.shared_kv_layers
        if self.swa_full:
            return f"SWA disabled (--swa-full) · {n_own} owning / {shared} shared layers"
        return (
            f"SWA window {info.sliding_window} · "
            f"{n_swa} SWA + {n_full} global owning layers · {shared} shared"
        )


_GGUF_META_CACHE: dict[str, tuple[float, GgufModelInfo]] = {}


def _gguf_read_string(fh) -> str:
    (n,) = struct.unpack("<Q", fh.read(8))
    return fh.read(n).decode("utf-8", errors="replace")


def _gguf_read_value(fh, vtype: int) -> Any:
    if vtype == 0:
        return struct.unpack("<B", fh.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", fh.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", fh.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", fh.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", fh.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", fh.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", fh.read(4))[0]
    if vtype == 7:
        return struct.unpack("<?", fh.read(1))[0]
    if vtype == 8:
        return _gguf_read_string(fh)
    if vtype == 10:
        return struct.unpack("<Q", fh.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", fh.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", fh.read(8))[0]
    if vtype == 9:
        (etype,) = struct.unpack("<I", fh.read(4))
        (n,) = struct.unpack("<Q", fh.read(8))
        return [_gguf_read_value(fh, etype) for _ in range(n)]
    raise ValueError(f"Unsupported GGUF value type {vtype}")


def read_gguf_model_info(path: str | Path) -> GgufModelInfo:
    """Read architecture hyperparameters from a GGUF header (metadata only)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model not found: {p}")
    mtime = p.stat().st_mtime
    cached = _GGUF_META_CACHE.get(str(p))
    if cached and cached[0] == mtime:
        return cached[1]

    with p.open("rb") as fh:
        magic, version = struct.unpack("<II", fh.read(8))
        if magic != 0x46554747:
            raise ValueError("Not a GGUF file")
        if version < 2:
            raise ValueError(f"Unsupported GGUF version {version}")
        _tensor_count, meta_count = struct.unpack("<QQ", fh.read(16))
        meta: dict[str, Any] = {}
        for _ in range(meta_count):
            key = _gguf_read_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = _gguf_read_value(fh, vtype)

    arch = str(meta.get("general.architecture") or "")
    if not arch:
        raise ValueError("GGUF missing general.architecture")

    def _need(suffix: str) -> Any:
        key = f"{arch}.{suffix}"
        if key not in meta:
            raise ValueError(f"GGUF missing {key}")
        return meta[key]

    def _as_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        if isinstance(value, list):
            return int(value[0]) if value else default
        return int(value)

    n_layer = _as_int(_need("block_count"))
    n_embd = _as_int(_need("embedding_length"))
    n_head = _as_int(meta.get(f"{arch}.attention.head_count"), 0)
    if n_head <= 0:
        raise ValueError("GGUF missing attention.head_count")
    n_head_kv = _as_int(meta.get(f"{arch}.attention.head_count_kv"), n_head)

    key_len = meta.get(f"{arch}.attention.key_length")
    val_len = meta.get(f"{arch}.attention.value_length")
    n_embd_head_k = _as_int(key_len, max(1, n_embd // n_head))
    n_embd_head_v = _as_int(val_len, n_embd_head_k)

    train_ctx_raw = meta.get(f"{arch}.context_length")
    train_ctx = _as_int(train_ctx_raw) if train_ctx_raw is not None else None

    sliding_window = meta.get(f"{arch}.attention.sliding_window")
    sliding_window_i = _as_int(sliding_window) if sliding_window is not None else None
    pattern_raw = meta.get(f"{arch}.attention.sliding_window_pattern")
    swa_pattern: tuple[bool, ...] = ()
    if isinstance(pattern_raw, list):
        swa_pattern = tuple(bool(x) for x in pattern_raw)

    key_len_swa = meta.get(f"{arch}.attention.key_length_swa")
    val_len_swa = meta.get(f"{arch}.attention.value_length_swa")
    n_embd_head_k_swa = _as_int(key_len_swa) if key_len_swa is not None else None
    n_embd_head_v_swa = _as_int(val_len_swa) if val_len_swa is not None else None
    shared_kv_layers = _as_int(meta.get(f"{arch}.attention.shared_kv_layers"), 0)

    info = GgufModelInfo(
        path=str(p),
        architecture=arch,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_embd_head_k=n_embd_head_k,
        n_embd_head_v=n_embd_head_v,
        train_ctx=train_ctx,
        sliding_window=sliding_window_i,
        swa_pattern=swa_pattern,
        n_embd_head_k_swa=n_embd_head_k_swa,
        n_embd_head_v_swa=n_embd_head_v_swa,
        shared_kv_layers=shared_kv_layers,
    )
    _GGUF_META_CACHE[str(p)] = (mtime, info)
    return info


def parse_positive_int(text: str, default: int = 0) -> int:
    text = (text or "").strip().replace(",", "")
    if not text:
        return default
    try:
        return max(0, int(float(text)))
    except ValueError:
        return default


def forecast_ctx_ladder(current_ctx: int) -> list[int]:
    """Ctx sizes to show: current + common rungs at/above it."""
    ladder = [4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 131072, 163840, 196608, 262144]
    out = {max(1, current_ctx)}
    for n in ladder:
        if n >= current_ctx * 0.5:  # nearby below + all above
            out.add(n)
    return sorted(out)


def combo_down_arrow_path() -> Path:
    """Light chevron for dark-theme QComboBox (default arrow vanishes under QSS)."""
    path = config_dir() / "combo_down_arrow.png"
    if path.exists():
        return path
    pm = QPixmap(12, 8)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#c5cad3"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPolygon(QPolygon([QPoint(1, 1), QPoint(11, 1), QPoint(6, 7)]))
    painter.end()
    pm.save(str(path))
    return path


def build_stylesheet() -> str:
    arrow = combo_down_arrow_path().resolve().as_posix()
    return DARK_QSS.replace("{{DOWN_ARROW}}", arrow)


def optional_flag_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for knob in KNOBS:
        out[knob.use_key] = False
        if knob.kind != "bool":
            out[knob.value_key] = knob.default
    return out


def new_profile(name: str = "New Profile") -> dict[str, Any]:
    profile: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "binary": str(LLAMA_CPP_SERVER),
        "model": "",
        "ngl": "999",
        "threads": "8",
        "ctx_size": "8192",
        "host": "0.0.0.0",
        "port": "11434",
        "cache_type_k": "",
        "cache_type_v": "",
        "jinja": False,
        "no_mmap": False,
        "mlock": False,
        "raw_args": "",
    }
    profile.update(optional_flag_defaults())
    return profile


def promote_known_flags_from_raw(profile: dict[str, Any]) -> dict[str, Any]:
    """Move known optional flags out of raw_args into structured fields."""
    tokens = split_args(profile.get("raw_args", ""))
    kept: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        knob = RAW_FLAG_MAP.get(tok)
        if knob is not None:
            if knob.kind == "bool":
                if not profile.get(knob.use_key):
                    profile[knob.use_key] = True
                i += 1
                continue
            if i + 1 < len(tokens):
                if not profile.get(knob.use_key):
                    profile[knob.use_key] = True
                    profile[knob.value_key] = tokens[i + 1]
                i += 2
                continue
        kept.append(tok)
        i += 1
    if kept:
        profile["raw_args"] = (
            subprocess.list2cmdline(kept)
            if sys.platform == "win32"
            else " ".join(shlex.quote(t) for t in kept)
        )
    else:
        profile["raw_args"] = ""
    return profile


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    base = new_profile(profile.get("name", "Untitled"))
    if profile.get("id"):
        base["id"] = profile["id"]
    base.update({k: v for k, v in profile.items() if v is not None})
    for knob in KNOBS:
        base.setdefault(knob.use_key, False)
        if knob.kind != "bool":
            base.setdefault(knob.value_key, knob.default)
    if not base.get("_flags_promoted") or int(base.get("_schema_seen", 0) or 0) < PROFILES_SCHEMA:
        promote_known_flags_from_raw(base)
        base["_flags_promoted"] = True
        base["_schema_seen"] = PROFILES_SCHEMA
    return base


def default_seed_profiles() -> list[dict[str, Any]]:
    """Windows-adapted profiles from D:\\commands.txt (only entries with present GGUFs)."""
    cpp = str(LLAMA_CPP_SERVER)
    tq = str(TURBOQUANT_SERVER)
    m = lambda name: str(AI_STUFF / name)  # noqa: E731
    slots = str(SLOTS_DIR)

    seeds: list[dict[str, Any]] = [
        {
            "name": "gemma 4 E2B v2",
            "binary": cpp,
            "model": m("gemma-4-E2B-it-Q4_K_M.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "4096",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "128",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "gemma 4 E2B high context",
            "binary": cpp,
            "model": m("gemma-4-E2B-it-Q4_K_M.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Microsoft Model",
            "binary": cpp,
            "model": m("microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "57344",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Qwen3-Coder-30B short ctx",
            "binary": cpp,
            "model": m("Qwen3-Coder-30B-A3B-Q2_K_XL.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "4096",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Gemma 4 E4B",
            "binary": cpp,
            "model": m("gemma-4-E4B-it-Q4_K_M.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_reasoning": True,
            "reasoning": "auto",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "QWEN 3 30B Coder 10-15t/s",
            "binary": cpp,
            "model": m("Qwen3-Coder-30B-A3B-Q2_K_XL.gguf"),
            "ngl": "999",
            "threads": "8",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "jinja": True,
            "no_mmap": True,
            "use_ot": True,
            "ot": ".ffn_.*_exps.=CPU",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_slots": True,
            "use_slot_save_path": True,
            "slot_save_path": slots,
            "use_kv_unified": True,
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "QWEN 3 30B experimental fit",
            "binary": cpp,
            "model": m("Qwen3-Coder-30B-A3B-Q2_K_XL.gguf"),
            "ngl": "",
            "threads": "8",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "jinja": True,
            "no_mmap": True,
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_fit": True,
            "fit": "on",
            "use_no_warmup": True,
            "use_slots": True,
            "use_slot_save_path": True,
            "slot_save_path": slots,
            "use_kv_unified": True,
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "QWEN 3 30B faster lower ctx",
            "binary": cpp,
            "model": m("Qwen3-Coder-30B-A3B-Q2_K_XL.gguf"),
            "ngl": "999",
            "threads": "8",
            "ctx_size": "32768",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "jinja": True,
            "no_mmap": True,
            "use_ot": True,
            "ot": ".ffn_.*_exps.=CPU",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "2048",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Qwen Coder 30B no-OOM",
            "binary": cpp,
            "model": m("Qwen3-Coder-30B-A3B-Q2_K_XL.gguf"),
            "ngl": "999",
            "threads": "8",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "jinja": True,
            "no_mmap": True,
            "mlock": True,
            "use_ot": True,
            "ot": ".ffn_.*_exps.=CPU",
            "use_batch": True,
            "batch": "512",
            "use_ubatch": True,
            "ubatch": "512",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_reasoning_budget": True,
            "reasoning_budget": "0",
            "use_reasoning": True,
            "reasoning": "off",
            "use_np": True,
            "np": "1",
            "use_cache_reuse": True,
            "cache_reuse": "256",
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Gemma 4 E2B CodeX Distilled",
            "binary": cpp,
            "model": m("Gemma-4-e2b-CodeX-Distill-v1.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "jinja": True,
            "use_batch": True,
            "batch": "4096",
            "use_ubatch": True,
            "ubatch": "1024",
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_np": True,
            "np": "1",
            "use_no_context_shift": True,
            "use_chat_template_kwargs": True,
            "chat_template_kwargs": '{"enable_thinking":true}',
            "use_mmproj": True,
            "mmproj": m("gemma-4-e2b-it.BF16-mmproj.gguf"),
            "raw_args": "",
            "_flags_promoted": True,
        },
        {
            "name": "Gemma 4 E4B no cache quant",
            "binary": cpp,
            "model": m("gemma-4-E4B-it-Q4_K_M.gguf"),
            "ngl": "999",
            "threads": "6",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "jinja": True,
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_reasoning": True,
            "reasoning": "auto",
            "raw_args": "--no-context-shift",
            "_flags_promoted": True,
        },
        {
            "name": "Gemma 4 E4B Coder",
            "binary": cpp,
            "model": m("gemma-4-E4B-CODER.Q4_K_M.gguf"),
            "ngl": "-1",
            "threads": "6",
            "ctx_size": "131072",
            "host": "0.0.0.0",
            "port": "11434",
            "jinja": True,
            "use_np": True,
            "np": "1",
            "use_flash_attn": True,
            "flash_attn": "on",
            "use_reasoning": True,
            "reasoning": "auto",
            "raw_args": "--no-context-shift",
            "_flags_promoted": True,
        },
        {
            "name": "Qwen 3.6 35B TurboQuant",
            "binary": tq,
            "model": m("Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf"),
            "ngl": "999",
            "threads": "8",
            "ctx_size": "262144",
            "host": "0.0.0.0",
            "port": "11434",
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo3",
            "jinja": True,
            "no_mmap": True,
            "mlock": True,
            "use_n_cpu_moe": True,
            "n_cpu_moe": "35",
            "raw_args": "",
            "_flags_promoted": True,
        },
    ]

    out: list[dict[str, Any]] = []
    for seed in seeds:
        model_path = Path(seed["model"])
        if not model_path.is_file():
            continue
        if "CodeX" in seed["name"]:
            mmproj = AI_STUFF / "gemma-4-e2b-it.BF16-mmproj.gguf"
            if not mmproj.is_file():
                continue
        profile = new_profile(seed["name"])
        profile.update(seed)
        profile["id"] = str(uuid.uuid4())
        # Gemma 4 needs the official tool-calling template so Web UI can parse tool_calls
        name_l = profile["name"].lower()
        model_l = str(profile.get("model", "")).lower()
        if ("gemma" in name_l or "gemma" in model_l) and GEMMA4_CHAT_TEMPLATE.is_file():
            profile["jinja"] = True
            profile["use_chat_template_file"] = True
            profile["chat_template_file"] = str(GEMMA4_CHAT_TEMPLATE)
        out.append(normalize_profile(profile))
    return out


def load_profiles() -> list[dict[str, Any]]:
    path = profiles_path()
    if not path.exists():
        profiles = default_seed_profiles()
        save_profiles(profiles)
        return profiles
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    profiles = data.get("profiles", [])
    if not isinstance(profiles, list):
        profiles = []
    normalized = [normalize_profile(p) for p in profiles]
    # Persist schema upgrade / flag promotion once
    if data.get("schema") != PROFILES_SCHEMA or any(
        True for p in profiles if not p.get("_flags_promoted")
    ):
        save_profiles(normalized)
    return normalized


def save_profiles(profiles: list[dict[str, Any]]) -> None:
    path = profiles_path()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"schema": PROFILES_SCHEMA, "profiles": profiles},
            fh,
            indent=2,
            ensure_ascii=False,
        )


def split_args(raw: str) -> list[str]:
    """Split a raw argument string into argv tokens without using a shell.

    Uses Windows-oriented shlex rules, then strips surrounding quotes so values
    like -ot ".ffn_.*_exps.=CPU" and paths with spaces stay intact as single tokens.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    tokens = shlex.split(raw, posix=False)
    cleaned: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
            token = token[1:-1]
        cleaned.append(token)
    return cleaned


def quote_preview(arg: str) -> str:
    if not arg:
        return '""'
    if sys.platform == "win32":
        if any(ch in arg for ch in ' \t"&|^<>'):
            return '"' + arg.replace('"', '\\"') + '"'
        return arg
    return shlex.quote(arg)


def build_argv(profile: dict[str, Any]) -> list[str]:
    binary = (profile.get("binary") or "").strip()
    if not binary:
        raise ValueError("Binary path is empty.")
    argv = [binary]

    model = (profile.get("model") or "").strip()
    if model:
        argv.extend(["-m", model])

    ngl = (profile.get("ngl") or "").strip()
    if ngl:
        argv.extend(["-ngl", ngl])

    threads = (profile.get("threads") or "").strip()
    if threads:
        argv.extend(["-t", threads])

    ctx = (profile.get("ctx_size") or "").strip()
    if ctx:
        argv.extend(["-c", ctx])

    host = (profile.get("host") or "").strip()
    if host:
        argv.extend(["--host", host])

    port = (profile.get("port") or "").strip()
    if port:
        argv.extend(["--port", port])

    ctk = (profile.get("cache_type_k") or "").strip()
    if ctk:
        argv.extend(["--cache-type-k", ctk])

    ctv = (profile.get("cache_type_v") or "").strip()
    if ctv:
        argv.extend(["--cache-type-v", ctv])

    if profile.get("jinja"):
        argv.append("--jinja")
    if profile.get("no_mmap"):
        argv.append("--no-mmap")
    if profile.get("mlock"):
        argv.append("--mlock")

    for knob in KNOBS:
        if not profile.get(knob.use_key):
            continue
        if knob.kind == "bool":
            argv.append(knob.flag)
            continue
        val = (profile.get(knob.value_key) or "").strip()
        if val:
            argv.extend([knob.flag, val])

    argv.extend(split_args(profile.get("raw_args", "")))
    return argv


def resolve_bench_binary(server_binary: str | Path) -> Path:
    """Map a llama-server path to the sibling llama-bench binary."""
    path = Path(server_binary)
    name = path.name
    lower = name.lower()
    if "llama-server" in lower:
        idx = lower.find("llama-server")
        bench_name = name[:idx] + "llama-bench" + name[idx + len("llama-server") :]
        return path.with_name(bench_name)
    suffix = ".exe" if path.suffix.lower() == ".exe" else ""
    return path.with_name(f"llama-bench{suffix}")


def build_bench_argv(profile: dict[str, Any]) -> list[str]:
    """Build llama-bench argv from profile fields that bench actually supports."""
    server_bin = (profile.get("binary") or "").strip()
    if not server_bin:
        raise ValueError("Binary path is empty.")
    bench = resolve_bench_binary(server_bin)
    argv = [str(bench), "--progress", "-o", "json"]

    model = (profile.get("model") or "").strip()
    if model:
        argv.extend(["-m", model])

    ngl = (profile.get("ngl") or "").strip()
    if ngl:
        argv.extend(["-ngl", ngl])

    threads = (profile.get("threads") or "").strip()
    if threads:
        argv.extend(["-t", threads])

    ctk = (profile.get("cache_type_k") or "").strip()
    if ctk:
        argv.extend(["-ctk", ctk])

    ctv = (profile.get("cache_type_v") or "").strip()
    if ctv:
        argv.extend(["-ctv", ctv])

    if profile.get("no_mmap"):
        argv.extend(["-mmp", "0"])

    # Map optional knobs that have llama-bench equivalents
    if profile.get("use_batch"):
        val = (profile.get("batch") or "").strip()
        if val:
            argv.extend(["-b", val])
    if profile.get("use_ubatch"):
        val = (profile.get("ubatch") or "").strip()
        if val:
            argv.extend(["-ub", val])
    if profile.get("use_flash_attn"):
        val = (profile.get("flash_attn") or "").strip()
        if val:
            argv.extend(["-fa", val])
    if profile.get("use_n_cpu_moe"):
        val = (profile.get("n_cpu_moe") or "").strip()
        if val:
            argv.extend(["-ncmoe", val])
    if profile.get("use_ot"):
        val = (profile.get("ot") or "").strip()
        if val:
            argv.extend(["-ot", val])
    if profile.get("use_no_kv_offload"):
        argv.extend(["-nkvo", "1"])
    if profile.get("use_no_warmup"):
        argv.append("--no-warmup")
    if profile.get("use_embedding"):
        argv.extend(["-embd", "1"])

    return argv


def extract_json_array(text: str) -> Any | None:
    """Find the last top-level JSON array in mixed process output."""
    decoder = json.JSONDecoder()
    last: Any | None = None
    i = 0
    while True:
        start = text.find("[", i)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if isinstance(value, list):
            last = value
        i = end
    return last


def format_bench_test_label(row: dict[str, Any]) -> str:
    n_prompt = int(row.get("n_prompt") or 0)
    n_gen = int(row.get("n_gen") or 0)
    n_depth = int(row.get("n_depth") or 0)
    if n_prompt and n_gen:
        label = f"pg {n_prompt},{n_gen}"
    elif n_prompt:
        label = f"pp {n_prompt}"
    elif n_gen:
        label = f"tg {n_gen}"
    else:
        label = "—"
    if n_depth:
        label += f" @ d{n_depth}"
    return label


def format_command_preview(argv: list[str]) -> str:
    return " ".join(quote_preview(a) for a in argv)


@dataclass
class ResourceSnapshot:
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_percent: float = 0.0
    gpu_util: float = 0.0
    gpu_name: str = ""
    proc_ram_mb: float | None = None
    proc_vram_mb: float | None = None
    error: str = ""


def _nvidia_smi_query(query: str) -> list[str]:
    try:
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            text=True,
            timeout=2.5,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def sample_resources(server_pid: int | None = None) -> ResourceSnapshot:
    snap = ResourceSnapshot()
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap.ram_used_gb = (vm.total - vm.available) / (1024**3)
        snap.ram_total_gb = vm.total / (1024**3)
        snap.ram_percent = float(vm.percent)
        if server_pid:
            try:
                proc = psutil.Process(server_pid)
                # Include children (DLL hosts shouldn't matter, but be complete)
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.Error, OSError):
                        pass
                snap.proc_ram_mb = rss / (1024**2)
            except (psutil.Error, OSError):
                snap.proc_ram_mb = None
    except Exception as exc:  # noqa: BLE001
        snap.error = f"RAM: {exc}"

    rows = _nvidia_smi_query("name,memory.used,memory.total,utilization.gpu")
    if rows:
        parts = [p.strip() for p in rows[0].split(",")]
        if len(parts) >= 4:
            try:
                snap.gpu_name = parts[0]
                snap.vram_used_mb = float(parts[1])
                snap.vram_total_mb = float(parts[2])
                snap.gpu_util = float(parts[3])
                if snap.vram_total_mb > 0:
                    snap.vram_percent = 100.0 * snap.vram_used_mb / snap.vram_total_mb
            except ValueError:
                snap.error = (snap.error + " | " if snap.error else "") + "VRAM parse failed"
    else:
        snap.error = (snap.error + " | " if snap.error else "") + "nvidia-smi unavailable"

    if server_pid:
        try:
            flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                text=True,
                timeout=2.5,
            )
            total_proc = 0.0
            found = False
            for line in out.splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) < 2:
                    continue
                try:
                    pid = int(cols[0])
                    used = float(cols[1])
                except ValueError:
                    continue
                if pid == int(server_pid):
                    total_proc += used
                    found = True
            if found:
                snap.proc_vram_mb = total_proc
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return snap


def cuda_bin_dir() -> Path | None:
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidate = Path(cuda_path) / "bin"
        if candidate.is_dir():
            return candidate
    default = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin")
    return default if default.is_dir() else None


def kill_process_tree(pid: int) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=False,
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
        )
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def build_child_env(binary_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    path_parts = [str(binary_dir)]
    cuda = cuda_bin_dir()
    if cuda:
        path_parts.append(str(cuda))
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_parts + ([existing] if existing else []))
    return env


class _StreamReader(QThread):
    chunk = pyqtSignal(str)
    exited = pyqtSignal(int)

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        super().__init__()
        self._proc = proc

    def run(self) -> None:
        assert self._proc.stdout is not None
        try:
            while True:
                data = self._proc.stdout.read(4096)
                if not data:
                    break
                self.chunk.emit(data.decode("utf-8", errors="replace"))
        finally:
            code = self._proc.wait()
            self.exited.emit(int(code if code is not None else -1))


class ServerRunner(QObject):
    """Windows-safe llama-server runner with live logs and process-tree stop."""

    output = pyqtSignal(str)
    started = pyqtSignal(int)
    finished = pyqtSignal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: _StreamReader | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        if self._proc is None or self._proc.poll() is not None:
            return None
        return int(self._proc.pid)

    def start(self, argv: list[str], cwd: str, env: dict[str, str]) -> None:
        if self.running:
            raise RuntimeError("Server already running")
        flags = 0
        if sys.platform == "win32":
            flags = CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        self._proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            bufsize=0,
        )
        self._reader = _StreamReader(self._proc)
        self._reader.chunk.connect(self.output.emit)
        self._reader.exited.connect(self._on_exited)
        self._reader.start()
        self.started.emit(int(self._proc.pid))

    def stop(self) -> None:
        if self._proc is None:
            return
        pid = int(self._proc.pid)
        kill_process_tree(pid)
        try:
            self._proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            kill_process_tree(pid)
            try:
                self._proc.kill()
            except OSError:
                pass
        if self._reader is not None:
            self._reader.wait(5000)
        self._proc = None
        self._reader = None

    def _on_exited(self, code: int) -> None:
        self._proc = None
        self._reader = None
        self.finished.emit(code)


DARK_QSS = """
QWidget {
    background-color: #1e1f22;
    color: #e8eaed;
    font-size: 13px;
}
QMainWindow, QSplitter::handle {
    background-color: #1e1f22;
}
QListWidget {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
    border-radius: 6px;
    padding: 4px;
}
QListWidget::item {
    padding: 8px 10px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #3d5a80;
}
QMenuBar {
    background-color: #1e1f22;
    color: #e8eaed;
    spacing: 4px;
}
QMenuBar::item:selected {
    background-color: #3c3f44;
}
QMenu {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
}
QMenu::item:selected {
    background-color: #3d5a80;
}
QTableWidget {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
    border-radius: 4px;
    gridline-color: #3c3f44;
    selection-background-color: #3d5a80;
}
QHeaderView::section {
    background-color: #3c3f44;
    color: #e8eaed;
    padding: 6px;
    border: none;
    border-right: 1px solid #50545c;
}
QLineEdit, QPlainTextEdit {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
    border-radius: 4px;
    padding: 6px;
    selection-background-color: #3d5a80;
}
QComboBox {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
    border-radius: 4px;
    padding: 6px 28px 6px 8px;
    min-height: 18px;
}
QComboBox:editable {
    padding-right: 28px;
}
QComboBox:editable QLineEdit {
    background-color: transparent;
    border: none;
    padding: 0;
    selection-background-color: #3d5a80;
}
QComboBox::drop-down {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 26px;
    border: none;
    border-left: 1px solid #3c3f44;
    background-color: #3c3f44;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}
QComboBox::drop-down:hover {
    background-color: #4a4e56;
}
QComboBox::down-arrow {
    image: url({{DOWN_ARROW}});
    width: 12px;
    height: 8px;
}
QComboBox QAbstractItemView {
    background-color: #2b2d31;
    border: 1px solid #3c3f44;
    selection-background-color: #3d5a80;
    outline: 0;
}
QPlainTextEdit#logView {
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    background-color: #121316;
}
QPlainTextEdit#previewView {
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    background-color: #121316;
    color: #9aa0a6;
}
QGroupBox {
    border: 1px solid #3c3f44;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3c3f44;
    border: 1px solid #50545c;
    border-radius: 5px;
    padding: 7px 12px;
}
QPushButton:hover { background-color: #4a4e56; }
QPushButton:pressed { background-color: #2f3237; }
QPushButton:disabled { color: #7a7f87; background-color: #2a2c30; }
QPushButton#startBtn {
    background-color: #2d6a4f;
    border-color: #40916c;
    font-weight: 600;
}
QPushButton#startBtn:hover { background-color: #358f64; }
QPushButton#stopBtn {
    background-color: #9b2226;
    border-color: #ae2a2f;
    font-weight: 600;
}
QPushButton#stopBtn:hover { background-color: #bb2d33; }
QCheckBox { spacing: 8px; }
QLabel#statusLabel { font-weight: 600; padding: 4px 0; }
QLabel#resourceValue {
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
    color: #c5cad3;
}
QProgressBar {
    background-color: #121316;
    border: 1px solid #3c3f44;
    border-radius: 4px;
    text-align: center;
    height: 16px;
    color: #e8eaed;
}
QProgressBar#ramBar::chunk {
    background-color: #3d7ea6;
    border-radius: 3px;
}
QProgressBar#vramBar::chunk {
    background-color: #2d6a4f;
    border-radius: 3px;
}
QProgressBar#gpuBar::chunk {
    background-color: #b08900;
    border-radius: 3px;
}
QProgressBar[level="warn"]::chunk { background-color: #b5651d; }
QProgressBar[level="crit"]::chunk { background-color: #9b2226; }
QScrollBar:vertical {
    background: #1e1f22;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4a4e56;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def cleanup_hf_temp_files(local_dir: str | Path, filename: str) -> list[str]:
    """Remove incomplete / partial artifacts left by a cancelled hf_hub_download."""
    root = Path(local_dir)
    removed: list[str] = []
    if not root.is_dir():
        return removed

    base_name = Path(filename).name
    candidates: list[Path] = []

    # Process-unique temps next to the destination / under .cache
    for folder in (root, root / Path(filename).parent, root / ".cache"):
        if not folder.is_dir():
            continue
        try:
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                name = path.name
                if name.endswith(".incomplete"):
                    candidates.append(path)
                elif name.startswith(base_name) and name.endswith(".tmp"):
                    candidates.append(path)
        except OSError:
            continue

    # Never delete the completed target if it already existed before cancel —
    # only remove zero-length or obviously partial siblings named like the file
    # with a UUID incomplete suffix (handled above via .incomplete).

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        except OSError:
            continue
    return removed


class DownloadCancelled(Exception):
    """Raised from download progress hooks when the user cancels."""


class _HfSearchWorker(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, query: str, gguf_only: bool, token: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.query = query
        self.gguf_only = gguf_only
        self.token = token or None

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            kwargs: dict[str, Any] = {
                "search": self.query.strip() or None,
                "limit": 40,
                "sort": "downloads",
            }
            if self.gguf_only:
                kwargs["filter"] = "gguf"
            models = list(api.list_models(**kwargs))
            rows: list[dict[str, Any]] = []
            for m in models:
                rows.append(
                    {
                        "id": m.id,
                        "downloads": getattr(m, "downloads", None) or 0,
                        "likes": getattr(m, "likes", None) or 0,
                        "tags": list(getattr(m, "tags", None) or []),
                    }
                )
            self.finished_ok.emit(rows)
        except Exception as exc:  # noqa: BLE001 — surface any hub/network error
            self.failed.emit(str(exc))


class _HfFilesWorker(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, repo_id: str, gguf_only: bool, token: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.repo_id = repo_id
        self.gguf_only = gguf_only
        self.token = token or None

    def run(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            info = api.model_info(self.repo_id, files_metadata=True)
            rows: list[dict[str, Any]] = []
            for sibling in info.siblings or []:
                name = sibling.rfilename
                if self.gguf_only and not name.lower().endswith(".gguf"):
                    continue
                rows.append({"name": name, "size": getattr(sibling, "size", None)})
            rows.sort(key=lambda r: r["name"].lower())
            self.finished_ok.emit(rows)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _HfDownloadWorker(QThread):
    progress = pyqtSignal(int, int, str)  # received, total, status
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal(str)  # cleanup summary

    def __init__(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        token: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo_id = repo_id
        self.filename = filename
        self.local_dir = local_dir
        self.token = token or None
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def _snapshot_incomplete(self) -> set[str]:
        root = Path(self.local_dir)
        found: set[str] = set()
        if not root.is_dir():
            return found
        try:
            for path in root.rglob("*.incomplete"):
                if path.is_file():
                    found.add(str(path.resolve()))
        except OSError:
            pass
        return found

    def _cleanup_temps(self, before: set[str] | None = None) -> str:
        after = self._snapshot_incomplete()
        targets = after if before is None else after - before
        # Also catch any leftover incomplete paths under local_dir matching this download name
        if before is not None:
            for path in cleanup_hf_temp_files(self.local_dir, self.filename):
                targets.add(str(Path(path).resolve()) if Path(path).exists() else path)
        removed: list[str] = []
        for path_str in sorted(targets):
            path = Path(path_str)
            try:
                if path.is_file():
                    path.unlink()
                    removed.append(str(path))
            except OSError:
                continue
        # Second pass for anything the raise left before HF's finally ran
        for path_str in cleanup_hf_temp_files(self.local_dir, self.filename):
            if path_str not in removed:
                removed.append(path_str)
        if not removed:
            return "Cancelled. No leftover temp files found."
        if len(removed) == 1:
            return f"Cancelled. Removed temp file:\n{removed[0]}"
        return f"Cancelled. Removed {len(removed)} temp file(s)."

    def run(self) -> None:
        ensure_stdio()
        before = self._snapshot_incomplete()
        try:
            import huggingface_hub.constants as hf_constants
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import tqdm as hub_tqdm

            # Prefer classic HTTP progress (bytes + speed) over Xet "reconstructing"
            hf_constants.HF_HUB_DISABLE_XET = True

            worker = self

            class _ProgressBar(hub_tqdm):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    kwargs["disable"] = False
                    kwargs.setdefault("file", sys.stderr)
                    kwargs.setdefault("unit", "B")
                    kwargs.setdefault("unit_scale", True)
                    kwargs.setdefault("unit_divisor", 1024)
                    super().__init__(*args, **kwargs)
                    self._emit()

                def update(self, n: float | int = 1) -> bool | None:  # type: ignore[override]
                    if worker._cancel.is_set():
                        raise DownloadCancelled("Download cancelled by user")
                    result = super().update(n)
                    self._emit()
                    return result

                def _emit(self) -> None:
                    if worker._cancel.is_set():
                        raise DownloadCancelled("Download cancelled by user")
                    total = int(getattr(self, "total", 0) or 0)
                    current = int(getattr(self, "n", 0) or 0)
                    name = str(getattr(self, "desc", None) or worker.filename or "Downloading")
                    rate = None
                    try:
                        rate = self.format_dict.get("rate")
                    except Exception:  # noqa: BLE001
                        rate = None
                    if total > 0:
                        status = f"{name}: {format_bytes(current)} / {format_bytes(total)}"
                    else:
                        status = f"{name}: {format_bytes(current)}"
                    if rate:
                        status += f" · {format_bytes(rate)}/s"
                    worker.progress.emit(current, total, status)

            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=self.local_dir,
                token=self.token,
                tqdm_class=_ProgressBar,
            )
            # If cancel was requested but download already finished, keep the file.
            self.finished_ok.emit(str(Path(path).resolve()))
        except DownloadCancelled:
            self.cancelled.emit(self._cleanup_temps(before))
        except Exception as exc:  # noqa: BLE001
            if self._cancel.is_set():
                self.cancelled.emit(self._cleanup_temps(before))
            else:
                self.failed.emit(str(exc))


class HuggingFaceWindow(QDialog):
    """Separate window: search Hugging Face and download GGUF (or other) files."""

    model_path_chosen = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hugging Face — Search & Download")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(960, 640)
        self.setMinimumSize(720, 480)

        self._settings = load_settings()
        self._search_worker: _HfSearchWorker | None = None
        self._files_worker: _HfFilesWorker | None = None
        self._dl_worker: _HfDownloadWorker | None = None
        self._current_repo: str | None = None
        self._models: list[dict[str, Any]] = []
        self._files: list[dict[str, Any]] = []

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search models… e.g. gemma-4 Q4_K_M bartowski")
        self.search_edit.returnPressed.connect(self._start_search)
        self.gguf_only_chk = QCheckBox("GGUF only")
        self.gguf_only_chk.setChecked(True)
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self._start_search)
        search_row.addWidget(self.search_edit, stretch=1)
        search_row.addWidget(self.gguf_only_chk)
        search_row.addWidget(self.btn_search)
        root.addLayout(search_row)

        opts = QHBoxLayout()
        self.token_edit = QLineEdit(self._settings.get("hf_token", ""))
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("Optional HF token (gated models)")
        self.btn_save_token = QPushButton("Save prefs")
        self.btn_save_token.clicked.connect(self._save_prefs)
        self.dest_edit = QLineEdit(self._settings.get("download_dir", str(AI_STUFF)))
        self.btn_browse_dest = QPushButton("Browse…")
        self.btn_browse_dest.clicked.connect(self._browse_dest)
        opts.addWidget(QLabel("Token"))
        opts.addWidget(self.token_edit, stretch=2)
        opts.addWidget(QLabel("Download to"))
        opts.addWidget(self.dest_edit, stretch=3)
        opts.addWidget(self.btn_browse_dest)
        opts.addWidget(self.btn_save_token)
        root.addLayout(opts)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Models"))
        self.model_table = QTableWidget(0, 3)
        self.model_table.setHorizontalHeaderLabels(["Repo", "↓", "♥"])
        self.model_table.horizontalHeader().setStretchLastSection(False)
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.model_table.setColumnWidth(1, 72)
        self.model_table.setColumnWidth(2, 48)
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.itemSelectionChanged.connect(self._on_model_selected)
        left_l.addWidget(self.model_table)
        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addWidget(QLabel("Files"))
        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["File", "Size"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.setColumnWidth(1, 90)
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.file_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.file_table.verticalHeader().setVisible(False)
        right_l.addWidget(self.file_table)
        split.addWidget(right)

        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([380, 520])
        root.addWidget(split, stretch=1)

        actions = QHBoxLayout()
        self.btn_download = QPushButton("Download selected file")
        self.btn_download.setObjectName("startBtn")
        self.btn_download.clicked.connect(self._start_download)
        self.btn_cancel = QPushButton("Cancel download")
        self.btn_cancel.setObjectName("stopBtn")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_download)
        self.btn_use = QPushButton("Use in current profile")
        self.btn_use.setEnabled(False)
        self.btn_use.clicked.connect(self._use_in_profile)
        self._last_download: str | None = None
        actions.addWidget(self.btn_download)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_use)
        actions.addStretch(1)
        root.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        root.addWidget(self.progress)

        self.status = QLabel("Search for a model to begin.")
        self.status.setObjectName("resourceValue")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _token(self) -> str:
        return self.token_edit.text().strip()

    def _save_prefs(self) -> None:
        self._settings = {
            "hf_token": self._token(),
            "download_dir": self.dest_edit.text().strip() or str(AI_STUFF),
        }
        save_settings(self._settings)
        self.status.setText("Preferences saved.")

    def _browse_dest(self) -> None:
        start = self.dest_edit.text().strip() or str(AI_STUFF)
        path = QFileDialog.getExistingDirectory(self, "Download folder", start)
        if path:
            self.dest_edit.setText(path)

    def _set_busy(self, busy: bool, *, downloading: bool = False) -> None:
        self.btn_search.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        self.search_edit.setEnabled(not busy)
        self.btn_cancel.setEnabled(downloading)
        self.btn_browse_dest.setEnabled(not busy)

    def _start_search(self) -> None:
        if self._search_worker and self._search_worker.isRunning():
            return
        self._set_busy(True)
        self.status.setText("Searching Hugging Face…")
        self.model_table.setRowCount(0)
        self.file_table.setRowCount(0)
        self._models = []
        self._files = []
        self._current_repo = None
        worker = _HfSearchWorker(self.search_edit.text(), self.gguf_only_chk.isChecked(), self._token(), self)
        worker.finished_ok.connect(self._on_search_ok)
        worker.failed.connect(self._on_search_fail)
        worker.finished.connect(lambda: self._set_busy(False))
        self._search_worker = worker
        worker.start()

    def _on_search_ok(self, rows: list) -> None:
        self._models = rows
        self.model_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.model_table.setItem(i, 0, QTableWidgetItem(row["id"]))
            dl = QTableWidgetItem(f'{row["downloads"]:,}')
            dl.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.model_table.setItem(i, 1, dl)
            likes = QTableWidgetItem(str(row["likes"]))
            likes.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.model_table.setItem(i, 2, likes)
        self.status.setText(f"Found {len(rows)} model(s). Select one to list files.")

    def _on_search_fail(self, err: str) -> None:
        self.status.setText(f"Search failed: {err}")
        QMessageBox.warning(self, "Hugging Face", f"Search failed:\n{err}")

    def _on_model_selected(self) -> None:
        rows = self.model_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._models):
            return
        repo_id = self._models[idx]["id"]
        if self._files_worker and self._files_worker.isRunning():
            return
        self._current_repo = repo_id
        self.file_table.setRowCount(0)
        self._files = []
        self.status.setText(f"Listing files in {repo_id}…")
        self._set_busy(True)
        worker = _HfFilesWorker(repo_id, self.gguf_only_chk.isChecked(), self._token(), self)
        worker.finished_ok.connect(self._on_files_ok)
        worker.failed.connect(self._on_files_fail)
        worker.finished.connect(lambda: self._set_busy(False))
        self._files_worker = worker
        worker.start()

    def _on_files_ok(self, rows: list) -> None:
        self._files = rows
        self.file_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.file_table.setItem(i, 0, QTableWidgetItem(row["name"]))
            size_item = QTableWidgetItem(format_bytes(row.get("size")))
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.file_table.setItem(i, 1, size_item)
        repo = self._current_repo or ""
        self.status.setText(f"{len(rows)} file(s) in {repo}. Select one and download.")

    def _on_files_fail(self, err: str) -> None:
        self.status.setText(f"List files failed: {err}")
        QMessageBox.warning(self, "Hugging Face", f"Could not list files:\n{err}")

    def _selected_file(self) -> tuple[str, str] | None:
        if not self._current_repo:
            return None
        rows = self.file_table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._files):
            return None
        return self._current_repo, self._files[idx]["name"]

    def _start_download(self) -> None:
        picked = self._selected_file()
        if not picked:
            QMessageBox.information(self, "Hugging Face", "Select a model file first.")
            return
        if self._dl_worker and self._dl_worker.isRunning():
            return
        repo_id, filename = picked
        dest = self.dest_edit.text().strip() or str(AI_STUFF)
        dest_path = Path(dest)
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Hugging Face", f"Cannot create folder:\n{exc}")
            return
        self._save_prefs()
        self._set_busy(True, downloading=True)
        self.btn_use.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.status.setText(f"Downloading {filename}…")
        worker = _HfDownloadWorker(repo_id, filename, str(dest_path), self._token(), self)
        worker.progress.connect(self._on_dl_progress)
        worker.finished_ok.connect(self._on_dl_ok)
        worker.failed.connect(self._on_dl_fail)
        worker.cancelled.connect(self._on_dl_cancelled)
        worker.finished.connect(lambda: self._set_busy(False))
        self._dl_worker = worker
        worker.start()

    def _cancel_download(self) -> None:
        if self._dl_worker is None or not self._dl_worker.isRunning():
            return
        self.btn_cancel.setEnabled(False)
        self.status.setText("Cancelling download…")
        self._dl_worker.request_cancel()

    def _on_dl_progress(self, received: int, total: int, status: str) -> None:
        if total > 0:
            self.progress.setValue(min(1000, int(received / total * 1000)))
            pct = 100.0 * received / total
            self.progress.setFormat(f"{pct:.1f}%")
        else:
            self.progress.setValue(0)
            self.progress.setFormat(format_bytes(received))
        self.status.setText(status)

    def _on_dl_ok(self, path: str) -> None:
        self._last_download = path
        self.btn_use.setEnabled(True)
        self.progress.setValue(1000)
        self.progress.setFormat("100%")
        self.status.setText(f"Downloaded: {path}")

    def _on_dl_fail(self, err: str) -> None:
        self.progress.setFormat("%p%")
        self.status.setText(f"Download failed: {err}")
        QMessageBox.critical(self, "Hugging Face", f"Download failed:\n{err}")

    def _on_dl_cancelled(self, summary: str) -> None:
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.status.setText(summary)
        self.btn_use.setEnabled(False)
        self._last_download = None

    def _use_in_profile(self) -> None:
        if not self._last_download:
            return
        self.model_path_chosen.emit(self._last_download)


class BenchmarkWindow(QDialog):
    """Separate window: run llama-bench with the current profile and show results."""

    def __init__(self, profile: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        name = (profile.get("name") or "Untitled").strip() or "Untitled"
        self.setWindowTitle(f"Benchmark — {name}")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(980, 700)
        self.setMinimumSize(720, 480)

        self._profile = deepcopy(profile)
        self._log_buf = ""
        self._closing = False

        self.runner = ServerRunner(self)
        self.runner.output.connect(self._append_log)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)

        self._build_ui()
        QTimer.singleShot(0, self._start_bench)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        cmd_row = QHBoxLayout()
        self.status_label = QLabel("Preparing…")
        self.status_label.setObjectName("statusLabel")
        cmd_row.addWidget(self.status_label, stretch=1)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_bench)
        self.btn_copy = QPushButton("Copy results")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self._copy_results)
        self.btn_clear = QPushButton("Clear log")
        self.btn_clear.clicked.connect(lambda: self.log_view.clear())
        cmd_row.addWidget(self.btn_stop)
        cmd_row.addWidget(self.btn_copy)
        cmd_row.addWidget(self.btn_clear)
        root.addLayout(cmd_row)

        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setObjectName("previewView")
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setMaximumHeight(72)
        self.preview_edit.setFont(QFont("Consolas", 9))
        root.addWidget(self.preview_edit)

        split = QSplitter(Qt.Orientation.Vertical)

        results_panel = QWidget()
        results_l = QVBoxLayout(results_panel)
        results_l.setContentsMargins(0, 0, 0, 0)
        results_l.setSpacing(6)
        results_l.addWidget(QLabel("Results"))
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels(
            ["Test", "t/s", "±", "ngl", "threads", "batch", "backend", "model"]
        )
        hdr = self.results_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setMinimumHeight(120)
        results_l.addWidget(self.results_table)
        self.summary_label = QLabel("Waiting for llama-bench to finish…")
        self.summary_label.setObjectName("resourceValue")
        self.summary_label.setWordWrap(True)
        results_l.addWidget(self.summary_label)
        split.addWidget(results_panel)

        log_panel = QWidget()
        log_l = QVBoxLayout(log_panel)
        log_l.setContentsMargins(0, 0, 0, 0)
        log_l.setSpacing(6)
        log_l.addWidget(QLabel("Live log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(12000)
        self.log_view.setFont(QFont("Consolas", 10))
        log_l.addWidget(self.log_view, stretch=1)
        split.addWidget(log_panel)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([220, 420])
        root.addWidget(split, stretch=1)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self._log_buf += text
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        # Try to refresh results if a JSON array appeared mid-stream
        if '"avg_ts"' in text or text.rstrip().endswith("]"):
            self._try_parse_results(quiet=True)

    def _log_line(self, text: str) -> None:
        line = text if text.endswith("\n") else text + "\n"
        self._append_log(line)

    def _start_bench(self) -> None:
        try:
            argv = build_bench_argv(self._profile)
        except ValueError as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            self.status_label.setText("🔴 Failed")
            return

        binary = Path(argv[0])
        self.preview_edit.setPlainText(format_command_preview(argv))

        if not binary.is_file():
            msg = (
                f"llama-bench not found:\n{binary}\n\n"
                "It should sit next to your llama-server binary."
            )
            self._log_line(f"[launcher] {msg}")
            QMessageBox.critical(self, APP_NAME, msg)
            self.status_label.setText("🔴 Binary missing")
            return

        model = (self._profile.get("model") or "").strip()
        if model and not Path(model).is_file():
            msg = f"Model not found:\n{model}"
            self._log_line(f"[launcher] {msg}")
            QMessageBox.critical(self, APP_NAME, msg)
            self.status_label.setText("🔴 Model missing")
            return

        workdir = str(binary.parent)
        env = build_child_env(binary.parent)
        self._log_line(f"[launcher] Benchmark:\n{format_command_preview(argv)}")
        self._log_line(f"[launcher] cwd={workdir}")
        try:
            self.runner.start(argv, cwd=workdir, env=env)
        except OSError as exc:
            self._log_line(f"[launcher] Failed to start: {exc}")
            self.status_label.setText("🔴 Failed")
            self.btn_stop.setEnabled(False)

    def _stop_bench(self) -> None:
        if not self.runner.running:
            return
        pid = self.runner.pid
        self._log_line(f"[launcher] Stopping benchmark PID {pid}…")
        self.runner.stop()
        self.btn_stop.setEnabled(False)
        self.status_label.setText("🔴 Stopped")

    def _on_started(self, pid: int) -> None:
        self.btn_stop.setEnabled(True)
        self.status_label.setText(f"🟢 Running (PID: {pid})")
        self._log_line(f"[launcher] Running PID {pid}")

    def _on_finished(self, exit_code: int) -> None:
        self.btn_stop.setEnabled(False)
        if exit_code == 0:
            self.status_label.setText("🟢 Finished")
        else:
            self.status_label.setText(f"🔴 Exited ({exit_code})")
        self._log_line(f"[launcher] Process exited with code {exit_code}")
        self._try_parse_results(quiet=False)

    def _try_parse_results(self, quiet: bool = False) -> None:
        data = extract_json_array(self._log_buf)
        if not isinstance(data, list) or not data:
            if not quiet:
                self.summary_label.setText(
                    "No JSON results found in output. Check the log for errors."
                )
            return
        rows = [r for r in data if isinstance(r, dict)]
        if not rows:
            return
        self._fill_results(rows)

    def _fill_results(self, rows: list[dict[str, Any]]) -> None:
        self.results_table.setRowCount(len(rows))
        best_tg: tuple[float, str] | None = None
        best_pp: tuple[float, str] | None = None

        for i, row in enumerate(rows):
            test = format_bench_test_label(row)
            avg = float(row.get("avg_ts") or 0)
            std = float(row.get("stddev_ts") or 0)
            model = str(row.get("model_type") or row.get("model_filename") or "—")
            cells = [
                test,
                f"{avg:.2f}",
                f"{std:.2f}",
                str(row.get("n_gpu_layers", "—")),
                str(row.get("n_threads", "—")),
                str(row.get("n_batch", "—")),
                str(row.get("backends") or "—"),
                model,
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col in (1, 2, 3, 4, 5):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if col == 1:
                    item.setForeground(QColor("#9ec5fe"))
                self.results_table.setItem(i, col, item)

            if test.startswith("tg") and (best_tg is None or avg > best_tg[0]):
                best_tg = (avg, test)
            if test.startswith("pp") and (best_pp is None or avg > best_pp[0]):
                best_pp = (avg, test)

        parts: list[str] = [f"{len(rows)} result(s)"]
        if best_pp:
            parts.append(f"best PP {best_pp[1]} → {best_pp[0]:.2f} t/s")
        if best_tg:
            parts.append(f"best TG {best_tg[1]} → {best_tg[0]:.2f} t/s")
        gpu = str(rows[0].get("gpu_info") or "").strip()
        if gpu:
            parts.append(gpu)
        self.summary_label.setText(" · ".join(parts))
        self.btn_copy.setEnabled(True)

    def _copy_results(self) -> None:
        lines = ["\t".join(["Test", "t/s", "±", "ngl", "threads", "batch", "backend", "model"])]
        for r in range(self.results_table.rowCount()):
            cells = []
            for c in range(self.results_table.columnCount()):
                item = self.results_table.item(r, c)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))
        self._log_line("[launcher] Results copied to clipboard.")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.runner.running and not self._closing:
            reply = QMessageBox.question(
                self,
                APP_NAME,
                "Benchmark is still running. Stop it and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._closing = True
            self._stop_bench()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("llama-launcher")
        self.resize(1440, 820)

        self.profiles: list[dict[str, Any]] = load_profiles()
        self._loading = False
        self._current_id: str | None = None

        self.runner = ServerRunner(self)
        self.runner.output.connect(self._append_log)
        self.runner.started.connect(self._on_server_started)
        self.runner.finished.connect(self._on_server_finished)

        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(1000)
        self._resource_timer.timeout.connect(self._refresh_resources)

        self._hf_window: HuggingFaceWindow | None = None
        self._bench_windows: list[BenchmarkWindow] = []

        self._build_ui()
        self._build_menu()
        self._reload_list()
        if self.profiles:
            self.profile_list.setCurrentRow(0)
        self._update_running_ui(False)
        self._resource_timer.start()
        self._refresh_resources()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Sidebar ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Profiles"))

        self.profile_list = QListWidget()
        self.profile_list.currentItemChanged.connect(self._on_profile_selected)
        left_layout.addWidget(self.profile_list, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Delete")
        self.btn_add.clicked.connect(self._add_profile)
        self.btn_dup.clicked.connect(self._duplicate_profile)
        self.btn_del.clicked.connect(self._delete_profile)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_dup)
        btn_row.addWidget(self.btn_del)
        left_layout.addLayout(btn_row)

        self.btn_hf = QPushButton("Hugging Face…")
        self.btn_hf.clicked.connect(self._open_hf_window)
        left_layout.addWidget(self.btn_hf)

        splitter.addWidget(left)

        # --- Center: parameter editor ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        form_box = QGroupBox("Parameter Editor")
        form = QFormLayout(form_box)

        self.name_edit = QLineEdit()
        form.addRow("Profile name", self.name_edit)

        binary_row = QHBoxLayout()
        self.binary_edit = QLineEdit()
        btn_browse_bin = QPushButton("Browse…")
        btn_browse_bin.clicked.connect(self._browse_binary)
        binary_row.addWidget(self.binary_edit, stretch=1)
        binary_row.addWidget(btn_browse_bin)
        form.addRow("Binary", binary_row)

        model_row = QHBoxLayout()
        self.model_edit = QLineEdit()
        btn_browse_model = QPushButton("Browse…")
        btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit, stretch=1)
        model_row.addWidget(btn_browse_model)
        form.addRow("Model (-m)", model_row)

        self.ngl_edit = QLineEdit()
        self.threads_edit = QLineEdit()
        self.ctx_edit = QComboBox()
        self.ctx_edit.setObjectName("ctxCombo")
        self.ctx_edit.setEditable(True)
        self.ctx_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.ctx_edit.addItems(CTX_PRESETS)
        self.ctx_edit.setCurrentText("8192")
        self.ctx_edit.setMinimumContentsLength(10)
        self.ctx_edit.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.host_edit = QLineEdit()
        self.port_edit = QLineEdit()
        self.ctk_edit = QLineEdit()
        self.ctv_edit = QLineEdit()
        form.addRow("GPU layers (-ngl)", self.ngl_edit)
        form.addRow("Threads (-t)", self.threads_edit)
        form.addRow("Context (-c)  [select ▼ or type]", self.ctx_edit)
        form.addRow("Host", self.host_edit)
        form.addRow("Port", self.port_edit)
        form.addRow("cache-type-k", self.ctk_edit)
        form.addRow("cache-type-v", self.ctv_edit)

        toggles = QHBoxLayout()
        self.jinja_chk = QCheckBox("--jinja")
        self.nommap_chk = QCheckBox("--no-mmap")
        self.mlock_chk = QCheckBox("--mlock")
        toggles.addWidget(self.jinja_chk)
        toggles.addWidget(self.nommap_chk)
        toggles.addWidget(self.mlock_chk)
        toggles.addStretch(1)
        form.addRow("Flags", toggles)

        opt_box = QGroupBox("Optional knobs (enable with checkbox)")
        opt_layout = QVBoxLayout(opt_box)
        self.knob_widgets: dict[str, tuple[QCheckBox, QWidget]] = {}

        groups: dict[str, QFormLayout] = {}
        for knob in KNOBS:
            if knob.group not in groups:
                group_box = QGroupBox(knob.group)
                group_form = QFormLayout(group_box)
                groups[knob.group] = group_form
                opt_layout.addWidget(group_box)
            form_layout = groups[knob.group]
            chk = QCheckBox(knob.ui_label)
            if knob.kind == "bool":
                edit: QWidget = QLabel("(flag only)")
                edit.setEnabled(False)
                chk.toggled.connect(lambda _on, e=edit: e.setEnabled(False))
            elif knob.kind == "combo":
                combo = QComboBox()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
                combo.addItems(knob.choices)
                combo.setCurrentText(knob.default)
                combo.setEnabled(False)
                chk.toggled.connect(combo.setEnabled)
                edit = combo
            else:
                line = QLineEdit(knob.default)
                line.setEnabled(False)
                if "path" in knob.value_key or knob.value_key in {"mmproj", "slot_save_path"}:
                    line.setPlaceholderText("path…")
                chk.toggled.connect(line.setEnabled)
                edit = line
            form_layout.addRow(chk, edit)
            self.knob_widgets[knob.use_key] = (chk, edit)

        self.raw_edit = QPlainTextEdit()
        self.raw_edit.setPlaceholderText(
            "Extra args only — anything not covered by fields above"
        )
        self.raw_edit.setMaximumBlockCount(0)
        self.raw_edit.setFixedHeight(90)
        form.addRow("Raw custom arguments", self.raw_edit)

        editor_wrap = QWidget()
        editor_layout = QVBoxLayout(editor_wrap)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(form_box)
        editor_layout.addWidget(opt_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(editor_wrap)
        center_layout.addWidget(scroll, stretch=1)

        preview_box = QGroupBox("Command preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setObjectName("previewView")
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setFixedHeight(72)
        preview_layout.addWidget(self.preview_edit)
        center_layout.addWidget(preview_box)

        # Live resource meters for tuning ctx / ngl / quant
        res_box = QGroupBox("Live resources")
        res_layout = QVBoxLayout(res_box)
        res_layout.setSpacing(6)

        self.ram_bar = QProgressBar()
        self.ram_bar.setObjectName("ramBar")
        self.ram_bar.setRange(0, 1000)
        self.ram_bar.setFormat("%p%")
        self.ram_label = QLabel("RAM: —")
        self.ram_label.setObjectName("resourceValue")
        res_layout.addWidget(self.ram_label)
        res_layout.addWidget(self.ram_bar)

        self.vram_bar = QProgressBar()
        self.vram_bar.setObjectName("vramBar")
        self.vram_bar.setRange(0, 1000)
        self.vram_bar.setFormat("%p%")
        self.vram_label = QLabel("VRAM: —")
        self.vram_label.setObjectName("resourceValue")
        res_layout.addWidget(self.vram_label)
        res_layout.addWidget(self.vram_bar)

        self.gpu_bar = QProgressBar()
        self.gpu_bar.setObjectName("gpuBar")
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setFormat("%p%")
        self.gpu_label = QLabel("GPU util: —")
        self.gpu_label.setObjectName("resourceValue")
        res_layout.addWidget(self.gpu_label)
        res_layout.addWidget(self.gpu_bar)

        self.proc_label = QLabel("Server process: —")
        self.proc_label.setObjectName("resourceValue")
        res_layout.addWidget(self.proc_label)
        center_layout.addWidget(res_box)

        controls = QHBoxLayout()
        self.btn_save = QPushButton("Save Profile")
        self.btn_start = QPushButton("Start Server")
        self.btn_start.setObjectName("startBtn")
        self.btn_bench = QPushButton("Benchmark")
        self.btn_stop = QPushButton("Stop Server")
        self.btn_stop.setObjectName("stopBtn")
        self.status_label = QLabel("🔴 Stopped")
        self.status_label.setObjectName("statusLabel")
        self.btn_save.clicked.connect(self._save_current)
        self.btn_start.clicked.connect(self._start_server)
        self.btn_bench.clicked.connect(self._open_benchmark)
        self.btn_stop.clicked.connect(self._stop_server)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_bench)
        controls.addWidget(self.btn_stop)
        controls.addStretch(1)
        controls.addWidget(self.status_label)
        center_layout.addLayout(controls)

        splitter.addWidget(center)

        # --- Right sidebar: KV forecast + live log ---
        right = QWidget()
        right.setMinimumWidth(320)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_split = QSplitter(Qt.Orientation.Vertical)

        # KV / memory forecast (GGUF math + live usage anchor)
        kv_box = QGroupBox("KV / memory forecast")
        kv_layout = QVBoxLayout(kv_box)
        kv_layout.setSpacing(6)
        self.kv_meta_label = QLabel("Load a GGUF model to estimate KV growth.")
        self.kv_meta_label.setObjectName("resourceValue")
        self.kv_meta_label.setWordWrap(True)
        kv_layout.addWidget(self.kv_meta_label)
        self.kv_anchor_label = QLabel("")
        self.kv_anchor_label.setObjectName("resourceValue")
        self.kv_anchor_label.setWordWrap(True)
        kv_layout.addWidget(self.kv_anchor_label)
        self.kv_table = QTableWidget(0, 6)
        self.kv_table.setHorizontalHeaderLabels(
            ["Ctx", "KV cache", "Est. VRAM", "Est. RAM", "VRAM %", "Note"]
        )
        self.kv_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.kv_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.kv_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.kv_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.kv_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.kv_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.kv_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.kv_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.kv_table.verticalHeader().setVisible(False)
        self.kv_table.setMinimumHeight(140)
        kv_layout.addWidget(self.kv_table)
        right_split.addWidget(kv_box)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(8)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Live log"))
        log_header.addStretch(1)
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.clicked.connect(lambda: self.log_view.clear())
        log_header.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_header)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(8000)
        self.log_view.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_view, stretch=1)
        right_split.addWidget(log_panel)

        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 2)
        right_split.setSizes([280, 420])
        right_layout.addWidget(right_split, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 640, 520])

        for w in (
            self.name_edit,
            self.binary_edit,
            self.model_edit,
            self.ngl_edit,
            self.threads_edit,
            self.host_edit,
            self.port_edit,
            self.ctk_edit,
            self.ctv_edit,
        ):
            w.textChanged.connect(self._refresh_preview)
        self.ctx_edit.currentTextChanged.connect(self._refresh_preview)
        self.ctx_edit.editTextChanged.connect(self._refresh_preview)
        self.raw_edit.textChanged.connect(self._refresh_preview)
        for chk in (self.jinja_chk, self.nommap_chk, self.mlock_chk):
            chk.toggled.connect(self._refresh_preview)
        for chk, edit in self.knob_widgets.values():
            chk.toggled.connect(self._refresh_preview)
            if isinstance(edit, QLineEdit):
                edit.textChanged.connect(self._refresh_preview)
            elif isinstance(edit, QComboBox):
                edit.currentTextChanged.connect(self._refresh_preview)
                edit.editTextChanged.connect(self._refresh_preview)

        self.model_edit.textChanged.connect(self._refresh_kv_forecast)
        self.ngl_edit.textChanged.connect(self._refresh_kv_forecast)
        self.ctk_edit.textChanged.connect(self._refresh_kv_forecast)
        self.ctv_edit.textChanged.connect(self._refresh_kv_forecast)
        self.ctx_edit.currentTextChanged.connect(self._refresh_kv_forecast)
        self.ctx_edit.editTextChanged.connect(self._refresh_kv_forecast)
        if "use_np" in self.knob_widgets:
            chk, edit = self.knob_widgets["use_np"]
            chk.toggled.connect(self._refresh_kv_forecast)
            if isinstance(edit, QLineEdit):
                edit.textChanged.connect(self._refresh_kv_forecast)
        if "use_swa_full" in self.knob_widgets:
            chk, _edit = self.knob_widgets["use_swa_full"]
            chk.toggled.connect(self._refresh_kv_forecast)
        self._refresh_kv_forecast()

    # --- profile list ---

    def _reload_list(self, select_id: str | None = None) -> None:
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        for profile in self.profiles:
            item = QListWidgetItem(profile.get("name") or "Untitled")
            item.setData(Qt.ItemDataRole.UserRole, profile["id"])
            self.profile_list.addItem(item)
        self.profile_list.blockSignals(False)

        target = select_id or self._current_id
        if target:
            for i in range(self.profile_list.count()):
                if self.profile_list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                    self.profile_list.setCurrentRow(i)
                    return
        if self.profile_list.count():
            self.profile_list.setCurrentRow(0)

    def _find_profile(self, profile_id: str) -> dict[str, Any] | None:
        for p in self.profiles:
            if p["id"] == profile_id:
                return p
        return None

    def _on_profile_selected(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        if previous is not None and not self._loading:
            prev_id = previous.data(Qt.ItemDataRole.UserRole)
            if prev_id:
                self._persist_form_into(prev_id, auto_save=True)
        if current is None:
            return
        self._load_profile_into_form(current.data(Qt.ItemDataRole.UserRole))

    def _load_profile_into_form(self, profile_id: str) -> None:
        profile = self._find_profile(profile_id)
        if not profile:
            return
        self._loading = True
        self._current_id = profile_id
        self.name_edit.setText(profile.get("name", ""))
        self.binary_edit.setText(profile.get("binary", ""))
        self.model_edit.setText(profile.get("model", ""))
        self.ngl_edit.setText(str(profile.get("ngl", "")))
        self.threads_edit.setText(str(profile.get("threads", "")))
        self.ctx_edit.setCurrentText(str(profile.get("ctx_size", "")))
        self.host_edit.setText(profile.get("host", ""))
        self.port_edit.setText(str(profile.get("port", "")))
        self.ctk_edit.setText(profile.get("cache_type_k", ""))
        self.ctv_edit.setText(profile.get("cache_type_v", ""))
        self.jinja_chk.setChecked(bool(profile.get("jinja")))
        self.nommap_chk.setChecked(bool(profile.get("no_mmap")))
        self.mlock_chk.setChecked(bool(profile.get("mlock")))

        for knob in KNOBS:
            chk, edit = self.knob_widgets[knob.use_key]
            on = bool(profile.get(knob.use_key))
            chk.setChecked(on)
            if knob.kind == "bool":
                continue
            value = str(profile.get(knob.value_key, knob.default) or "")
            if isinstance(edit, QComboBox):
                edit.setCurrentText(value)
                edit.setEnabled(on)
            elif isinstance(edit, QLineEdit):
                edit.setText(value)
                edit.setEnabled(on)

        self.raw_edit.setPlainText(profile.get("raw_args", ""))
        self._loading = False
        self._refresh_preview()

    def _form_as_profile(self, profile_id: str) -> dict[str, Any]:
        existing = self._find_profile(profile_id) or new_profile()
        data: dict[str, Any] = {
            **existing,
            "id": profile_id,
            "name": self.name_edit.text().strip() or "Untitled",
            "binary": self.binary_edit.text().strip(),
            "model": self.model_edit.text().strip(),
            "ngl": self.ngl_edit.text().strip(),
            "threads": self.threads_edit.text().strip(),
            "ctx_size": self.ctx_edit.currentText().strip(),
            "host": self.host_edit.text().strip(),
            "port": self.port_edit.text().strip(),
            "cache_type_k": self.ctk_edit.text().strip(),
            "cache_type_v": self.ctv_edit.text().strip(),
            "jinja": self.jinja_chk.isChecked(),
            "no_mmap": self.nommap_chk.isChecked(),
            "mlock": self.mlock_chk.isChecked(),
            "raw_args": self.raw_edit.toPlainText().strip(),
            "_flags_promoted": True,
            "_schema_seen": PROFILES_SCHEMA,
        }
        for knob in KNOBS:
            chk, edit = self.knob_widgets[knob.use_key]
            data[knob.use_key] = chk.isChecked()
            if knob.kind == "bool":
                continue
            if isinstance(edit, QComboBox):
                data[knob.value_key] = edit.currentText().strip()
            elif isinstance(edit, QLineEdit):
                data[knob.value_key] = edit.text().strip()
        return data

    def _persist_form_into(self, profile_id: str, auto_save: bool = False) -> None:
        updated = self._form_as_profile(profile_id)
        for i, p in enumerate(self.profiles):
            if p["id"] == profile_id:
                self.profiles[i] = updated
                break
        if auto_save:
            save_profiles(self.profiles)
            # refresh name in list without losing selection
            for i in range(self.profile_list.count()):
                item = self.profile_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == profile_id:
                    item.setText(updated["name"])
                    break

    def _save_current(self) -> None:
        if not self._current_id:
            return
        self._persist_form_into(self._current_id, auto_save=True)
        self._log_line(f"[launcher] Saved profile → {profiles_path()}")

    def _add_profile(self) -> None:
        if self._current_id:
            self._persist_form_into(self._current_id, auto_save=True)
        profile = new_profile(f"Profile {len(self.profiles) + 1}")
        self.profiles.append(profile)
        save_profiles(self.profiles)
        self._reload_list(select_id=profile["id"])

    def _duplicate_profile(self) -> None:
        if not self._current_id:
            return
        self._persist_form_into(self._current_id, auto_save=True)
        src = self._find_profile(self._current_id)
        if not src:
            return
        clone = deepcopy(src)
        clone["id"] = str(uuid.uuid4())
        clone["name"] = f"{src.get('name', 'Profile')} (copy)"
        self.profiles.append(clone)
        save_profiles(self.profiles)
        self._reload_list(select_id=clone["id"])

    def _delete_profile(self) -> None:
        if not self._current_id:
            return
        if len(self.profiles) <= 1:
            QMessageBox.warning(self, APP_NAME, "Keep at least one profile.")
            return
        name = self.name_edit.text().strip() or "this profile"
        reply = QMessageBox.question(
            self,
            APP_NAME,
            f"Delete “{name}”?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.profiles = [p for p in self.profiles if p["id"] != self._current_id]
        self._current_id = None
        save_profiles(self.profiles)
        self._reload_list()

    def _browse_binary(self) -> None:
        start = self.binary_edit.text().strip() or str(Path.home())
        filt = "Executable (*.exe);;All files (*)" if sys.platform == "win32" else "All files (*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select llama-server binary", start, filt)
        if path:
            self.binary_edit.setText(str(Path(path)))

    def _browse_model(self) -> None:
        start = self.model_edit.text().strip() or str(AI_STUFF)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model", start, "GGUF (*.gguf);;All files (*)"
        )
        if path:
            self.model_edit.setText(str(Path(path)))

    def _build_menu(self) -> None:
        tools = self.menuBar().addMenu("&Tools")
        act_hf = tools.addAction("Hugging Face Search & Download…")
        act_hf.triggered.connect(self._open_hf_window)

    def _open_hf_window(self) -> None:
        if self._hf_window is None:
            self._hf_window = HuggingFaceWindow(self)
            self._hf_window.model_path_chosen.connect(self._apply_hf_model)
        self._hf_window.show()
        self._hf_window.raise_()
        self._hf_window.activateWindow()

    def _open_benchmark(self) -> None:
        if self._current_id:
            self._persist_form_into(self._current_id, auto_save=True)
        profile = self._current_form_profile()
        try:
            build_bench_argv(profile)
        except ValueError as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        win = BenchmarkWindow(profile, self)
        self._bench_windows.append(win)
        win.finished.connect(lambda *_a, w=win: self._on_bench_closed(w))
        win.show()
        win.raise_()
        win.activateWindow()

    def _on_bench_closed(self, win: BenchmarkWindow) -> None:
        try:
            self._bench_windows.remove(win)
        except ValueError:
            pass

    def _apply_hf_model(self, path: str) -> None:
        self.model_edit.setText(path)
        self._refresh_preview()
        if self._current_id:
            self._persist_form_into(self._current_id, auto_save=True)
        self.raise_()
        self.activateWindow()

    def _current_form_profile(self) -> dict[str, Any]:
        pid = self._current_id or str(uuid.uuid4())
        return self._form_as_profile(pid)

    def _refresh_preview(self) -> None:
        if self._loading:
            return
        try:
            argv = build_argv(self._current_form_profile())
            self.preview_edit.setPlainText(format_command_preview(argv))
        except Exception as exc:  # noqa: BLE001
            self.preview_edit.setPlainText(f"(invalid) {exc}")

    # --- process ---

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def _log_line(self, text: str) -> None:
        self._append_log(text if text.endswith("\n") else text + "\n")

    def _update_running_ui(self, running: bool, pid: int | None = None) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        if running and pid:
            self.status_label.setText(f"🟢 Running (PID: {pid})")
        else:
            self.status_label.setText("🔴 Stopped")
        self._refresh_resources()

    def _set_usage_bar(self, bar: QProgressBar, percent: float) -> None:
        value = max(0, min(1000, int(round(percent * 10))))
        bar.setValue(value if bar.maximum() == 1000 else max(0, min(100, int(round(percent)))))
        if percent >= 90:
            level = "crit"
        elif percent >= 75:
            level = "warn"
        else:
            level = "ok"
        bar.setProperty("level", level)
        bar.style().unpolish(bar)
        bar.style().polish(bar)

    def _refresh_resources(self) -> None:
        if not hasattr(self, "ram_bar"):
            return
        pid = self.runner.pid if self.runner.running else None
        snap = sample_resources(pid)

        self.ram_label.setText(
            f"RAM: {snap.ram_used_gb:.1f} / {snap.ram_total_gb:.1f} GB  ({snap.ram_percent:.0f}%)"
        )
        self._set_usage_bar(self.ram_bar, snap.ram_percent)

        if snap.vram_total_mb > 0:
            free = max(0.0, snap.vram_total_mb - snap.vram_used_mb)
            name = snap.gpu_name or "GPU"
            self.vram_label.setText(
                f"VRAM ({name}): {snap.vram_used_mb:.0f} / {snap.vram_total_mb:.0f} MiB"
                f"  ({snap.vram_percent:.0f}%)  free {free:.0f} MiB"
            )
            self._set_usage_bar(self.vram_bar, snap.vram_percent)
            self.gpu_label.setText(f"GPU util: {snap.gpu_util:.0f}%")
            self.gpu_bar.setValue(int(round(snap.gpu_util)))
            self._set_usage_bar(self.gpu_bar, snap.gpu_util)
        else:
            self.vram_label.setText("VRAM: unavailable (nvidia-smi)")
            self.vram_bar.setValue(0)
            self.gpu_label.setText("GPU util: —")
            self.gpu_bar.setValue(0)

        if pid:
            parts = [f"Server PID {pid}"]
            if snap.proc_ram_mb is not None:
                parts.append(f"RAM {snap.proc_ram_mb:.0f} MiB")
            if snap.proc_vram_mb is not None:
                parts.append(f"VRAM {snap.proc_vram_mb:.0f} MiB")
            elif snap.vram_total_mb > 0:
                parts.append("VRAM (process) n/a")
            self.proc_label.setText(" · ".join(parts))
        else:
            self.proc_label.setText("Server process: not running")

        if snap.error and snap.vram_total_mb <= 0:
            self.proc_label.setText(f"Monitor note: {snap.error}")

        self._last_resource_snap = snap
        self._refresh_kv_forecast()

    def _parallel_slots(self) -> int:
        if "use_np" not in self.knob_widgets:
            return 1
        chk, edit = self.knob_widgets["use_np"]
        if not chk.isChecked():
            return 1
        if isinstance(edit, QLineEdit):
            return max(1, parse_positive_int(edit.text(), 1))
        return 1

    def _swa_full_enabled(self) -> bool:
        if "use_swa_full" not in self.knob_widgets:
            return False
        chk, _edit = self.knob_widgets["use_swa_full"]
        return bool(chk.isChecked())

    def _refresh_kv_forecast(self) -> None:
        if not hasattr(self, "kv_table"):
            return
        model_path = self.model_edit.text().strip()
        ctx = parse_positive_int(self.ctx_edit.currentText().strip(), 0)
        ctk = self.ctk_edit.text().strip() or "f16"
        ctv = self.ctv_edit.text().strip() or "f16"
        ngl = parse_positive_int(self.ngl_edit.text().strip(), 999)
        swa_full = self._swa_full_enabled()

        if not model_path:
            self.kv_meta_label.setText("Set a model path to estimate KV growth.")
            self.kv_anchor_label.setText("")
            self.kv_table.setRowCount(0)
            return
        try:
            info = read_gguf_model_info(model_path)
        except Exception as exc:  # noqa: BLE001
            self.kv_meta_label.setText(f"Could not read GGUF metadata: {exc}")
            self.kv_anchor_label.setText("")
            self.kv_table.setRowCount(0)
            return

        cfg = KvModelConfig(
            info=info,
            cache_type_k=ctk,
            cache_type_v=ctv,
            n_parallel=self._parallel_slots(),
            swa_full=swa_full,
        )
        bpt = cfg.bytes_per_token
        train = f"{info.train_ctx:,}" if info.train_ctx else "?"
        self.kv_meta_label.setText(
            f"{info.architecture} · {info.n_layer} layers · "
            f"heads {info.n_head}/{info.n_head_kv} · "
            f"head_dim K/V {info.n_embd_head_k}/{info.n_embd_head_v} · "
            f"train ctx {train} · "
            f"ΔKV ≈ {format_bytes(bpt)}/token · {cfg.swa_summary()}"
            f" (ctk={ctk}, ctv={ctv}, -np={cfg.n_parallel})"
        )

        snap = getattr(self, "_last_resource_snap", None)
        if snap is None:
            snap = sample_resources(self.runner.pid if self.runner.running else None)
            self._last_resource_snap = snap

        kv_now = cfg.kv_bytes(ctx) if ctx > 0 else 0.0
        # Match the Live resources VRAM bar (system nvidia-smi), not process-only.
        live_vram_mb = snap.vram_used_mb if (self.runner.running and snap.vram_total_mb > 0) else None
        live_ram_mb = None
        if self.runner.running and snap.ram_total_gb > 0:
            live_ram_mb = snap.ram_used_gb * 1024.0

        kv_on_gpu = ngl != 0  # -ngl 0 => KV typically on host RAM

        if live_vram_mb is not None and ctx > 0:
            fixed_vram_mb = max(0.0, live_vram_mb - (kv_now / (1024**2)))
            anchored = True
        else:
            try:
                file_mb = Path(model_path).stat().st_size / (1024**2)
            except OSError:
                file_mb = 0.0
            if ngl >= info.n_layer or ngl >= 99:
                weight_frac = 1.0
            elif ngl <= 0:
                weight_frac = 0.0
            else:
                weight_frac = min(1.0, ngl / max(1, info.n_layer))
            fixed_vram_mb = file_mb * weight_frac + 400.0
            anchored = False
            live_vram_mb = fixed_vram_mb + (kv_now / (1024**2))

        if live_ram_mb is not None and ctx > 0:
            if kv_on_gpu:
                fixed_ram_mb = live_ram_mb
            else:
                fixed_ram_mb = max(0.0, live_ram_mb - (kv_now / (1024**2)))
        else:
            fixed_ram_mb = (snap.ram_used_gb * 1024.0) if snap.ram_total_gb else 0.0
            if not kv_on_gpu and ctx > 0:
                fixed_ram_mb = max(0.0, fixed_ram_mb - (kv_now / (1024**2)))

        if anchored:
            self.kv_anchor_label.setText(
                f"Anchored to live GPU/RAM @ ctx {ctx:,}: "
                f"VRAM {live_vram_mb:.0f} MiB − est KV {kv_now / (1024**2):.0f} MiB "
                f"⇒ fixed ≈ {fixed_vram_mb:.0f} MiB"
                + (
                    f" · RAM fixed ≈ {fixed_ram_mb:.0f} MiB"
                    if live_ram_mb is not None
                    else ""
                )
                + (" · KV on GPU" if kv_on_gpu else " · KV on RAM (-ngl 0)")
            )
        else:
            self.kv_anchor_label.setText(
                "Server not running — showing formula estimate from GGUF + model file size. "
                "Start the server to anchor against real VRAM/RAM."
            )

        vram_total_mb = snap.vram_total_mb or 0.0
        ram_total_mb = snap.ram_total_gb * 1024.0 if snap.ram_total_gb else 0.0

        sig = (
            model_path,
            ctx,
            ctk,
            ctv,
            ngl,
            cfg.n_parallel,
            swa_full,
            round(live_vram_mb or 0),
            round(live_ram_mb or 0),
            round(kv_now / (1024**2)),
            round(vram_total_mb),
            round(ram_total_mb),
            anchored,
            kv_on_gpu,
        )
        if sig == getattr(self, "_kv_forecast_sig", None):
            return
        self._kv_forecast_sig = sig

        rows_ctx = forecast_ctx_ladder(ctx if ctx > 0 else 8192)
        self.kv_table.setRowCount(len(rows_ctx))
        for i, n_ctx in enumerate(rows_ctx):
            kv_b = cfg.kv_bytes(n_ctx)
            kv_mb = kv_b / (1024**2)
            is_current = n_ctx == ctx and anchored and live_vram_mb is not None

            if is_current:
                # Exact live numbers for the row matching the running config
                est_vram = live_vram_mb
                est_ram = live_ram_mb if live_ram_mb is not None else fixed_ram_mb
            elif kv_on_gpu:
                est_vram = fixed_vram_mb + kv_mb
                est_ram = fixed_ram_mb
            else:
                est_vram = fixed_vram_mb
                est_ram = fixed_ram_mb + kv_mb

            note = ""
            if n_ctx == ctx:
                note = "current"
            if info.train_ctx and n_ctx > info.train_ctx:
                note = (note + " · " if note else "") + "above train ctx"
            if vram_total_mb > 0:
                if est_vram >= vram_total_mb:
                    note = (note + " · " if note else "") + "VRAM OOM risk"
                elif est_vram >= vram_total_mb * 0.9:
                    note = (note + " · " if note else "") + "VRAM tight"
            if ram_total_mb > 0 and est_ram >= ram_total_mb * 0.95:
                note = (note + " · " if note else "") + "RAM tight"

            pct = (100.0 * est_vram / vram_total_mb) if vram_total_mb > 0 else 0.0
            headroom = max(0.0, vram_total_mb - est_vram) if vram_total_mb > 0 else None

            cells = [
                f"{n_ctx:,}" + (" ←" if n_ctx == ctx else ""),
                f"{kv_mb:.0f} MiB",
                f"{est_vram:.0f} / {vram_total_mb:.0f} MiB"
                if vram_total_mb
                else f"{est_vram:.0f} MiB",
                f"{est_ram:.0f} / {ram_total_mb:.0f} MiB"
                if ram_total_mb
                else f"{est_ram:.0f} MiB",
                f"{pct:.0f}%" if vram_total_mb else "—",
                note
                + (
                    f" · free ~{headroom:.0f} MiB"
                    if headroom is not None and "OOM" not in note
                    else ""
                ),
            ]
            for col, text in enumerate(cells):
                if col == 4:
                    continue
                item = QTableWidgetItem(text)
                if col in (1, 2, 3):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if n_ctx == ctx:
                    item.setForeground(QColor("#9ec5fe"))
                if vram_total_mb and est_vram >= vram_total_mb:
                    item.setForeground(QColor("#f87171"))
                elif vram_total_mb and est_vram >= vram_total_mb * 0.9:
                    item.setForeground(QColor("#fbbf24"))
                self.kv_table.setItem(i, col, item)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(min(1000, int(pct * 10)) if vram_total_mb else 0)
            bar.setFormat(f"{pct:.0f}%" if vram_total_mb else "—")
            bar.setTextVisible(True)
            self._set_usage_bar(bar, pct if vram_total_mb else 0.0)
            self.kv_table.setCellWidget(i, 4, bar)

    def _start_server(self) -> None:
        if self.runner.running:
            QMessageBox.warning(self, APP_NAME, "A server is already running. Stop it first.")
            return
        if self._current_id:
            self._persist_form_into(self._current_id, auto_save=True)

        profile = self._current_form_profile()
        try:
            argv = build_argv(profile)
        except ValueError as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            return

        binary = Path(argv[0])
        if not binary.is_file():
            QMessageBox.critical(self, APP_NAME, f"Binary not found:\n{binary}")
            return
        model = profile.get("model") or ""
        if model and not Path(model).is_file():
            QMessageBox.critical(self, APP_NAME, f"Model not found:\n{model}")
            return

        workdir = str(binary.parent)
        env = build_child_env(binary.parent)

        self._log_line(f"[launcher] Starting:\n{format_command_preview(argv)}")
        self._log_line(f"[launcher] cwd={workdir}")
        try:
            self.runner.start(argv, cwd=workdir, env=env)
        except OSError as exc:
            self._log_line(f"[launcher] Failed to start: {exc}")
            self._update_running_ui(False)

    def _stop_server(self) -> None:
        if not self.runner.running:
            self._update_running_ui(False)
            return
        pid = self.runner.pid
        self._log_line(f"[launcher] Stopping process tree PID {pid}…")
        self.runner.stop()
        self._update_running_ui(False)
        self._log_line("[launcher] Stopped.")

    def _on_server_started(self, pid: int) -> None:
        self._update_running_ui(True, pid)
        self._log_line(f"[launcher] Running PID {pid}")

    def _on_server_finished(self, exit_code: int) -> None:
        self._log_line(f"[launcher] Process exited with code {exit_code}")
        self._update_running_ui(False)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._current_id:
            self._persist_form_into(self._current_id, auto_save=True)
        if self.runner.running:
            self._stop_server()
        for win in list(self._bench_windows):
            if win.runner.running:
                win._stop_bench()
            win.close()
        super().closeEvent(event)


def main() -> int:
    ensure_stdio()
    # Ensure slots dir exists for profiles that use it
    SLOTS_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
