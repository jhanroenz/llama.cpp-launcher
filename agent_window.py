#!/usr/bin/env python3
"""PyQt Agent chat window for llama-launcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from agent_harness import (
    AgentCallbacks,
    AgentCancelled,
    AgentConfig,
    AgentError,
    normalize_endpoint_host,
    profile_api_key,
    run_agent_turn,
)


class _AgentWorker(QThread):
    assistant_text = pyqtSignal(str)
    tool_start = pyqtSignal(str, str, str)  # id, name, args_json
    tool_end = pyqtSignal(str, str, str)  # id, name, result
    status = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal(object)  # updated history list

    def __init__(
        self,
        config: AgentConfig,
        history: list[dict[str, Any]],
        user_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._history = history
        self._user_text = user_text
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        def cancel_check() -> bool:
            return self._cancel

        cbs = AgentCallbacks(
            on_assistant=lambda t: self.assistant_text.emit(t),
            on_tool_start=lambda cid, name, args: self.tool_start.emit(
                cid, name, json.dumps(args, ensure_ascii=False)[:2000]
            ),
            on_tool_end=lambda cid, name, result: self.tool_end.emit(cid, name, result[:8000]),
            on_status=lambda s: self.status.emit(s),
        )
        try:
            updated = run_agent_turn(
                self._config,
                self._history,
                self._user_text,
                callbacks=cbs,
                cancel_check=cancel_check,
            )
            self.finished_ok.emit(updated)
        except AgentCancelled:
            self.failed.emit("Cancelled.")
        except AgentError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Unexpected error: {exc}")


class AgentWindow(QDialog):
    """Chat UI that runs the Python agent harness against a live llama-server."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        load_settings_fn: Any = None,
        save_settings_fn: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Agent — llama-launcher")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(980, 720)
        self.setMinimumSize(720, 520)

        self._load_settings = load_settings_fn
        self._save_settings = save_settings_fn
        self._settings: dict[str, Any] = self._load_settings() if self._load_settings else {}
        self._history: list[dict[str, Any]] = []
        self._worker: _AgentWorker | None = None
        self._server_running = False
        self._host = "127.0.0.1"
        self._port = "11434"
        self._api_key = ""
        self._pending_assistant = False

        self._build_ui()
        self._apply_settings_to_form()
        self._update_busy(False)
        self._refresh_endpoint_label()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.endpoint_label = QLabel()
        self.endpoint_label.setObjectName("resourceValue")
        self.endpoint_label.setWordWrap(True)
        root.addWidget(self.endpoint_label)

        warn = QLabel(
            "Shell tools run on this machine with cwd=workspace (not fully sandboxed). "
            "File/memory tools are confined to workspace / vault paths."
        )
        warn.setObjectName("resourceValue")
        warn.setWordWrap(True)
        root.addWidget(warn)

        paths = QGroupBox("Paths & agent settings")
        form = QFormLayout(paths)

        vault_row = QHBoxLayout()
        self.vault_edit = QLineEdit()
        self.vault_edit.setPlaceholderText("Obsidian vault folder…")
        btn_vault = QPushButton("Browse…")
        btn_vault.clicked.connect(self._browse_vault)
        vault_row.addWidget(self.vault_edit, stretch=1)
        vault_row.addWidget(btn_vault)
        form.addRow("Obsidian vault", vault_row)

        ws_row = QHBoxLayout()
        self.workspace_edit = QLineEdit()
        self.workspace_edit.setPlaceholderText("Workspace for file/shell tools…")
        btn_ws = QPushButton("Browse…")
        btn_ws.clicked.connect(self._browse_workspace)
        ws_row.addWidget(self.workspace_edit, stretch=1)
        ws_row.addWidget(btn_ws)
        form.addRow("Workspace", ws_row)

        knobs = QHBoxLayout()
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 100)
        self.max_steps_spin.setValue(20)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setDecimals(2)
        self.temp_spin.setValue(0.2)
        knobs.addWidget(QLabel("Max steps"))
        knobs.addWidget(self.max_steps_spin)
        knobs.addWidget(QLabel("Temperature"))
        knobs.addWidget(self.temp_spin)
        knobs.addStretch(1)
        self.btn_save_prefs = QPushButton("Save prefs")
        self.btn_save_prefs.clicked.connect(self._save_prefs)
        knobs.addWidget(self.btn_save_prefs)
        form.addRow("Limits", knobs)
        root.addWidget(paths)

        split = QSplitter(Qt.Orientation.Vertical)

        chat_box = QGroupBox("Chat")
        chat_layout = QVBoxLayout(chat_box)
        self.chat_view = QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setObjectName("previewView")
        chat_layout.addWidget(self.chat_view)
        split.addWidget(chat_box)

        tools_box = QGroupBox("Tool activity")
        tools_layout = QVBoxLayout(tools_box)
        self.tool_view = QPlainTextEdit()
        self.tool_view.setReadOnly(True)
        self.tool_view.setMaximumBlockCount(2000)
        tools_layout.addWidget(self.tool_view)
        split.addWidget(tools_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, stretch=1)

        input_row = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Message the agent… (Ctrl+Enter to send)")
        self.input_edit.setFixedHeight(80)
        self.input_edit.installEventFilter(self)
        input_row.addWidget(self.input_edit, stretch=1)

        btns = QVBoxLayout()
        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self._send)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_clear = QPushButton("Clear chat")
        self.btn_clear.clicked.connect(self._clear_chat)
        btns.addWidget(self.btn_send)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        input_row.addLayout(btns)
        root.addLayout(input_row)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("resourceValue")
        root.addWidget(self.status_label)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if obj is self.input_edit and event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event
            if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
                key_event.modifiers() & Qt.KeyboardModifier.ControlModifier
            ):
                self._send()
                return True
        return super().eventFilter(obj, event)

    def _apply_settings_to_form(self) -> None:
        self.vault_edit.setText(str(self._settings.get("obsidian_vault") or ""))
        self.workspace_edit.setText(str(self._settings.get("agent_workspace") or ""))
        try:
            self.max_steps_spin.setValue(int(self._settings.get("agent_max_steps") or 20))
        except (TypeError, ValueError):
            self.max_steps_spin.setValue(20)
        try:
            self.temp_spin.setValue(float(self._settings.get("agent_temperature") or 0.2))
        except (TypeError, ValueError):
            self.temp_spin.setValue(0.2)

    def _collect_settings(self) -> dict[str, Any]:
        out = dict(self._settings)
        out["obsidian_vault"] = self.vault_edit.text().strip()
        out["agent_workspace"] = self.workspace_edit.text().strip()
        out["agent_max_steps"] = self.max_steps_spin.value()
        out["agent_temperature"] = self.temp_spin.value()
        return out

    def _save_prefs(self) -> None:
        self._settings = self._collect_settings()
        if self._save_settings:
            self._save_settings(self._settings)
        self.status_label.setText("Preferences saved.")

    def _browse_vault(self) -> None:
        start = self.vault_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Obsidian vault", start)
        if path:
            self.vault_edit.setText(str(Path(path)))

    def _browse_workspace(self) -> None:
        start = self.workspace_edit.text().strip() or self.vault_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select agent workspace", start)
        if path:
            self.workspace_edit.setText(str(Path(path)))

    def set_endpoint(
        self,
        *,
        host: str,
        port: str,
        api_key: str = "",
        server_running: bool = False,
        profile_name: str = "",
    ) -> None:
        self._host = normalize_endpoint_host(host)
        self._port = str(port or "11434").strip() or "11434"
        self._api_key = api_key or ""
        self._server_running = bool(server_running)
        self._profile_name = profile_name
        self._refresh_endpoint_label()
        self._update_busy(self._worker is not None and self._worker.isRunning())

    def _refresh_endpoint_label(self) -> None:
        state = "🟢 Running" if self._server_running else "🔴 Server stopped"
        name = getattr(self, "_profile_name", "") or ""
        name_bit = f" · profile “{name}”" if name else ""
        self.endpoint_label.setText(
            f"{state}{name_bit} · http://{self._host}:{self._port}/v1/chat/completions"
        )

    def _append_chat(self, text: str) -> None:
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertPlainText(text if text.endswith("\n") else text + "\n")
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    def _append_tool(self, text: str) -> None:
        self.tool_view.moveCursor(QTextCursor.MoveOperation.End)
        self.tool_view.insertPlainText(text if text.endswith("\n") else text + "\n")
        self.tool_view.moveCursor(QTextCursor.MoveOperation.End)

    def _update_busy(self, busy: bool) -> None:
        can_send = (not busy) and self._server_running
        self.btn_send.setEnabled(can_send)
        self.btn_stop.setEnabled(busy)
        self.input_edit.setEnabled(not busy)
        self.btn_clear.setEnabled(not busy)

    def _build_config(self) -> AgentConfig | None:
        vault_raw = self.vault_edit.text().strip()
        ws_raw = self.workspace_edit.text().strip()
        if not ws_raw:
            ws_raw = vault_raw or str(Path.home())
        workspace = Path(ws_raw).expanduser()
        vault: Path | None = None
        if vault_raw:
            vault = Path(vault_raw).expanduser()
            if not vault.is_dir():
                QMessageBox.warning(self, "Agent", f"Obsidian vault does not exist:\n{vault}")
                return None
        return AgentConfig(
            host=self._host,
            port=self._port,
            api_key=self._api_key,
            workspace=workspace,
            vault=vault,
            max_steps=self.max_steps_spin.value(),
            temperature=float(self.temp_spin.value()),
        )

    def _send(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._server_running:
            QMessageBox.warning(
                self,
                "Agent",
                "llama-server is not running. Start a profile from the main window first.",
            )
            return
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        config = self._build_config()
        if config is None:
            return

        # Persist prefs quietly on each send
        self._settings = self._collect_settings()
        if self._save_settings:
            self._save_settings(self._settings)

        self._append_chat(f"\nYou: {text}\n")
        self.input_edit.clear()
        self._pending_assistant = True
        self._append_chat("Agent: ")

        self._worker = _AgentWorker(config, list(self._history), text, self)
        self._worker.assistant_text.connect(self._on_assistant)
        self._worker.tool_start.connect(self._on_tool_start)
        self._worker.tool_end.connect(self._on_tool_end)
        self._worker.status.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished.connect(self._on_thread_finished)
        self._update_busy(True)
        self.status_label.setText("Running…")
        self._worker.start()

    def _stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self.status_label.setText("Stopping…")

    def _clear_chat(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._history = []
        self.chat_view.clear()
        self.tool_view.clear()
        self.status_label.setText("Chat cleared.")

    def _on_assistant(self, text: str) -> None:
        if self._pending_assistant:
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
            self.chat_view.insertPlainText(text + "\n")
            self._pending_assistant = False
        else:
            self._append_chat(f"Agent: {text}\n")

    def _on_tool_start(self, call_id: str, name: str, args_json: str) -> None:
        self._append_tool(f"→ {name} ({call_id})\n  args: {args_json}\n")

    def _on_tool_end(self, call_id: str, name: str, result: str) -> None:
        snippet = result if len(result) <= 1500 else result[:1500] + "…"
        self._append_tool(f"← {name} ({call_id})\n{snippet}\n\n")

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_failed(self, message: str) -> None:
        if self._pending_assistant:
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
            self.chat_view.insertPlainText(f"[error] {message}\n")
            self._pending_assistant = False
        else:
            self._append_chat(f"[error] {message}\n")
        self.status_label.setText(message)

    def _on_finished_ok(self, history: object) -> None:
        if isinstance(history, list):
            self._history = history
        if self._pending_assistant:
            # Model returned empty content but finished
            self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
            self.chat_view.insertPlainText("(no text)\n")
            self._pending_assistant = False
        self.status_label.setText("Done")

    def _on_thread_finished(self) -> None:
        self._update_busy(False)
        self._worker = None

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self._worker and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(3000)
        # Save prefs on close
        self._settings = self._collect_settings()
        if self._save_settings:
            self._save_settings(self._settings)
        super().closeEvent(event)


def endpoint_from_profile(profile: dict[str, Any]) -> tuple[str, str, str]:
    host = normalize_endpoint_host(str(profile.get("host") or "127.0.0.1"))
    port = str(profile.get("port") or "11434").strip() or "11434"
    return host, port, profile_api_key(profile)
