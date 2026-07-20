# llama-launcher

Desktop GUI for managing and running advanced `llama-server` / `llama-bench` profiles. It is a visual profile editor and process wrapper for custom llama.cpp builds — not a model runner like LM Studio.

Profiles are stored at:

- **Windows:** `%APPDATA%\llama_launcher\profiles.json`
- **Linux / macOS:** `~/.config/llama_launcher/profiles.json`

## Prerequisites

- **Python 3.10+** (3.11+ recommended)
- A built **llama-server** binary from [llama.cpp](https://github.com/ggerganov/llama.cpp) (or your fork)
- Optional: CUDA / GPU drivers if you use GPU offload

## Fresh clone setup

```bash
git clone git@github.com:jhanroenz/llama.cpp-launcher.git
cd llama.cpp-launcher
```

### 1. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python main.py
```

## First-run checklist

1. Open or create a profile.
2. Set **Binary** to your `llama-server` / `llama-server.exe` path (Browse).
3. Set **Model** to a `.gguf` file (Browse), or download one via the Hugging Face helper in the app.
4. Tune knobs / raw args as needed, then **Save** and **Start Server**.

Default seed profiles may point at machine-specific paths — update them to match your install.

## Dependencies

| Package           | Purpose                          |
|-------------------|----------------------------------|
| PyQt6             | Desktop UI                       |
| psutil            | Process / system helpers         |
| huggingface_hub   | Optional model search & download |

## Notes

- The app does **not** ship llama.cpp; point each profile at your own binary.
- `llama-bench` is expected next to `llama-server` when you use the bench window.
- Live logs appear in the app; Stop ends the server process tree (important on Windows so VRAM is released).
