You are an expert Python developer specialized in desktop automation toolkits (PyQt6 / PySide6) and cross-platform systems programming.

### OBJECTIVE
Build a lightweight, highly flexible Python desktop GUI manager named `llama-launcher` that manages, fine-tunes, saves, and executes highly specific, advanced `llama-server` configurations. 

The user runs custom, bleeding-edge forks of llama.cpp that utilize highly complex command-line arguments. Standard LLM desktop runners (like LM Studio) hide or break these options. This app must act purely as an advanced visual profile editor and terminal execution wrapper.

### CRITICAL CONTEXT: SYSTEM ENVIRONMENT & CROSS-PLATFORM COMPATIBILITY
- **Current Environment:** The user is currently running this manager on **Windows**.
- **The Challenge:** The example commands provided by the user use Linux-style paths (e.g., `/run/media/...`), but the application **must inspect the host system (`sys.platform`)** and handle execution in a clean, Windows-native format.
- **Windows Process Execution Requirements:** When spawning the `llama-server` executable on Windows, the process engine must properly handle paths with spaces, forward vs. backward slashes (`/` vs `\`), and use correct Windows process management flags (e.g., using `subprocess.CREATE_NO_WINDOW` if needed to prevent an ugly command prompt window from popping up outside the GUI, while still completely capturing `stdout`/`stderr`).

### TECHNICAL REQUIREMENTS & FEATURES
1. **Tech Stack:** Python 3, PyQt6 (or PySide6), `subprocess` module for process management, and standard `json` library for storage. No bulky external database dependencies.
2. **Profile Management:**
   - Left-hand sidebar showing a list of saved model profiles.
   - Buttons to "Add Profile", "Duplicate Profile", and "Delete Profile".
   - Settings must save automatically to `~/.config/llama_launcher/profiles.json` (or the Windows equivalent `%APPDATA%\llama_launcher\profiles.json`) whenever a profile is saved.
3. **The Parameter Editor (Right-Hand Panel) & Binary Selection:**
   - **Binary Executable Selector:** Include a dedicated text field paired with a visual "Browse" button. Clicking this button must open a native system file dialog (`QFileDialog`) allowing the user to browse and explicitly target the specific `llama-server.exe` binary they want to use for that profile.
   - Must provide explicit text inputs/toggles for core knobs but MUST include a "Raw Custom Arguments" multi-line text area to append arbitrary, experimental strings seamlessly.
   - For reference and immediate testing, seed the application configuration with these two default profiles out-of-the-box (adapted dynamically to the system's pathing rules):
     
     *Profile 1: Qwen-TurboQuant*
     - Binary: "/run/media/janroenz/New Volume/llama-cpp-turboquant/build/bin/llama-server" (or Windows executable equivalent)
     - Arguments: -m "/run/media/janroenz/New Volume/AI stuff/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf" -ngl 999 --n-cpu-moe 35 --no-mmap -t 8 --ctx-size 262144 --cache-type-k turbo4 --cache-type-v turbo3 --jinja --host 0.0.0.0 --port 11434 --mlock

     *Profile 2: Qwen-Coder-Advanced*
     - Binary: "/run/media/janroenz/New Volume/llama.cpp/build/bin/llama-server" (or Windows executable equivalent)
     - Arguments: -m "/run/media/janroenz/New Volume/AI stuff/Qwen3-Coder-30B-A3B-Q2_K_XL.gguf" -ngl 999 -ot ".ffn_.*_exps.=CPU" --no-mmap -t 8 -c 131072 -b 512 -ub 512 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --jinja --host 0.0.0.0 --port 11434 --reasoning-budget 0 --reasoning off -np 1 --mlock --cache-reuse 256

4. **Robust Windows Process Execution Engine:**
   - When the user clicks a "Start Server" button, the script must spin up the user-selected binary with its associated flags as a non-blocking background process using `QProcess` or `subprocess.Popen`.
   - The GUI main loop must remain completely responsive while the server runs.
   - **Live Log Stream:** Implement a scrollable, read-only multi-line text terminal box inside the app that captures and prints `stdout` and `stderr` from the server process *live* in real-time.
   - **Safe Windows Termination:** Provide a prominent "Stop Server" button. On Windows, it must cleanly terminate the process tree (using standard Python `os.kill` or taskkill procedures if necessary) to ensure the server doesn't get orphaned, leaving massive models hanging in GPU VRAM.
5. **UI/UX Polish:**
   - Use a clean, dark-mode-first aesthetic (using native QPalette tweaks or basic QSS).
   - Display a status indicator showing whether the server is currently "🔴 Stopped" or "🟢 Running (PID: X)".

### CRITICAL CODE QUALITY DIRECTIVES
- Do not use placeholders, comments like `# TODO`, or cut corners. 
- Implement robust token/string parsing so that complex flags involving quotation marks or regular expressions (like `-ot ".ffn_.*_exps.=CPU"`) do not break, drop quotes, or split improperly when passed to the shell execution layer on Windows.
- Write the entire implementation inside a single, clean, self-contained Python script file that can be immediately run with `python main.py`.
