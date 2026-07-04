# WhisperFlow

Free, fully-local voice dictation for Windows 11 — a Wispr Flow alternative with zero cloud, zero subscription, and a **non-destructive** cleanup pass (the raw transcript is always preserved).

Press **Ctrl+Win**, speak, and the text lands in whatever app has focus — browser, IDE, terminal, chat. Speech-to-text runs on your own GPU via [faster-whisper](https://github.com/SYSTRAN/faster-whisper); nothing ever leaves your machine.

## Quick start

```powershell
# deps (one-time; faster-whisper + CUDA assumed present)
python -m pip install -r requirements.txt

# run with tray + overlay
python app.py

# or headless (console logging, no UI)
python app.py --headless
```

First run downloads the default model (`large-v3-turbo`, ~1.5GB) to the HuggingFace cache and holds it in VRAM (~1.5GB) for instant transcription.

## Install on a new machine (teammates)

Windows 10/11 only. You need a **microphone** and **Python 3.11+** (3.13 recommended — the Microsoft Store or [python.org](https://www.python.org/downloads/) build; both include the `tkinter` used by the pill/tray). Nothing in the repo hardcodes another user's paths — autostart resolves the Python path per-machine on first run.

1. **Get the code.** Ask the repo owner to add you as a collaborator (it's private), then:
   ```powershell
   git clone https://github.com/umeshsugara-ai/whisperflow
   cd whisperflow
   ```
2. **Install dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. **Pick a model for your hardware:**
   ```powershell
   python app.py --recommend
   ```
   Paste the printed `[model]` block into `config.toml`. No NVIDIA GPU? It suggests `small` on CPU, or the **Gemini cloud** engine — set the `GEMINI_API_KEY` env var.
4. **Tune `config.toml`:** set `[hotkey].combo` (e.g. `ctrl+windows` or `alt+windows`) and `[model].language` (`""` auto, `en`, `hi`, `hinglish`).
5. **First run:**
   ```powershell
   python app.py
   ```
   Downloads the model (~1.5GB) and **auto-registers autostart**. A slim pill appears at the bottom of the screen; hover it to see the hotkey.
6. **Reboot** → it launches automatically (windowless). Toggle anytime via tray → **"Start on Windows login"**, or `python app.py --install-autostart` / `--uninstall-autostart`.

**If dictation types nothing:** Windows → Sound → **Input** → confirm the mic isn't muted/at 0 and the Test bar moves when you speak (and any Nahimic/Realtek mic effect isn't muting it).

### Set up with Claude Code

Prefer to let an AI agent do the setup? Install [Claude Code](https://claude.com/claude-code), open a terminal in an empty folder, and paste a prompt like:

> Clone `https://github.com/umeshsugara-ai/whisperflow` (I have collaborator access) and set up WhisperFlow on my Windows machine: install `requirements.txt` with my Python 3.13, run `python app.py --recommend` and update `config.toml` `[model]` to match my hardware, set `[hotkey].combo` to `"ctrl+windows"`, then launch `python app.py` and confirm the log at `logs/whisperflow.log` reaches the "ready" line. If I have no NVIDIA GPU, configure the Gemini cloud engine instead and tell me to set `GEMINI_API_KEY`.

Claude Code will run the commands, edit `config.toml`, launch the app, and verify it reaches the **ready** state — then autostart takes over on the next reboot. Point it at the sections of this README if it needs model/config details.

## Usage

| Action | How |
|---|---|
| Hold-to-talk | Hold **Ctrl+Win**, speak, release → text is injected |
| Toggle mode | Tap **Ctrl+Win** (<350ms), speak freely, tap again to finish |
| Cancel a recording | **Esc** |
| Open the app window | **Right-click the pill**, double-click the tray icon, or just run `python app.py` again |
| Get the raw (uncleaned) transcript | Tray → "Copy last RAW transcript", or the History screen |
| Switch cleanup level live | Settings screen, or Tray → Cleanup tier (persists now) |

### The app window

Right-click the bottom pill, double-click the tray icon (or tray → "Open WhisperFlow"),
or launch the app a second time — all open the main window. A manual (non-autostart)
launch opens it automatically:

- **Home** — lifetime stats (total words, average WPM, day streak, dictations), a plain-language status strip ("All good ✓" or recent warnings), and your latest dictations with one-click copy.
- **History** — searchable list of every dictation with the RAW and cleaned text side by side.
- **Dictionary** — add vocabulary words and "when I say → write instead" rules; saved straight to config.toml.
- **Settings** — hotkey, language, cleanup tier, overlay pill, start-on-login. No config-file editing needed; restart-required fields say so.

Closing the window just hides it — WhisperFlow keeps running in the tray.

## Which model should I use?

```powershell
python app.py --recommend
```

Detects your GPU/VRAM/RAM/CPU and prints the best `[model]` settings for your machine (e.g. no NVIDIA GPU → `small` on CPU, or the cloud engine if you have an API key). The app also warns at startup if your config doesn't match your hardware.

## Configuration — `config.toml`

- **Engine**: `[model].engine = "local"` (default — fully on-device, private) or `"gemini"` — bring-your-own-key cloud transcription for machines that can't run a local model. Set your key via the `GEMINI_API_KEY` env var (or `[model].api_key`). Default cloud model `gemini-2.5-flash`; use `gemini-2.5-pro` for higher accuracy. **Privacy note:** the cloud engine sends dictation audio to Google — the app logs a clear notice when it's active. (TTS-named models like `gemini-2.5-pro-preview-tts` are text-to-speech and are rejected — they can't transcribe.)
- **Model swap**: set `[model].name` to `large-v3-turbo` (default), `large-v3` (best Hindi accuracy, slower), `medium`, `small`, or any raw HF CTranslate2 repo id.
- **Hinglish**: if auto-detect keeps choosing the wrong language, set `[model].language = "hi"`.
- **Cleanup tiers**: `off` = verbatim; `rules` = deterministic filler/punctuation cleanup (default); `llm` = local Ollama model (optional — install [Ollama](https://ollama.com) and `ollama pull qwen2.5:3b-instruct`). If Ollama is down, dictation silently degrades to `rules` — it never blocks.
- **Dictionary**: `[dictionary].vocabulary` biases recognition toward your terms; `[[dictionary.replacements]]` fixes persistent mis-hearings post-STT.
- Most changes apply via tray → "Reload config"; model/hotkey changes need a restart.

## Autostart on boot

WhisperFlow registers itself to start automatically at Windows login on first run
(a per-user `HKCU\...\Run` entry — no admin, fully reversible). With Store Python the
entry launches `wscript.exe //B run.vbs`, which starts the app hidden via the
`python.exe` alias — Store Python's `pythonw.exe` alias fails silently at logon, so
it is never used. The entry also **self-heals**: if it goes stale (moved folder,
changed Python, an old broken command), the next manual launch rewrites it.
After a reboot the resting pill just reappears; the log line
`started via Windows logon autostart` confirms it worked.

- Turn it off/on anytime: Settings screen or tray → **"Start on Windows login"**.
- Or from a terminal: `python app.py --install-autostart` / `python app.py --uninstall-autostart`.
- To opt out of the first-run auto-registration entirely, set `[startup].auto_register = false`
  in `config.toml` before the first launch.

## Privacy & history

Every dictation appends `{raw, injected, tier, ...}` to `history.jsonl` (local file, trimmed to `[history].max_entries`), and lifetime totals roll up into `stats.json` (word/dictation counts only — no text). Open the History screen to inspect, or clear it from there anytime. Audio is never written to disk.

## Known limitations

- **Elevated apps** (Run as administrator): Windows UIPI blocks injection from a non-elevated process. Run WhisperFlow elevated too if you need to dictate into elevated windows.
- **WSL terminals**: injection uses real unicode key events (not simulated Ctrl+V), which works in Windows Terminal — but if a specific target misbehaves, set `[inject].method = "paste"`.
- Record-then-transcribe, not streaming: text appears ~1–2.5s after you stop speaking (10s dictation on an RTX 4060).

## Development

```powershell
python -m pytest tests/ -q          # 108 unit tests
python scripts/test_inject.py --self-test
python scripts/test_overlay.py --cycle
python scripts/test_stt.py --smoke
python scripts/test_audio.py --duration 1 --check
```

Architecture and decision log: see `research/` (STORM briefing on how Wispr Flow works and where it breaks) and the plan in `C:/Users/Lenovo/.claude/plans/abb-agar-mujhe-apne-witty-snail.md`.
