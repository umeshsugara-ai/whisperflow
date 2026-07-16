# WhisperFlow

Free voice dictation for Windows 11 — a Wispr Flow alternative with zero subscription, bring-your-own-key speech engines (Groq is free), and a **non-destructive** cleanup pass (the raw transcript is always preserved).

<p>
  <a href="https://github.com/umeshsugara-ai/whisperflow/releases/latest/download/WhisperFlow-Setup.exe"><b>⬇ Download for Windows</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/umeshsugara-ai/whisperflow/releases/latest">all releases</a>
</p>

> No Python, no git, no setup steps — run the installer, click through the wizard, done. First launch walks you through picking a speech engine (Groq is free, 30-second signup).

Press your hotkey (**Ctrl+Win** by default, fully customizable), speak, and the text lands in whatever app has focus — browser, IDE, terminal, chat. Speech-to-text goes through the cloud provider you pick (Groq / Gemini / OpenAI / Deepgram / NVIDIA — your own API key, audio only, never stored by WhisperFlow), or fully on-device via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) when running from source on a machine with a GPU.

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

### Option A — the .exe installer (easiest, no Python needed)

Download **`WhisperFlow-Setup.exe`** (~29MB) from the [GitHub Releases](https://github.com/umeshsugara-ai/whisperflow/releases) page. This build covers cloud speech engines (Groq/Gemini/OpenAI/Deepgram/NVIDIA) — local (on-device) mode isn't included right now.

1. Run the installer and click through the wizard — it asks about **start with Windows** and a **desktop shortcut**, then installs per-user (no admin needed).
2. **Windows may show a "Windows protected your PC" SmartScreen warning** the first time — this is normal for an app without a paid code-signing certificate, not a sign anything's wrong. Click **"More info" → "Run anyway"** to continue.
3. Finish with "Launch WhisperFlow" checked. On first launch it walks you through picking a speech engine and, for cloud providers, entering an API key (with a link to get a free one).
4. Settings, dictation history, and logs live in `%LOCALAPPDATA%\WhisperFlow`. To use a cloud engine, add its API key via Settings → Speech engine, or put e.g. `GROQ_API_KEY=your-key` in a `.env` file in that folder.

Uninstall from Windows Settings → Apps; it force-closes WhisperFlow first (no more "still running after uninstall" — fixed 2026-07-16) and asks whether to keep your history/settings.

**Building the installer (maintainer):** install [Inno Setup 6](https://jrsoftware.org/isinfo.php), then run `powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1` → `installer\Output\WhisperFlow-Setup.exe`.

### Option B — developer install (git clone)

Windows 10/11 only. You need a **microphone** and **Python 3.11+** (3.13 recommended — the Microsoft Store or [python.org](https://www.python.org/downloads/) build; both include the `tkinter` used by the pill/tray). Nothing in the repo hardcodes another user's paths — autostart resolves the Python path per-machine on first run.

1. **Get the code:**
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
   Paste the printed `[model]` block into `config.toml`. No NVIDIA GPU? It suggests `small` on CPU, or the **Gemini cloud** engine — see the API-key step below.
4. **API key (only if using the Gemini cloud engine or `gemini` cleanup tier — skip otherwise):**
   create a file named `.env` in the WhisperFlow folder (next to `app.py`) containing:
   ```
   GEMINI_API_KEY=paste-your-key-here
   ```
   Get a free key at https://aistudio.google.com/apikey. The `.env` file is gitignored —
   your key stays on your machine. (Alternative: a Windows user env var —
   `setx GEMINI_API_KEY "your-key"`, then sign out and back in.)
5. **Tune `config.toml`:** set `[hotkey].combo` (e.g. `ctrl+windows` or `alt+windows`) and `[model].language` (`""` auto, `en`, `hi`, `hinglish`).
6. **First run:**
   ```powershell
   python app.py
   ```
   Downloads the model (~1.5GB), **auto-registers autostart**, and opens the app window. A slim pill also appears at the bottom of the screen; hover it to see the hotkey.
7. **Reboot** → it launches automatically (windowless, pill only). Toggle anytime via Settings or tray → **"Start on Windows login"**, or `python app.py --install-autostart` / `--uninstall-autostart`.
8. **(Optional) Taskbar shortcut with the WhisperFlow icon:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
   ```
   Creates a Start Menu entry that opens the product window (not a console) — right-click it → **Pin to taskbar**. Pass `-Name "yourName"` to customize the label, e.g. `-File scripts\create_shortcut.ps1 -Name "myWhisperFlow"`.

**If dictation types nothing** (pill opens and closes, flat waveform, "No speech — check mic ⚠" flash): open the app window → **Settings → Microphone** — pick your real mic from the dropdown, hit **Test mic**, and Save when the bar moves. The usual culprit is Windows quietly making a virtual mic the default (e.g. "Microphone (Camo)", which records silence unless the Camo app is streaming); picking your physical mic in the dropdown pins it, no Windows Sound hunting needed. If the bar stays flat on *every* mic, check Windows Settings → Privacy & security → Microphone → "Let desktop apps access your microphone", and that input volume isn't 0. The verdict under Test mic names the exact device WhisperFlow opened, and warns when a pinned mic wasn't found or a known-virtual mic is in use.

### Set up with Claude Code

Prefer to let an AI agent do the setup? Install [Claude Code](https://claude.com/claude-code), open a terminal in an empty folder, and paste a prompt like:

> Clone `https://github.com/umeshsugara-ai/whisperflow` and set up WhisperFlow on my Windows machine: install `requirements.txt` with my Python 3.13, run `python app.py --recommend` and update `config.toml` `[model]` to match my hardware, set `[hotkey].combo` to `"ctrl+windows"`, then launch `python app.py` and confirm the log at `logs/whisperflow.log` reaches the "ready" line. If I have no NVIDIA GPU, configure the Gemini cloud engine instead, ask me for my GEMINI_API_KEY, and put it in a `.env` file next to app.py.

Claude Code will run the commands, edit `config.toml`, launch the app, and verify it reaches the **ready** state — then autostart takes over on the next reboot. Point it at the sections of this README if it needs model/config details.

## Usage

Your **hotkey** is whatever `[hotkey].combo` is set to — **`Ctrl+Win` by default**. The gestures below use it; substitute your own combo if you changed it (see *Change your hotkey* below).

| Action | How |
|---|---|
| Hold-to-talk | Hold your hotkey (default **Ctrl+Win**), speak, release → text is injected |
| Toggle mode | Tap the hotkey (<350ms), speak freely, tap again to finish |
| Live typing | In toggle mode, pause naturally mid-dictation → the sentence so far types out while you keep talking (`[streaming]` in config, on by default) |
| Cancel a recording | **Esc** |
| Open the app window | **Right-click the pill**, double-click the tray icon, or just run `python app.py` again |
| Get the raw (uncleaned) transcript | Tray → "Copy last RAW transcript", or the History screen |
| Switch cleanup level live | Settings screen, or Tray → Cleanup tier (persists now) |

#### Change your hotkey

Two ways — the change takes effect after you **restart** WhisperFlow (the hotkey isn't hot-reloaded):

- **In the app (easiest):** open the app window → **Settings** → pick a combo from the **Hotkey** dropdown (`Ctrl+Win`, `Alt+Win`, or `Win+Space`) → it saves automatically. A "restart to apply" note appears.
- **By hand:** edit `[hotkey].combo` in `config.toml` — keys joined with `+`, e.g. `combo = "alt+windows"`. Any combo the [`keyboard`](https://github.com/boppreh/keyboard) library understands works. Avoid `alt+space` (Windows system-menu shortcut).

Whatever you set, the pill and the Home screen "How to dictate" card show your actual combo — so it always matches what you press. The **first-run card** on the Home screen is the fastest way for a new teammate to learn the gestures.

### The app window

Right-click the bottom pill, double-click the tray icon (or tray → "Open WhisperFlow"),
or launch the app a second time — all open the main window. A manual (non-autostart)
launch opens it automatically:

- **Home** — lifetime stats (total words, average WPM, day streak, dictations), a plain-language status strip ("All good ✓" or recent warnings), and your latest dictations with one-click copy.
- **History** — searchable list of every dictation with the RAW and cleaned text side by side.
- **Dictionary** — add vocabulary words and "when I say → write instead" rules; saved straight to config.toml.
- **Settings** — hotkey, language, cleanup tier, live typing, overlay pill, start-on-login, a **microphone picker** (dropdown of your real input devices) and a **Test mic** button with a live level bar (the 10-second answer to "why is nothing transcribing"). No config-file editing needed — everything applies on Save, no restart.

Closing the window just hides it — WhisperFlow keeps running in the tray.

## Which model should I use?

```powershell
python app.py --recommend
```

Detects your GPU/VRAM/RAM/CPU and prints the best `[model]` settings for your machine (e.g. no NVIDIA GPU → `small` on CPU, or the cloud engine if you have an API key). The app also warns at startup if your config doesn't match your hardware.

## Which speech engine should I pick?

WhisperFlow never picks silently — on first launch you'll see a chooser with a
recommendation and the full list, and you confirm before anything is saved.
Change it anytime in **Settings → Speech engine**.

| Engine | Privacy | Cost | Quality | Speed | Needs |
|---|---|---|---|---|---|
| **Groq** | ☁ Cloud | Free — 2,000/day | Better | Instant | A free account (30 seconds to sign up) |
| **Gemini** | ☁ Cloud | Free tier | Better | Fast | A free Google account |
| **OpenAI** | ☁ Cloud | Paid (~$0.006/min) | Best | Fast | Billing set up on your OpenAI account |
| **Deepgram** | ☁ Cloud | $200 free credit, then paid | Best | Fast | A free account to start |
| **NVIDIA** | ☁ Cloud | Free credits on signup | Better | Fast | A free build.nvidia.com account (English dictation only) |
| **Local** | 🔒 Fully offline | Free | Best | Depends on your GPU | Running from source (developer install) with a decent NVIDIA GPU |

**Quick picks:**
- Just want it to work for free → **Groq** (instant, free tier is generous —
  this is what the app recommends automatically).
- Want the best possible accuracy and don't mind paying a little → **OpenAI**
  or **Deepgram**.
- Developer with a good NVIDIA GPU who wants everything to stay on the
  machine → run from source (Option B) and pick **Local**.

> **Install size:** the installer is ~29MB and includes only the cloud
> engines. Local (on-device) inference isn't part of the installed build —
> it works when running from source with `faster-whisper` installed.

Each cloud option needs its own free (or paid) API key — the in-app chooser and
Settings screen walk you through getting one with a direct sign-up link, or you
can add it by hand to a `.env` file next to your data folder:
```
GROQ_API_KEY=paste-your-key-here
```

## Configuration — `config.toml`

- **Engine**: `[model].engine` = `groq` | `gemini` | `openai` | `deepgram` | `nvidia` (cloud, bring-your-own-key) or `local` (fully on-device — from-source installs only). Each cloud provider reads its key from its own env var (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `NVIDIA_API_KEY`) — put it in a `.env` file next to your config, or set `[model].api_key`. Gemini's default cloud model is `gemini-2.5-flash-lite`; use `gemini-2.5-pro` for higher accuracy. **Privacy note:** cloud engines send dictation audio to that provider — the app logs a clear notice when one is active. (TTS-named models like `gemini-2.5-pro-preview-tts` are text-to-speech and are rejected — they can't transcribe.)
- **Model swap**: Settings → Speech engine has a **Model dropdown** per provider with cost/quality notes (e.g. Gemini `flash-lite` cheapest → `pro` best). The box is typable — when a provider releases a new model, just type its id; no app update needed. Applies live on Save. (Config-file equivalent: `[model].cloud_model` for cloud, `[model].name` for local — `large-v3-turbo` default, `large-v3` best Hindi, `medium`, `small`, or any raw HF CTranslate2 repo id.)
- **Hinglish**: if auto-detect keeps choosing the wrong language, set `[model].language = "hi"`.
- **Cleanup tiers**: `off` = verbatim; `rules` = deterministic filler/punctuation cleanup (default); `llm` = local Ollama model (optional — install [Ollama](https://ollama.com) and `ollama pull qwen2.5:3b-instruct`). If Ollama is down, dictation silently degrades to `rules` — it never blocks.
- **Dictionary**: `[dictionary].vocabulary` biases recognition toward your terms; `[[dictionary.replacements]]` fixes persistent mis-hearings post-STT.
- **Live typing**: `[streaming].enabled` (default `true`) transcribes at natural pauses *while you're still speaking* — a long dictation types out sentence by sentence instead of arriving in one block at the end. Chunks close after `pause_s` (0.7s) of silence once `min_chunk_s` (2s) is buffered, or force-cut at `max_chunk_s` (30s). Earlier chunks are fed back as prompt context so punctuation/casing carry across boundaries. While the hotkey is physically held (hold-to-talk) the text is transcribed in the background but injected on release — held modifiers never corrupt keystrokes; the win there is that the transcription latency is already paid down when you release.
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
- Live typing is pause-based chunking, not true word-by-word streaming: text appears at natural pauses (and once injected it can't be retro-corrected by later context). The final fragment still lands ~1–2.5s after you stop speaking (10s dictation on an RTX 4060).
- With live typing on, switching windows mid-dictation sends the later chunks to the newly focused window (text follows your cursor, Wispr-style). Chunks that can't be typed safely at the moment (hotkey held, focus mid-switch) wait and flush together at the end.

## Development

```powershell
python -m pytest tests/ -q          # 215 unit tests
python scripts/test_inject.py --self-test
python scripts/test_overlay.py --cycle
python scripts/test_stt.py --smoke
python scripts/test_audio.py --duration 1 --check
```

Architecture and decision log: see `research/` (STORM briefing on how Wispr Flow works and where it breaks) and the plan in `C:/Users/Lenovo/.claude/plans/abb-agar-mujhe-apne-witty-snail.md`.

## License

[MIT](LICENSE) — © 2026 Umesh Sugara.
