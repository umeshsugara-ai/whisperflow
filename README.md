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

## Usage

| Action | How |
|---|---|
| Hold-to-talk | Hold **Ctrl+Win**, speak, release → text is injected |
| Toggle mode | Tap **Ctrl+Win** (<350ms), speak freely, tap again to finish |
| Cancel a recording | **Esc** |
| Get the raw (uncleaned) transcript | Tray → "Copy last RAW transcript" |
| Switch cleanup level live | Tray → Cleanup tier → Off / Rules / LLM |

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

`Win+R` → `shell:startup` → create a shortcut to `run.vbs`.

## Privacy & history

Every dictation appends `{raw, injected, tier, ...}` to `history.jsonl` (local file, trimmed to `[history].max_entries`). Tray → "Open history" to inspect, or delete the file anytime. Audio is never written to disk.

## Known limitations

- **Elevated apps** (Run as administrator): Windows UIPI blocks injection from a non-elevated process. Run WhisperFlow elevated too if you need to dictate into elevated windows.
- **WSL terminals**: injection uses real unicode key events (not simulated Ctrl+V), which works in Windows Terminal — but if a specific target misbehaves, set `[inject].method = "paste"`.
- Record-then-transcribe, not streaming: text appears ~1–2.5s after you stop speaking (10s dictation on an RTX 4060).

## Development

```powershell
python -m pytest tests/ -q          # 38 unit tests
python scripts/test_inject.py --self-test
python scripts/test_overlay.py --cycle
python scripts/test_stt.py --smoke
python scripts/test_audio.py --duration 1 --check
```

Architecture and decision log: see `research/` (STORM briefing on how Wispr Flow works and where it breaks) and the plan in `C:/Users/Lenovo/.claude/plans/abb-agar-mujhe-apne-witty-snail.md`.
