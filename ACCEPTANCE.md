# WhisperFlow — Acceptance Checklist

Fill in each cell after testing. Automated gates already passed (38 unit tests + inject self-test + overlay cycle + STT smoke + audio check + mutex/log checks); this file tracks the MANUAL pass.

## How to run a test session

```powershell
cd D:\whisperFlowMy
python app.py            # tray + overlay
```

Hold **Ctrl+Win** and speak (release to inject), or tap for toggle mode, **Esc** to cancel.

## Injection matrix

For each cell: dictate a short sentence and check the text lands **verbatim** (including Devanagari), clipboard unchanged (type mode) / restored (paste mode: set `[inject].method = "paste"` in config.toml, reload from tray).

Fixtures to speak/inject (also available via `python scripts/test_inject.py --countdown --fixture N --mode type|paste`):
- 0 English: "The quick brown fox jumps over the lazy dog."
- 1 Devanagari: "नमस्ते दुनिया, यह एक परीक्षण है"
- 2 Hinglish: "Kal meeting hai na, toh please deck ready rakhna yaar."
- 3 Symbols: `it's "quoted" — em-dash &<tags> 100% #done`

| Target | type EN | type HI | type Hinglish | type symbols | paste EN | paste HI | Notes |
|---|---|---|---|---|---|---|---|
| Notepad | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| Chrome — Gmail compose | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| Chrome — address bar | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| Chrome — WhatsApp Web input | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| VS Code — editor | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| VS Code — integrated terminal | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | classic paste-injection failure site |
| Windows Terminal — PowerShell | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| Windows Terminal — Git Bash | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | |
| **Claude Code CLI** | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | **flagship test — Wispr's documented regression site** |
| Elevated Notepad (Run as admin) | ☐ | — | — | — | ☐ | — | expected FAIL non-elevated (UIPI); verify graceful (no crash, error flash) |

## Trigger modes & recovery

| Check | Pass |
|---|---|
| Hold Ctrl+Win >350ms, speak, release → text injected | ☐ |
| Tap Ctrl+Win (<350ms), speak, tap again → text injected | ☐ |
| Esc during recording → cancelled, nothing injected | ☐ |
| Typing normally never triggers recording | ☐ |
| Overlay shows "Recording · <your mic name>" with correct device | ☐ |
| Overlay never steals focus (dictate into Notepad while overlay visible) | ☐ |
| Second `python app.py` prints "already running" | ☐ |

## STT quality (EN/HI/Hinglish)

| Check | Pass |
|---|---|
| `python scripts/test_stt.py --live --duration 6` — English sentence accurate | ☐ |
| Same — Hindi sentence accurate (try `--language hi`) | ☐ |
| Same — Hinglish mixed sentence accurate | ☐ |
| `--model small` loads a different model (registry swap works) | ☐ |
| Latency ballpark: 10s speech → text within ~2–3s | ☐ |

## Cleanup & history (non-destructive contract)

| Check | Pass |
|---|---|
| Tier "rules": "um" / "matlab" / "yaar" stripped, content words untouched | ☐ |
| Tier "off": verbatim text injected | ☐ |
| Tray → "Copy last RAW transcript" returns pre-cleanup text | ☐ |
| history.jsonl contains both raw + injected for every dictation | ☐ |
| Tier "llm" without Ollama running → still injects (rules-fallback) | ☐ |

## Daily-driver day

| Check | Pass |
|---|---|
| One full workday of real use, no crash/freeze | ☐ |
| Bluetooth mic connect/disconnect mid-day picked up + shown in overlay | ☐ |
| Filler list tuned from real transcripts (edit `[cleanup].extra_fillers`) | ☐ |
| Reboot → autostart via shell:startup shortcut → first dictation works | ☐ |
| RAM footprint acceptable (Task Manager: expect ~150–250MB + 1.5GB VRAM) | ☐ |
