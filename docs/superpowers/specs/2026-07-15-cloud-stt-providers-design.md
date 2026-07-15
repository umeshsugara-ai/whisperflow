# WhisperFlow — Multi-Provider Cloud STT + Slim Installer (design)

Date: 2026-07-15
Status: approved-in-brainstorm, pending spec review

## Context / problem

WhisperFlow's local engine (faster-whisper) is excellent on an NVIDIA GPU but on
weak/low-GPU or GPU-less machines it is **slow, low-quality, and RAM-heavy** — the
exact machines many teammates and would-be users have. Meanwhile hosted Whisper
(Groq) runs the *same* `whisper-large-v3-turbo` model in the cloud in a fraction of
a second, on a **free tier generous enough for any human** (2,000 requests/day,
28,800 audio-seconds/day, verified against Groq docs 2026-07-15).

Separately, the current installer is ~1GB because it bundles CUDA runtime + local
inference machinery — dead weight for a user who will only ever use cloud.

Goal: make WhisperFlow a **global product** where a user on any machine gets fast,
good dictation — picking a cloud provider tiered from free (Groq) to premium
(paid, higher quality), guided step-by-step to their own free API key — and where
the base install is light, pulling the heavy local-inference pack **only if** the
user chooses local.

### Non-goals
- No LangChain / no STT SDKs. The app's deliberate "plain REST, no SDK" design
  stays (keeps the frozen build lean). Multi-provider pluggability is achieved with
  a thin **provider registry**, not a framework.
- Not building a key-vault or per-provider billing dashboard. Keys live in `.env`.
- NVIDIA Riva (gRPC) full integration is **documented but deferred** (see Phase A).

## Architecture: provider registry + engine dispatch

One new data module `whisperflow/stt/providers.py` holds a registry — one `Provider`
dataclass per entry:

```
id: str                 # config value for [model].engine, e.g. "groq"
display_name: str       # "Groq (free, fast)"
kind: str               # openai_compatible | gemini | deepgram | local
base_url: str           # for openai_compatible
default_model: str      # e.g. "whisper-large-v3-turbo"
api_key_env: str        # e.g. "GROQ_API_KEY"
signup_url: str         # where to generate a key
free_note: str          # "2000 requests/day free"
cost_tier: str          # free | freemium | paid
quality_tier: str       # good | better | best
setup_steps: list[str]  # step-by-step key-generation guide
```

`create_engine(cfg.model)` (registry.py) dispatches on the provider's `kind`:

| kind | engine | status |
|---|---|---|
| `openai_compatible` | **new** `OpenAICompatibleEngine` | Phase A |
| `gemini` | existing `GeminiEngine` | built |
| `deepgram` | **new** small `DeepgramEngine` | Phase A |
| `local` | existing `FasterWhisperEngine` | built |

The single `OpenAICompatibleEngine` covers Groq, OpenAI, and any OpenAI-compatible
`/audio/transcriptions` endpoint — only `base_url` + `api_key_env` + `model` differ.

### Provider table (initial registry)

| id | Tier | Model | Key env | Signup | kind |
|---|---|---|---|---|---|
| `groq` | 🟢 free hero | whisper-large-v3-turbo | `GROQ_API_KEY` | console.groq.com/keys | openai_compatible |
| `gemini` | 🟢 free | gemini-2.5-flash | `GEMINI_API_KEY` | aistudio.google.com/apikey | gemini |
| `openai` | 🟡 paid, better | gpt-4o-transcribe | `OPENAI_API_KEY` | platform.openai.com/api-keys | openai_compatible |
| `deepgram` | 🟡 paid, best | nova-3 | `DEEPGRAM_API_KEY` | console.deepgram.com | deepgram |
| `nvidia` | 🟢 free credits | whisper-large-v3 | `NVIDIA_API_KEY` | build.nvidia.com | documented-only (gRPC; deferred) |
| `local` | ⚪ private/offline | large-v3-turbo…small | — | — | local |

## Phase A — multi-provider cloud STT engine

Files:
- **new** `whisperflow/stt/providers.py` — registry + `Provider` dataclass + `get(id)`,
  `all_providers()`, `cloud_providers()`.
- **new** `whisperflow/stt/openai_compatible_engine.py` — `OpenAICompatibleEngine`
  modeled on `gemini_engine.py`: builds a `multipart/form-data` POST (helper
  `_multipart_body`) to `{base_url}/audio/transcriptions` with `file` (in-memory WAV
  via the existing `_float32_to_wav_bytes` pattern), `model`, `language`,
  `response_format=json`, `temperature=0`, and `prompt` (vocabulary/initial_prompt,
  capped ≤224 tokens per OpenAI/Groq limit). Bearer-token auth header. Reuses
  `RawResult`. 25MB / 10s-min-billing are non-issues for short dictation but the
  engine logs a warning if audio > provider max.
- **new** `whisperflow/stt/deepgram_engine.py` — small Deepgram REST engine
  (`api.deepgram.com/v1/listen`, `nova-3`, raw WAV body, `Authorization: Token`).
- **edit** `whisperflow/stt/registry.py` — `create_engine` dispatches on
  `providers.get(cfg.engine).kind`; unknown id → clear error.
- **edit** `whisperflow/config.py` — `VALID_ENGINES` derived from the registry ids;
  `resolve_api_key()` reads the provider's `api_key_env` (registry-driven) when
  `[model].api_key` is empty; `_validate` keeps "cloud engine needs a key".
- **edit** `whisperflow/sysinfo.py` `recommend()` — weak/GPU-less machines recommend
  **`groq`** (free, fast, same model) as the primary cloud option instead of gemini;
  `has_api_key` check generalizes to "any cloud key present".

Error handling: HTTP errors surface `RuntimeError(f"{provider} API error {code}: …")`
(same shape as Gemini); network errors → "unreachable"; 401 → a friendly "your
{PROVIDER} key looks invalid — regenerate at {signup_url}" mapped in the controller's
error → overlay path.

Tests: multipart body builder; registry lookups; `create_engine` dispatch per kind;
`recommend()` picks groq on a GPU-less spec; config validation for each provider id.
(Live API calls are NOT in the unit suite — a manual `scripts/test_cloud_stt.py`
smoke script hits each provider with a real key.)

## Phase B — onboarding / step-by-step key guide

- **edit** `whisperflow/ui/main_window.py` SettingsPage — new **"Speech engine"**
  section: a provider dropdown (each row: display name + free/paid badge + quality
  note), and when a cloud provider is selected: its `free_note`, a **"Get a free key →"**
  button that opens `signup_url` in the browser, the numbered `setup_steps`, and a
  key entry field that writes `{API_KEY_ENV}=…` to `<data_dir>/.env` (new helper
  `config.set_env_var(key, value)` — create/update `.env` in place, never echo the
  key to logs). Switching engine writes `[model].engine` via the existing
  `save_config`. A "restart to apply" note (engine change isn't hot-reloaded).
- **edit** `whisperflow/ui/main_window.py` HomePage — first-run: if `recommend()`
  found no GPU AND no cloud key is set, show a dismissible card:
  *"No GPU detected — local dictation will be slow. Get a free Groq key (2,000
  dictations/day) for instant cloud transcription."* → **"Set up now"** opens the
  Settings Speech-engine section. Dismissal persists (reuse the `.guide_dismissed`
  pattern, separate marker `.cloud_hint_dismissed`).
- **edit** `README.md` — a "Choose your speech engine" section: the provider table
  above + per-provider step-by-step (signup link → create key → paste in Settings or
  `.env`). Frame local as "private/offline, needs a good GPU".

Tests: `set_env_var` create/update/idempotent + no-log; provider dropdown builds all
registry rows; first-run cloud card shows only when GPU-less & keyless & undismissed.

## Phase C — slim installer (light base + downloadable local-pack)

Split the frozen build so the base installer is cloud-ready and light; local
inference is fetched on demand.

- **Base build** (new spec variant) excludes `ctranslate2`, the nvidia CUDA wheels,
  `faster_whisper`'s heavy paths, torch — target ~150MB. It contains all cloud
  engines + UI. `installer/whisperflow.spec` grows a `WF_BUILD=cloud|full` switch
  (env-driven) controlling the `binaries`/`excludes`.
- **Local-inference pack**: a pre-zipped `whisperflow-local-pack-<ver>.zip`
  (ctranslate2 `.pyd` + CUDA DLLs + faster_whisper) built from the SAME PyInstaller/
  Python env, published as a GitHub release asset.
- **On-demand fetch**: when the user selects a **local** model (Settings) or on first
  run if a config somehow specifies local without the pack present, the app downloads
  the pack from the pinned release URL into `<data_dir>\local-pack\`, verifies a
  SHA-256, extracts, and at startup prepends that dir to `sys.path` **before** importing
  faster-whisper (new `whisperflow/localpack.py`: `is_installed()`, `ensure_installed(progress_cb)`,
  `activate()`). A progress UI reuses the model-download status strip.
- **Startup guard**: `create_engine` for `kind=local` calls `localpack.activate()`;
  if absent, raises a friendly "Local mode needs a one-time download — opening setup"
  routed to the pack installer instead of a crash.

**Risk (flagged):** loading native extension modules into a PyInstaller-frozen app at
runtime is fragile — the pack MUST be built in the identical interpreter/ABI, and
`sys.path` injection before first import must be airtight. This phase is **gated by a
smoke test** (build cloud-base, install, download pack, reach `ready — hotkey` with a
local model from a non-repo dir). **If the pack approach proves fragile in testing,
auto-fallback to shipping two installer variants** (`WhisperFlow-Cloud-Setup.exe` ~150MB
and `WhisperFlow-Full-Setup.exe` ~1GB) — same user outcome, no runtime native-load risk.

- **edit** `scripts/build_installer.ps1` — build cloud-base by default; `-Full` flag
  builds the fat variant; `-LocalPack` builds+zips the pack for release. README +
  `gh release` updated to publish base installer + local pack.

Tests: `localpack.is_installed/ensure_installed/activate` against a fake pack dir
(monkeypatched download); SHA mismatch aborts cleanly; `create_engine(local)` without
pack raises the friendly guided error, not ImportError.

## Config / data changes

- `[model].engine` accepts any registry id (was `local|gemini`).
- `.env` gains provider-specific keys (`GROQ_API_KEY`, `OPENAI_API_KEY`, …); `.env` is
  the canonical secret store (gitignored). `set_env_var` is the writer.
- New markers in `<data_dir>`: `.cloud_hint_dismissed`, `local-pack/`.
- All dev-mode paths unchanged (`data_dir()` = repo root when not frozen).

## Sequencing

A → B → C (each independently shippable). A alone lets any user go cloud today; B
makes it discoverable/guided; C slims distribution. Ship a release after A+B, another
after C.

## Verification (end to end)

1. **A**: full `pytest` green; manual `scripts/test_cloud_stt.py` transcribes a WAV via
   Groq + OpenAI with real keys; `recommend()` returns `groq` on a GPU-less spec.
2. **B**: launch app on a (simulated) GPU-less config → first-run cloud card → "Set up
   now" → paste Groq key → `.env` written → restart → dictation via Groq works end to end.
3. **C**: build cloud-base installer (~150MB) → install from clean machine/dir → cloud
   dictation works with zero CUDA present → pick a local model → pack downloads +
   activates → local dictation reaches `ready — hotkey`. If fragile → two-variant fallback
   verified instead.
4. Commit per phase; releases via `gh release create` (base installer + local pack assets).

## Risks summary

- Native-pack runtime load (Phase C) — mitigated by test-gate + two-variant fallback.
- Provider free-tier terms change — registry `free_note` is easy to update; no logic
  depends on exact numbers.
- Multipart-in-urllib correctness — covered by a body-builder unit test + live smoke.
- Key leakage — `set_env_var` and all engines must never log key values (assert in tests).
