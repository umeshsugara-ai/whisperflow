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
  *(Superseded 2026-07-16: NVCF turned out to expose a plain-HTTPS offline route for
  parakeet-ctc-1.1b-asr — `https://{function-id}.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions`
  — so `nvidia` shipped as the 5th provider via `nvidia_engine.py`, English-only,
  no gRPC client needed. whisper-large-v3/canary on NVCF remain gRPC-only and out.)*

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
cost_tier: str          # free | freemium | paid
cost_note: str          # "Free — 2000/day" | "~$0.006/min"
quality_tier: str       # good | better | best
speed_note: str         # "Instant" | "Fast" | "Depends on your GPU"
setup_steps: list[str]  # step-by-step key-generation guide
```

Privacy is derived, not stored: `kind == "local"` → 🔒 Offline/Private, else ☁ Cloud.
These fields feed the plain-language **badges** in the picker (Phase B).

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
| `gemini` | 🟢 free | gemini-2.5-flash-lite | `GEMINI_API_KEY` | aistudio.google.com/apikey | gemini |
| `openai` | 🟡 paid, better | gpt-4o-transcribe | `OPENAI_API_KEY` | platform.openai.com/api-keys | openai_compatible |
| `deepgram` | 🟡 paid, best | nova-3 | `DEEPGRAM_API_KEY` | console.deepgram.com | deepgram |
| `nvidia` | 🟢 free credits | parakeet-ctc-1_1b-asr | `NVIDIA_API_KEY` | build.nvidia.com | implemented 2026-07-16 (HTTP NVCF; English-only) |
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
- **edit** `whisperflow/config.py` `ModelConfig.cloud_model` default — `gemini-2.5-flash`
  → `gemini-2.5-flash-lite` (verified 2026-07-15 against ai.google.dev/gemini-api/docs/pricing:
  audio input $0.30/M tokens vs $1.00/M for `-flash`, same audio-input capability, still
  current/stable — not deprecated). `gemini-2.5-pro` stays documented as the "higher
  accuracy, costs more" alternative for users who want it. Any `*-tts-*` model id is
  rejected by the existing guard in `gemini_engine.py` (TTS models cannot transcribe;
  unrelated to this change, already correct).

Error handling: HTTP errors surface `RuntimeError(f"{provider} API error {code}: …")`
(same shape as Gemini); network errors → "unreachable"; 401 → a friendly "your
{PROVIDER} key looks invalid — regenerate at {signup_url}" mapped in the controller's
error → overlay path.

Tests: multipart body builder; registry lookups; `create_engine` dispatch per kind;
`recommend()` picks groq on a GPU-less spec; config validation for each provider id.
(Live API calls are NOT in the unit suite — a manual `scripts/test_cloud_stt.py`
smoke script hits each provider with a real key.)

## Phase B — onboarding + decision-support ("which do I pick?")

**No engine is ever picked silently.** The system computes a recommendation and
pre-selects/highlights it, but `config.toml` is only written after the user explicitly
confirms — either by accepting the recommendation with one click or by picking a
different provider. This changes Phase A's first-run flow: `bootstrap_config()` (added in
Phase A, `app.py`) currently writes the recommended config unattended before the UI even
shows. Phase B replaces that call site with a blocking **first-run chooser** (see below);
`bootstrap_config`'s hardware-probe + `recommend()` logic is reused as-is, only the
"write immediately" step moves behind user confirmation.

Three decision-support layers, all reachable from both the first-run chooser and the
Settings screen (same UI component, two entry points):

**Layer 1 — system recommendation + reason (how the SYSTEM decides, user still confirms).**
`recommend(specs, has_api_key)` already probes hardware; it returns the pick plus a human
`reason`. The chooser pre-selects/highlights this provider with a **"★ Recommended for
your PC — {reason}"** line (e.g. "No GPU detected — Groq is free and instant") and a
**"Use this"** button. The reason string comes straight from `recommend()`, no new logic
— only WHEN it's applied changes (on confirm, not on load).

**Layer 2 — plain-language badges (how the USER eyeballs it).**
- **edit** `whisperflow/ui/main_window.py` SettingsPage — new **"Speech engine"** section.
  The provider dropdown/list shows, per row, badges built from the registry fields —
  privacy (🔒 Offline / ☁ Cloud), cost (💚 Free / 💛 Paid + `cost_note`), quality
  (Good/Better/Best), speed (`speed_note`). Example row: *"Groq — ☁ Cloud · 💚 Free
  (2000/day) · Better · ⚡ Instant"* vs *"Local — 🔒 Offline · 💚 Free · Best · needs GPU"*.
  When a cloud provider is selected: a **"Get a free key →"** button opens `signup_url`,
  the numbered `setup_steps` render inline, and a key field writes `{API_KEY_ENV}=…` to
  `<data_dir>/.env` (new helper `config.set_env_var(key, value)` — create/update in place,
  never echo the key to logs). Switching engine writes `[model].engine` via `save_config`;
  a "restart to apply" note appears (engine isn't hot-reloaded).

**Layer 3 — optional "Help me choose" (2 questions, for the unsure).**
- A small **"Help me choose"** button opens a 2-question mini-helper: (1) *"Fully
  private/offline, or is fast cloud OK?"* (2) *"Free, or willing to pay a little for the
  best accuracy?"*. A pure function `providers.choose(privacy_pref, budget_pref, specs)`
  maps the answers (+ hardware) to a provider id and highlights it in the list. Offline+any
  → local (warns if no GPU); cloud+free → groq; cloud+paid → openai/deepgram. Unit-tested,
  no UI state.

**First-run chooser (replaces silent `bootstrap_config` write).**
- **new** `whisperflow/ui/first_run.py` — a blocking Toplevel shown by `app.py` in place of
  today's unattended `bootstrap_config()` call, whenever `config.toml` doesn't exist yet.
  Same layout/badges as the Settings Speech-engine section (Layers 1-3 above, shared
  rendering code — no duplicated UI logic), opened pre-scrolled to the recommended row.
  Two ways forward: **"Use recommended (Groq)"** (one click, no key prompt shown yet — see
  below) or pick any other provider from the list. Choosing **local** or a cloud provider
  the user already has a key for (detected via env/`​.env`) writes the config and continues
  straight into the app. Choosing a cloud provider with **no key yet** shows that
  provider's `setup_steps` + signup link + key field inline (same widget Settings uses) —
  the config is written, and the app continues, only once a key is entered or the user
  explicitly picks "I'll add a key later" (which falls back to `local`, with the reason
  logged, so the app is never left half-configured). This dialog owns the config-write
  call (`sysinfo.recommend()` → build `Config` → `save_config`), reusing exactly the
  dataclass-building logic `bootstrap_config()` already has (Phase A) — refactor
  `bootstrap_config` into a pure `build_recommended_config(specs, rec) -> Config`
  (no I/O) that both the chooser and any future non-interactive path (`--headless` first
  run, tests) can call, then have the chooser handle confirmation + `save_config` itself.
- **edit** HomePage — the old "no GPU, no key" dismissible nudge card is removed (the
  first-run chooser now covers this moment); HomePage keeps only the existing first-run
  "How to dictate" guide card (unrelated, already shipped).

**Docs.**
- **edit** `README.md` — "Which speech engine should I pick?" with a plain flowchart
  (Good NVIDIA GPU + want offline? → Local · else → Groq, free · want best accuracy & OK to
  pay? → OpenAI/Deepgram) + the provider table + per-provider step-by-step (signup → create
  key → paste in Settings or `.env`).

Tests: `set_env_var` create/update/idempotent + never-logged; `providers.choose()` maps each
(privacy, budget, hardware) combo to the expected id; picker builds a badge row per registry
entry; recommended provider is pre-selected with its reason; `build_recommended_config(specs,
rec)` is pure (no file I/O, easy to unit-test) and produces a `Config` matching `rec`; the
first-run chooser does NOT write `config.toml` until a provider is confirmed (no-write-on-open
is itself a test); confirming a keyless cloud provider without entering a key falls back to
`local` rather than saving a broken cloud config.

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

**Outcome (2026-07-16, real smoke test on a live machine):** the risk materialized.
All 6 code tasks shipped and were verified correct in isolation (140→177 tests, two
whole-branch reviews each caught and fixed real integration bugs — an un-updated
`.iss` referencing the wrong exe name, a pack missing `av`/`onnxruntime`, a silent
startup crash instead of a visible error). The end-to-end smoke test (download real
pack from a GitHub release → verify → extract → activate → load `ctranslate2` inside
the frozen cloud-base exe) reproducibly hung for 5-10+ minutes at the exact step the
spec flagged as risky — CPU idle, no error, `MpOav.dll` (Windows Defender's on-access
scan filter) loaded into the process, consistent with Defender's cloud-reputation
check stalling on multiple large freshly-extracted unsigned native DLLs. A real bug
was found and fixed along the way (`activate()` only did `sys.path.insert()`, missing
the `os.add_dll_directory()` a compiled extension needs to resolve its own dependent
DLLs on Windows) and stayed in the code — it's a correct, necessary fix regardless —
but the hang persisted identically after the fix, meaning Defender interaction (not
purely the DLL-search-path bug) is the dominant cause, and that isn't something this
codebase can control.
**Decision: fall back to two static installer variants**, per this section's own
pre-authorized escape hatch. `WhisperFlow-Setup.exe` (cloud, ~29MB — smaller than the
~150MB estimate) is the default recommended download; `WhisperFlow-Full-Setup.exe`
(~1GB, includes local inference, zero runtime native-load risk — this build path was
already proven reliable in Phase 1-4 testing) is offered as the explicit second option
for anyone who wants Local mode from first launch.

**Final outcome (2026-07-16, later the same day):** Umesh decided to drop the Full
installer entirely — `WhisperFlow-Setup.exe` (cloud-only, ~29MB) is the **single**
distributed build. The Full-Setup exe and the local-pack zip were deleted from the
GitHub release; `localpack.py` (whose download URL pointed at the deleted asset),
its tests, and the `-Full`/`-LocalPack` build-script paths were removed from the
repo. Local (on-device) mode remains available only when running from source with
faster-whisper installed, and the engine picker gates it honestly on cloud installs.

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
- Provider free-tier terms change — registry `cost_note` is easy to update; no logic
  depends on exact numbers.
- Multipart-in-urllib correctness — covered by a body-builder unit test + live smoke.
- Key leakage — `set_env_var` and all engines must never log key values (assert in tests).
