# NVIDIA build.nvidia.com (build-nvidia)

## Quick Re-Run
```bash
# Verify the key page still behaves as documented (logged-out):
# open in a browser and confirm the "Sign In to Get Started with NVIDIA AI"
# wall renders IN PLACE (URL must stay /settings/api-keys, no redirect):
start https://build.nvidia.com/settings/api-keys

# Verify the WhisperFlow integration end-to-end (needs a real key):
set NVIDIA_API_KEY=nvapi-...
python scripts/test_cloud_stt.py nvidia
```

**Last successful run:** 2026-07-16 | key-flow verified logged-out; ASR HTTP endpoint verified from model API pages | ~15m
**Output file(s):** `whisperflow/stt/nvidia_engine.py`, `whisperflow/stt/providers.py` (nvidia row)

## Portal Profile
| Field | Value |
|-------|-------|
| URL | https://build.nvidia.com |
| Type | api_first / spa (Next.js) |
| Intent explored | automate (document key-creation flow) + decide (which ASR models are HTTP-reachable) |
| AI involvement | Tier 1 (custom engine script; endpoints stable, no browser needed at runtime) |
| Browser tool used | Claude in-app browser (JS extraction; page renders with innerText hidden — read outerHTML) |
| Anti-bot | None hit on page loads; APIs need Bearer key. Cookie banner (OneTrust-style, "Reject Optional" works) |
| Auth | NVIDIA account (email-first signup, free, no card) → `nvapi-` API key |
| Data format | JSON API (NVCF invocation endpoints) |
| Unique key field | NVCF function-id (UUID per hosted model) |

## How it works
build.nvidia.com is NVIDIA's hosted NIM catalog. Each hosted model has an NVCF
function-id; inference goes to per-function URLs. **API keys are created at
`https://build.nvidia.com/settings/api-keys`** — that URL is a safe deep-link:
logged-out it shows the sign-in wall in place (no redirect), and after
login/signup the user lands on the API Keys page with a "Generate API Key"
button. Keys start with `nvapi-`.

## Endpoints / URLs
```
KEYS:      https://build.nvidia.com/settings/api-keys
ASR-HTTP:  https://{function-id}.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions
           (multipart: language=en-US, file=@audio.wav; Authorization: Bearer nvapi-...)
ASR-GRPC:  grpc.nvcf.nvidia.com:443 (metadata: function-id, authorization)
MODEL-API: https://build.nvidia.com/{publisher}/{model-slug}/api
```

Function-ids verified 2026-07-16 (from each model's /api page):
```
parakeet-ctc-1_1b-asr  1598d209-5e27-4d3c-8079-4751568b1081   HTTP + gRPC (English)
whisper-large-v3       b702f636-f60c-4a3d-a6f4-f3568c13bd7d   gRPC ONLY
canary-1b-asr          (page shows gRPC only)                  gRPC ONLY
```

## Flow map (key creation)
1. Navigate to `https://build.nvidia.com/settings/api-keys`
2. Logged out → "Sign In to Get Started with NVIDIA AI" wall renders in place:
   email field → **Next** (creates/joins NVIDIA Developer Program account; free, no card)
3. After auth, the API Keys page loads at the same URL
4. Click **Generate API Key**; copy the `nvapi-...` value (WhisperFlow user pastes it into the app)

## Gotchas & edge cases
- The SPA hides `document.body.innerText` (returns empty even when rendered) —
  scrape `document.documentElement.outerHTML` instead.
- OneTrust cookie banner blocks text extraction until dismissed; "Reject
  Optional" button click via JS works.
- Only parakeet-ctc-1.1b-asr has the plain-HTTPS transcription route; the
  multilingual models are gRPC-only on NVCF — do NOT assume whisper-large-v3
  is HTTP-reachable (its /api page has no HTTP section).
- NVCF HTTP route is sized for short clips (~5MB); WhisperFlow enforces this
  pre-flight via Provider.max_upload_bytes.
- Screenshot capture via the in-app browser pane timed out repeatedly on this
  site (renderer issue) — textual JS extraction used as the evidence trail.

## Change detection
- `https://build.nvidia.com/settings/api-keys` must return the sign-in wall
  in place when logged out (URL unchanged). If it 404s or redirects to a
  different keys URL, update providers.py `signup_url` + `setup_steps`.
- Re-check parakeet's /api page for the HTTP curl block and function-id
  `1598d209-5e27-4d3c-8079-4751568b1081`; if the UUID changed, update
  `FUNCTION_IDS` in `whisperflow/stt/nvidia_engine.py`.

## History
| Date | Intent | Findings | Duration | Tool | Notes |
|---|---|---|---|---|---|
| 2026-07-16 | decide | HTTP ASR route exists only for parakeet; function-ids captured | ~10m | in-app browser (JS) | drove the nvidia_engine.py implementation |
| 2026-07-16 | automate | key page = /settings/api-keys, email-first signup, in-place login wall | ~5m | in-app browser (JS) | fixed WhisperFlow setup_steps + signup_url; Umesh supplied the URL |
