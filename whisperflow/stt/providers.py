"""Cloud + local speech-to-text provider registry.

A plain data table, not a framework — each provider is a `Provider` row
describing how to reach it (kind + base_url + key env var) and how to
explain it to a non-technical user (cost/quality/speed notes, signup link,
step-by-step key guide). `create_engine()` (registry.py) dispatches on
`kind`; the Settings UI (Phase B) renders badges straight from these
fields — no per-provider UI code needed to add a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    id: str
    display_name: str
    kind: str  # openai_compatible | gemini | deepgram | nvidia | local
    base_url: str  # "" for gemini/local (their engines hardcode the endpoint)
    default_model: str
    api_key_env: str  # "" for local
    signup_url: str  # "" for local
    cost_tier: str  # free | freemium | paid
    cost_note: str
    quality_tier: str  # good | better | best
    speed_note: str
    setup_steps: tuple[str, ...] = field(default_factory=tuple)
    max_upload_bytes: int = 0  # provider's per-request audio limit; 0 = none known


PROVIDERS: dict[str, Provider] = {
    "local": Provider(
        id="local",
        display_name="Local (on-device)",
        kind="local",
        base_url="",
        default_model="large-v3-turbo",
        api_key_env="",
        signup_url="",
        cost_tier="free",
        cost_note="Free — one-time ~1.5GB model download, then fully offline",
        quality_tier="best",
        speed_note="Depends on your GPU",
        setup_steps=(),
    ),
    "groq": Provider(
        id="groq",
        display_name="Groq (free, fast cloud)",
        kind="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        default_model="whisper-large-v3-turbo",
        api_key_env="GROQ_API_KEY",
        signup_url="https://console.groq.com/keys",
        cost_tier="free",
        cost_note="Free — 2,000 requests/day",
        quality_tier="better",
        speed_note="Instant",
        setup_steps=(
            "Open console.groq.com/keys (click 'Get a free key' below).",
            "Sign in with Google or GitHub — no credit card needed.",
            "Click 'Create API Key', give it any name, and copy the key.",
            "Paste it into the field below.",
        ),
        max_upload_bytes=25_000_000,  # Groq free-tier file limit is 25MB
    ),
    "gemini": Provider(
        id="gemini",
        display_name="Google Gemini (free)",
        kind="gemini",
        base_url="",
        default_model="gemini-2.5-flash-lite",
        api_key_env="GEMINI_API_KEY",
        signup_url="https://aistudio.google.com/apikey",
        cost_tier="free",
        cost_note="Free tier — generous daily quota",
        quality_tier="better",
        speed_note="Fast",
        setup_steps=(
            "Open aistudio.google.com/apikey (click 'Get a free key' below).",
            "Sign in with your Google account.",
            "Click 'Create API key' and copy it.",
            "Paste it into the field below.",
        ),
        # generateContent caps the request at 20MB and audio goes inline as
        # base64 (4/3 inflation) — ~14MB of WAV is the safe ceiling
        max_upload_bytes=14_000_000,
    ),
    "openai": Provider(
        id="openai",
        display_name="OpenAI (paid, high accuracy)",
        kind="openai_compatible",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-transcribe",
        api_key_env="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
        cost_tier="paid",
        cost_note="~$0.006/minute — add billing to your OpenAI account",
        quality_tier="best",
        speed_note="Fast",
        setup_steps=(
            "Open platform.openai.com/api-keys (click 'Get a key' below).",
            "Sign in and add a payment method (Settings > Billing) — required even for light use.",
            "Click 'Create new secret key' and copy it immediately (shown once).",
            "Paste it into the field below.",
        ),
        max_upload_bytes=25_000_000,  # OpenAI audio endpoints cap files at 25MB
    ),
    "nvidia": Provider(
        id="nvidia",
        display_name="NVIDIA (free credits, English-only)",
        kind="nvidia",
        base_url="",  # per-model NVCF function URL, hardcoded in nvidia_engine.py
        default_model="parakeet-ctc-1_1b-asr",
        api_key_env="NVIDIA_API_KEY",
        signup_url="https://build.nvidia.com",
        cost_tier="free",
        cost_note="Free credits on signup — no card. English dictation only",
        quality_tier="better",
        speed_note="Fast",
        setup_steps=(
            "Open build.nvidia.com (click 'Get a free key' below) and sign up free.",
            "Click your avatar (top right) → API Keys → 'Generate API Key'.",
            "Copy the key — it starts with 'nvapi-'.",
            "Paste it into the field below.",
        ),
        max_upload_bytes=5_000_000,  # NVCF HTTP route is sized for short clips (<5MB)
    ),
    "deepgram": Provider(
        id="deepgram",
        display_name="Deepgram (paid, best accuracy)",
        kind="deepgram",
        base_url="https://api.deepgram.com/v1",
        default_model="nova-3",
        api_key_env="DEEPGRAM_API_KEY",
        signup_url="https://console.deepgram.com",
        cost_tier="paid",
        cost_note="$200 free credit, then pay-as-you-go",
        quality_tier="best",
        speed_note="Fast",
        setup_steps=(
            "Open console.deepgram.com (click 'Get a key' below) and sign up.",
            "Go to API Keys in the left sidebar.",
            "Click 'Create a New API Key', copy it.",
            "Paste it into the field below.",
        ),
    ),
}


def get(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise KeyError(f"unknown speech engine {provider_id!r}") from None


def all_providers() -> list[Provider]:
    return list(PROVIDERS.values())


def cloud_providers() -> list[Provider]:
    return [p for p in PROVIDERS.values() if p.kind != "local"]


def is_cloud(provider_id: str) -> bool:
    return get(provider_id).kind != "local"


def choose(privacy_pref: str, budget_pref: str, specs) -> str:
    """Map the 2-question "Help me choose" answers to a provider id.

    privacy_pref: "private" (fully offline) | "cloud_ok" (cloud is fine)
    budget_pref:  "free" | "paid_ok"
    specs: anything with a `.vram_mb` attribute (sysinfo.SystemSpecs) — kept
    duck-typed so this module has zero dependency on sysinfo.py.

    Privacy always wins: "private" returns "local" regardless of budget
    (the local engine works with or without a GPU — just slower on CPU).
    """
    if privacy_pref not in ("private", "cloud_ok"):
        raise ValueError(f"privacy_pref must be 'private' or 'cloud_ok', got {privacy_pref!r}")
    if budget_pref not in ("free", "paid_ok"):
        raise ValueError(f"budget_pref must be 'free' or 'paid_ok', got {budget_pref!r}")
    if privacy_pref == "private":
        return "local"
    return "groq" if budget_pref == "free" else "openai"
