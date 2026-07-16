"""STT engine dispatch: provider id -> engine instance.

`ModelConfig.engine` is a provider id from `whisperflow.stt.providers`
(e.g. "local", "groq", "gemini", "openai", "deepgram"). Which concrete
engine class handles it is decided by the provider's `kind` — adding a new
provider that reuses an existing kind (e.g. another openai_compatible
service) needs zero changes here, just a new providers.py row.
"""

from __future__ import annotations

from whisperflow.config import ModelConfig

from . import providers
from .base import SttEngine

# friendly local-model name -> HF repo id (all CTranslate2/faster-whisper format)
FASTER_WHISPER_MODELS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # default; ~1.5GB
    "large-v3": "Systran/faster-whisper-large-v3",  # best Hindi accuracy; ~2.9GB, ~5x slower
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def resolve_model_id(name: str) -> str:
    """Friendly registry name or raw HF repo id passthrough."""
    return FASTER_WHISPER_MODELS.get(name, name)


_ENGINE_BY_KIND = {
    "gemini": "whisperflow.stt.gemini_engine.GeminiEngine",
    "openai_compatible": "whisperflow.stt.openai_compatible_engine.OpenAICompatibleEngine",
    "deepgram": "whisperflow.stt.deepgram_engine.DeepgramEngine",
    "local": "whisperflow.stt.faster_whisper_engine.FasterWhisperEngine",
}


def _try_import_faster_whisper() -> None:
    """Indirection so tests can simulate 'not importable' without actually
    uninstalling the package. Raises ImportError exactly like a real
    failed import would."""
    import faster_whisper  # noqa: F401


def local_inference_available() -> bool:
    """True if this build can actually run the local (on-device) engine right
    now — faster_whisper is importable (dev checkout only; the distributed
    installer is cloud-only and never bundles it). Used by the engine picker
    to honestly show/hide Local instead of letting a user pick a dead end."""
    try:
        _try_import_faster_whisper()
        return True
    except ImportError:
        return False


def _ensure_local_available() -> None:
    """No-op on a dev checkout, where faster_whisper is already importable.
    On the distributed cloud-only build, fails FAST with a friendly error —
    app.py's startup error handler catches this and reopens the engine
    picker immediately so the user isn't left staring at nothing."""
    try:
        _try_import_faster_whisper()
    except ImportError:
        raise RuntimeError(
            "Local (on-device) mode isn't included in this install — "
            "switch to a free cloud engine like Groq in Settings."
        ) from None


def create_engine(cfg: ModelConfig) -> SttEngine:
    provider = providers.get(cfg.engine)
    if provider.kind == "local":
        _ensure_local_available()
    module_path, class_name = _ENGINE_BY_KIND[provider.kind].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(cfg)
