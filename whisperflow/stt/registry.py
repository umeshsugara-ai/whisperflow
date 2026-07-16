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


def _ensure_local_available() -> None:
    """No-op on a dev checkout or WF_BUILD=full frozen build, where
    faster_whisper is already importable. Falls back to the on-demand
    local-inference pack only when it isn't (a WF_BUILD=cloud build) —
    downloading it synchronously on first use if needed."""
    try:
        _try_import_faster_whisper()
        return
    except ImportError:
        pass

    import logging

    from whisperflow import localpack

    log = logging.getLogger(__name__)
    if not localpack.is_installed():
        try:
            localpack.ensure_installed(progress_cb=lambda msg: log.warning("%s", msg))
        except RuntimeError as exc:
            raise RuntimeError(
                "Local (on-device) mode needs a one-time download (~800MB) and it "
                f"failed: {exc}. Open Settings and pick Local again to retry, or "
                "switch to a free cloud engine like Groq in the meantime."
            ) from exc
    localpack.activate()


def create_engine(cfg: ModelConfig) -> SttEngine:
    provider = providers.get(cfg.engine)
    if provider.kind == "local":
        _ensure_local_available()
    module_path, class_name = _ENGINE_BY_KIND[provider.kind].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(cfg)
