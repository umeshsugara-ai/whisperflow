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


def create_engine(cfg: ModelConfig) -> SttEngine:
    provider = providers.get(cfg.engine)
    module_path, class_name = _ENGINE_BY_KIND[provider.kind].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(cfg)
