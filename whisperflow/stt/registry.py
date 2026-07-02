"""Model registry: friendly name -> (engine, HF repo id).

Lifted from the proven AIOS transcribe pipeline. Any name not in the
registry is passed through as a raw HuggingFace CTranslate2 repo id, so
users can point at any compatible model without a code change.
"""

from __future__ import annotations

from whisperflow.config import ModelConfig

from .base import SttEngine

# friendly name -> HF repo id (all CTranslate2/faster-whisper format)
FASTER_WHISPER_MODELS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # default; ~1.5GB
    "large-v3": "Systran/faster-whisper-large-v3",  # best Hindi accuracy; ~2.9GB, ~5x slower
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def resolve_model_id(name: str) -> str:
    """Friendly registry name or raw HF repo id passthrough."""
    return FASTER_WHISPER_MODELS.get(name, name)


def create_engine(cfg: ModelConfig) -> SttEngine:
    if cfg.engine == "gemini":
        from .gemini_engine import GeminiEngine

        return GeminiEngine(cfg)
    from .faster_whisper_engine import FasterWhisperEngine

    return FasterWhisperEngine(cfg)
