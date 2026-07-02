"""faster-whisper engine — ports the proven AIOS transcribe pipeline.

Parameters carried over verbatim from the canonical pipeline (tested on
RTX 4060, ~6x realtime, EN/HI/Hinglish): beam_size=1, VAD filter on,
condition_on_previous_text=False, int8_float16 on CUDA.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from whisperflow.config import ModelConfig

from .base import RawResult, SttEngine
from .registry import resolve_model_id

log = logging.getLogger(__name__)

# Romanized-Hindi seed: whisper continues in the SCRIPT of the initial
# prompt, so a Latin-script Hindi seed makes Hinglish come out as
# "kya tum sun rahe ho" instead of mangled English or Devanagari.
HINGLISH_SEED = (
    "Haan toh main keh raha tha ki kal office jaana hai, aur phir hum log "
    "milke kaam khatam karenge, theek hai na?"
)


def resolve_language(language: str, initial_prompt: str) -> tuple[str, str]:
    """Map the user-facing language setting to whisper parameters.

    "hinglish" -> language "hi" + a romanized seed prepended to the prompt
    so output stays in Latin script. Everything else passes through.
    """
    if language.strip().lower() == "hinglish":
        return "hi", f"{HINGLISH_SEED} {initial_prompt}".strip()
    return language, initial_prompt


class FasterWhisperEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.model_id = resolve_model_id(cfg.name)
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        log.info("loading %s on %s (%s)...", self.model_id, self.cfg.device, self.cfg.compute_type)
        self._model = WhisperModel(
            self.model_id,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )
        log.info("model loaded in %.1fs", time.perf_counter() - t0)

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "",
        initial_prompt: str = "",
    ) -> RawResult:
        if self._model is None:
            raise RuntimeError("engine not loaded — call load() first")

        language, initial_prompt = resolve_language(language, initial_prompt)

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=language or None,
            initial_prompt=initial_prompt or None,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.perf_counter() - t0

        return RawResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_s=float(info.duration),
            transcribe_seconds=elapsed,
        )
