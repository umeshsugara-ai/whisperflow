"""STT engine interface — the seam for pluggable models.

FasterWhisperEngine is the only MVP implementation; a future Parakeet/NeMo
engine implements the same two methods and registers in registry.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RawResult:
    text: str
    language: str
    language_probability: float
    duration_s: float
    transcribe_seconds: float  # wall-clock inference time


class SttEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights (called once at daemon startup)."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,  # float32 mono @16k
        language: str = "",  # "" = auto-detect
        initial_prompt: str = "",  # vocabulary bias
    ) -> RawResult: ...
