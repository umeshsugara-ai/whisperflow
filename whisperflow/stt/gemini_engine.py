"""BYOK cloud STT engine — Google Gemini audio transcription.

For users who can't (or don't want to) run a local model: bring your own
API key, and dictation audio is transcribed by a Gemini audio-input model
(default gemini-2.5-flash-lite; gemini-2.5-pro for higher accuracy).

PRIVACY: this engine sends the recorded audio to Google's API — the exact
opposite of the local engine's fully-on-device guarantee. It is opt-in via
[model].engine = "gemini", and the app logs a clear notice at startup.
Note: the "-tts" Gemini models are text-to-SPEECH and cannot transcribe;
this engine needs an audio-input model like gemini-2.5-flash-lite.

Uses plain REST (urllib, no SDK dependency): audio is wrapped as in-memory
WAV and sent as inlineData to generateContent with a strict verbatim-
transcription prompt.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import struct
import time
import urllib.error
import urllib.request

import numpy as np

from whisperflow.config import ModelConfig

from .base import RawResult, SttEngine

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SAMPLE_RATE = 16_000

TRANSCRIBE_PROMPT = (
    "Transcribe this audio VERBATIM. Output only the spoken words as text, "
    "with natural punctuation. Do not translate, summarize, describe the "
    "audio, or add any commentary. Preserve the language exactly as spoken, "
    "including Hindi (Devanagari script) and mixed Hindi-English (Hinglish "
    "in Latin script as spoken).{language_hint}{vocab_hint}"
)


def _float32_to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Minimal in-memory 16-bit PCM WAV (no filesystem, no extra deps)."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


class GeminiEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.model_id = cfg.cloud_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"Gemini engine needs an API key — set [model].api_key or ${self.cfg.api_key_env}"
            )
        if "tts" in self.model_id.lower():
            raise RuntimeError(
                f"{self.model_id!r} is a text-to-speech model and cannot transcribe audio; "
                "use an audio-input model such as gemini-2.5-flash-lite or gemini-2.5-pro"
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to Google (%s). "
            "Switch [model].engine to 'local' for fully on-device transcription.",
            self.model_id,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "",
        initial_prompt: str = "",
    ) -> RawResult:
        if not self._api_key:
            raise RuntimeError("engine not loaded — call load() first")

        t0 = time.perf_counter()
        duration_s = len(audio) / SAMPLE_RATE

        language_hint = f" The audio is primarily in language code '{language}'." if language else ""
        vocab_hint = (
            f" Domain terms that may occur (spell them exactly like this): {initial_prompt}."
            if initial_prompt
            else ""
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": TRANSCRIBE_PROMPT.format(language_hint=language_hint, vocab_hint=vocab_hint)},
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": base64.b64encode(_float32_to_wav_bytes(audio)).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }

        req = urllib.request.Request(
            API_URL.format(model=self.model_id),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
                # see openai_compatible_engine.py — urllib's default UA gets
                # blocked by Cloudflare-fronted APIs as bot traffic.
                "User-Agent": "WhisperFlow/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Gemini API unreachable: {exc}") from exc

        text = self._extract_text(body)
        return RawResult(
            text=text,
            language=language or "auto",
            language_probability=0.0,  # cloud API doesn't report one
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )

    @staticmethod
    def _extract_text(body: dict) -> str:
        try:
            parts = body["candidates"][0]["content"]["parts"]
            return " ".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError):
            feedback = body.get("promptFeedback", {})
            raise RuntimeError(f"Gemini returned no transcription (feedback: {feedback})") from None
