"""Deepgram STT engine — raw-WAV POST to /v1/listen (not multipart; Deepgram
takes the audio bytes directly as the request body with a Content-Type
header), Token auth. Same "plain urllib REST" pattern as the other cloud
engines.
"""

from __future__ import annotations

import logging
import time
import urllib.parse
import urllib.request

import numpy as np

from whisperflow.config import ModelConfig

from . import providers
from .base import RawResult, SttEngine, check_upload_size, request_json
from .gemini_engine import SAMPLE_RATE, _float32_to_wav_bytes

log = logging.getLogger(__name__)


class DeepgramEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.provider = providers.get("deepgram")
        self.model_id = cfg.cloud_model or self.provider.default_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"Deepgram needs an API key — set [model].api_key or ${self.provider.api_key_env}"
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to Deepgram (%s).", self.model_id
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

        params = {"model": self.model_id, "punctuate": "true"}
        if language:
            params["language"] = language
        query = urllib.parse.urlencode(params)

        wav_bytes = _float32_to_wav_bytes(audio)
        check_upload_size(len(wav_bytes), self.provider)
        req = urllib.request.Request(
            f"{self.provider.base_url}/listen?{query}",
            data=wav_bytes,
            headers={
                "Content-Type": "audio/wav",
                "Authorization": f"Token {self._api_key}",
                # see openai_compatible_engine.py — urllib's default UA gets
                # blocked by Cloudflare-fronted APIs as bot traffic.
                "User-Agent": "WhisperFlow/1.0",
            },
        )
        body = request_json(
            req,
            provider_name="Deepgram",
            signup_url=self.provider.signup_url,
        )

        text = self._extract_text(body)
        return RawResult(
            text=text,
            language=language or "auto",
            language_probability=0.0,
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )

    @staticmethod
    def _extract_text(body: dict) -> str:
        try:
            return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Deepgram returned no transcription: {str(body)[:300]}") from None
