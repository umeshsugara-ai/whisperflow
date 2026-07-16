"""NVIDIA (build.nvidia.com) STT engine — NVCF-hosted Riva ASR over plain
HTTPS. Same "stdlib urllib, no SDK" pattern as the other cloud engines.

NVCF's offline HTTP route is a multipart POST to
    https://{function-id}.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions
with `Authorization: Bearer nvapi-...`. Each hosted model has its own
function-id UUID (FUNCTION_IDS below, from the model's build.nvidia.com API
page). Only parakeet-ctc-1.1b-asr exposes this HTTP route today — NVIDIA's
multilingual models (whisper-large-v3, canary-1b) are gRPC-only on NVCF,
which this app deliberately doesn't ship a client for. Parakeet is
English-only; load() warns when the configured language isn't English.
"""

from __future__ import annotations

import logging
import time
import urllib.request

import numpy as np

from whisperflow.config import ModelConfig

from . import providers
from .base import RawResult, SttEngine, check_upload_size, request_json
from .gemini_engine import SAMPLE_RATE, _float32_to_wav_bytes
from .openai_compatible_engine import _multipart_body

log = logging.getLogger(__name__)

API_URL = "https://{function_id}.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions"

# model slug -> NVCF function-id (from the model's build.nvidia.com API page)
FUNCTION_IDS: dict[str, str] = {
    "parakeet-ctc-1_1b-asr": "1598d209-5e27-4d3c-8079-4751568b1081",
}


class NvidiaEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.provider = providers.get("nvidia")
        self.model_id = cfg.cloud_model or self.provider.default_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"NVIDIA needs an API key — set [model].api_key or ${self.provider.api_key_env}"
            )
        if self.model_id not in FUNCTION_IDS:
            raise RuntimeError(
                f"unknown NVIDIA model {self.model_id!r} — supported: "
                + ", ".join(sorted(FUNCTION_IDS))
            )
        if self.cfg.language and self.cfg.language != "en":
            log.warning(
                "NVIDIA's %s model is English-only — [model].language=%r will be "
                "transcribed as English. Pick Groq or Gemini for Hindi/Hinglish.",
                self.model_id,
                self.cfg.language,
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to NVIDIA (%s).", self.model_id
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

        wav_bytes = _float32_to_wav_bytes(audio)
        check_upload_size(len(wav_bytes), self.provider)
        body, content_type = _multipart_body(
            # NVCF wants BCP-47 ("en-US"), not whisper-style "en"; parakeet
            # is English-only so this is always en-US regardless of config.
            fields={"language": "en-US"},
            file_field="file",
            filename="audio.wav",
            file_bytes=wav_bytes,
            content_type="audio/wav",
        )

        req = urllib.request.Request(
            API_URL.format(function_id=FUNCTION_IDS[self.model_id]),
            data=body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self._api_key}",
                # see openai_compatible_engine.py — urllib's default UA gets
                # blocked by Cloudflare-fronted APIs as bot traffic.
                "User-Agent": "WhisperFlow/1.0",
            },
        )
        result = request_json(
            req,
            provider_name="NVIDIA",
            signup_url=self.provider.signup_url,
        )

        text = self._extract_text(result)
        return RawResult(
            text=text,
            language="en",
            language_probability=0.0,
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )

    @staticmethod
    def _extract_text(body: dict) -> str:
        if isinstance(body.get("text"), str):
            return body["text"].strip()
        # some NIM builds return the Riva shape instead of the flat one
        try:
            return body["results"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"NVIDIA returned no transcription: {str(body)[:300]}") from None
