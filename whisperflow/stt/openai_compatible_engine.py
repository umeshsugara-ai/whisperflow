"""Generic OpenAI-compatible STT engine — covers Groq and OpenAI (and any
other provider that implements the same POST /audio/transcriptions
multipart endpoint). One engine, `base_url`/`model`/`key` differ per
provider (see providers.py).

Uses plain `urllib` multipart encoding (no `requests` dependency, matching
gemini_engine.py's "no SDK" pattern) — audio is wrapped as in-memory WAV,
same helper as the Gemini engine.
"""

from __future__ import annotations

import logging
import time
import urllib.request
import uuid

import numpy as np

from whisperflow.config import ModelConfig

from . import providers
from .base import RawResult, SttEngine, check_upload_size, request_json
from .faster_whisper_engine import resolve_language
from .gemini_engine import SAMPLE_RATE, _float32_to_wav_bytes

log = logging.getLogger(__name__)

# OpenAI/Groq cap the transcription prompt at 224 tokens; this is a rough
# character budget (not a real tokenizer) just to avoid an obvious 400.
MAX_PROMPT_CHARS = 800


def _multipart_body(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    """Build a multipart/form-data body + its Content-Type header value."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                "utf-8"
            )
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAICompatibleEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.provider = providers.get(cfg.engine)
        self.model_id = cfg.cloud_model or self.provider.default_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"{self.provider.display_name} needs an API key — set [model].api_key "
                f"or ${self.provider.api_key_env}"
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to %s (%s).",
            self.provider.display_name,
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

        # "hinglish" is WhisperFlow's own value — whisper APIs reject it with a
        # 400. Same mapping as the local engine: language "hi" + a Roman-script
        # seed prompt, so whisper keeps the output in Latin script.
        language, initial_prompt = resolve_language(language, initial_prompt)

        t0 = time.perf_counter()
        duration_s = len(audio) / SAMPLE_RATE

        fields = {"model": self.model_id, "response_format": "json", "temperature": "0"}
        if language:
            fields["language"] = language
        if initial_prompt:
            fields["prompt"] = initial_prompt[:MAX_PROMPT_CHARS]

        wav_bytes = _float32_to_wav_bytes(audio)
        check_upload_size(len(wav_bytes), self.provider)
        body, content_type = _multipart_body(
            fields=fields,
            file_field="file",
            filename="audio.wav",
            file_bytes=wav_bytes,
            content_type="audio/wav",
        )

        req = urllib.request.Request(
            f"{self.provider.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self._api_key}",
                # urllib's default UA ("Python-urllib/3.x") is a well-known
                # bot signature that Cloudflare-fronted APIs (Groq included)
                # block outright — surfaces as a generic 403 with no useful
                # detail, easily mistaken for a bad key. A real UA fixes it.
                "User-Agent": "WhisperFlow/1.0",
            },
        )
        result = request_json(
            req,
            provider_name=self.provider.display_name,
            signup_url=self.provider.signup_url,
        )

        text = result.get("text", "").strip()
        return RawResult(
            text=text,
            language=language or result.get("language", "auto"),
            language_probability=0.0,  # not reported by this API shape
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )
