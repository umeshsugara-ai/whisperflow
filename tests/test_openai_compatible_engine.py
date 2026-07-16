# -*- coding: utf-8 -*-
"""Generic OpenAI-compatible STT engine (Groq, OpenAI) — request/response, mocked HTTP."""

import io
import json

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.openai_compatible_engine import OpenAICompatibleEngine, _multipart_body


def cfg(**kw) -> ModelConfig:
    from whisperflow.stt import providers

    defaults = dict(engine="groq", api_key="test-groq-key")
    defaults.update(kw)

    # Use provider's default model for cloud engines (unless explicitly overridden)
    if 'cloud_model' not in kw:
        provider = providers.get(defaults['engine'])
        if provider.kind != 'local':
            defaults['cloud_model'] = provider.default_model

    return ModelConfig(**defaults)


def test_multipart_body_contains_field_and_file():
    body, content_type = _multipart_body(
        fields={"model": "whisper-large-v3-turbo", "language": "en"},
        file_field="file",
        filename="audio.wav",
        file_bytes=b"RIFF....",
        content_type="audio/wav",
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1]
    text = body.decode("latin-1")
    assert boundary in text
    assert 'name="model"' in text
    assert "whisper-large-v3-turbo" in text
    assert 'name="file"; filename="audio.wav"' in text
    assert "Content-Type: audio/wav" in text
    assert text.rstrip().endswith(f"--{boundary}--")


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    engine = OpenAICompatibleEngine(cfg(api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_load_resolves_provider_and_model():
    engine = OpenAICompatibleEngine(cfg())
    engine.load()
    assert engine.model_id == "whisper-large-v3-turbo"
    assert engine.provider.id == "groq"


def test_transcribe_sends_bearer_auth_and_parses_text(monkeypatch):
    engine = OpenAICompatibleEngine(cfg())
    engine.load()

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps({"text": "hello from groq"}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    audio = np.zeros(32000, dtype=np.float32)  # 2s @ 16kHz
    result = engine.transcribe(audio, language="en", initial_prompt="Vidysea")

    assert result.text == "hello from groq"
    assert result.duration_s == 2.0
    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert captured["headers"].get("Authorization") == "Bearer test-groq-key"
    # a default urllib User-Agent ("Python-urllib/3.x") gets blocked by
    # Cloudflare-fronted APIs (Groq included) as bot traffic — a real UA
    # must be sent, or every Groq request 403s regardless of a valid key.
    assert "python-urllib" not in captured["headers"].get("User-agent", "").lower()
    assert b"whisper-large-v3-turbo" in captured["body"]
    assert b"Vidysea" in captured["body"]


def test_transcribe_raises_readable_error_on_http_401(monkeypatch):
    import urllib.error

    engine = OpenAICompatibleEngine(cfg())
    engine.load()

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    # 401 maps to a friendly "fix your key" message with the signup link —
    # not the raw HTTP status the user can't act on
    with pytest.raises(RuntimeError, match="Groq.*rejected your API key.*console.groq.com"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_registry_dispatches_openai_compatible_for_groq_and_openai():
    from whisperflow.stt.registry import create_engine

    assert isinstance(create_engine(cfg(engine="groq")), OpenAICompatibleEngine)
    assert isinstance(create_engine(cfg(engine="openai", api_key="k")), OpenAICompatibleEngine)
