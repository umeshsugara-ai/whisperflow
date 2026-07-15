# -*- coding: utf-8 -*-
"""Deepgram STT engine — request building + response parsing, mocked HTTP."""

import io
import json

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.deepgram_engine import DeepgramEngine


def cfg(**kw) -> ModelConfig:
    defaults = dict(
        engine="deepgram",
        api_key="test-dg-key",
        cloud_model="nova-3",
        api_key_env="DEEPGRAM_API_KEY"
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    engine = DeepgramEngine(cfg(api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_transcribe_sends_token_auth_and_parses_response(monkeypatch):
    engine = DeepgramEngine(cfg())
    engine.load()

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "results": {
                        "channels": [
                            {"alternatives": [{"transcript": "hello from deepgram"}]}
                        ]
                    }
                }
            ).encode("utf-8")

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

    audio = np.zeros(16000, dtype=np.float32)  # 1s
    result = engine.transcribe(audio)

    assert result.text == "hello from deepgram"
    assert result.duration_s == 1.0
    assert "nova-3" in captured["url"]
    assert captured["headers"].get("Authorization") == "Token test-dg-key"
    assert captured["body"].startswith(b"RIFF")  # raw WAV body, not multipart


def test_transcribe_raises_readable_error_on_http_error(monkeypatch):
    import urllib.error

    engine = DeepgramEngine(cfg())
    engine.load()

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad audio"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Deepgram API error 400"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_registry_dispatches_deepgram():
    from whisperflow.stt.registry import create_engine

    assert isinstance(create_engine(cfg()), DeepgramEngine)
