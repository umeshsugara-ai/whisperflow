# -*- coding: utf-8 -*-
"""NVIDIA (build.nvidia.com / NVCF) STT engine — request building + response
parsing, mocked HTTP."""

import io
import json

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.nvidia_engine import FUNCTION_IDS, NvidiaEngine


def cfg(**kw) -> ModelConfig:
    defaults = dict(
        engine="nvidia",
        api_key="nvapi-test-key",
        cloud_model="parakeet-ctc-1_1b-asr",
        api_key_env="NVIDIA_API_KEY",
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    engine = NvidiaEngine(cfg(api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_load_rejects_unknown_model():
    engine = NvidiaEngine(cfg(cloud_model="whisper-large-v3"))
    # whisper-large-v3 IS hosted on build.nvidia.com but only over gRPC,
    # which this app doesn't speak — the error must name what IS supported
    with pytest.raises(RuntimeError, match="parakeet-ctc-1_1b-asr"):
        engine.load()


def test_transcribe_posts_multipart_to_the_models_function_url(monkeypatch):
    engine = NvidiaEngine(cfg())
    engine.load()

    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data
        return FakeResponse({"text": "hello from nvidia"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    audio = np.zeros(16000, dtype=np.float32)  # 1s
    result = engine.transcribe(audio)

    assert result.text == "hello from nvidia"
    assert result.duration_s == 1.0
    assert (
        captured["url"]
        == f"https://{FUNCTION_IDS['parakeet-ctc-1_1b-asr']}.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions"
    )
    assert captured["headers"].get("Authorization") == "Bearer nvapi-test-key"
    assert "python-urllib" not in captured["headers"].get("User-agent", "").lower()
    assert b'name="language"' in captured["body"]
    assert b"en-US" in captured["body"]  # NVCF wants BCP-47, not whisper-style "en"
    assert b'name="file"' in captured["body"]


def test_transcribe_parses_riva_shaped_response(monkeypatch):
    engine = NvidiaEngine(cfg())
    engine.load()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=0: FakeResponse(
            {"results": [{"alternatives": [{"transcript": "riva shape"}]}]}
        ),
    )
    assert engine.transcribe(np.zeros(16000, dtype=np.float32)).text == "riva shape"


def test_transcribe_raises_readable_error_when_no_transcription(monkeypatch):
    engine = NvidiaEngine(cfg())
    engine.load()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=0: FakeResponse({"unexpected": True})
    )
    with pytest.raises(RuntimeError, match="NVIDIA returned no transcription"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_transcribe_raises_readable_error_on_http_error(monkeypatch):
    import urllib.error

    engine = NvidiaEngine(cfg())
    engine.load()

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad audio"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="NVIDIA API error 400"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_recording_over_5mb_fails_before_upload(monkeypatch):
    engine = NvidiaEngine(cfg())
    engine.load()

    def must_not_be_called(req, timeout=0):
        raise AssertionError("upload must not happen for an oversized recording")

    monkeypatch.setattr("urllib.request.urlopen", must_not_be_called)
    # 16kHz * 16-bit mono = 32KB/s -> >5MB needs >156s of audio
    audio = np.zeros(16000 * 160, dtype=np.float32)
    with pytest.raises(RuntimeError, match="too long"):
        engine.transcribe(audio)


def test_registry_dispatches_nvidia():
    from whisperflow.stt.registry import create_engine

    assert isinstance(create_engine(cfg()), NvidiaEngine)
