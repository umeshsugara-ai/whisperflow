# -*- coding: utf-8 -*-
"""Gemini BYOK engine — request building + response parsing with a mocked API."""

import io
import json
import wave

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.gemini_engine import GeminiEngine, _float32_to_wav_bytes
from whisperflow.stt.registry import create_engine


def cfg(**kw) -> ModelConfig:
    defaults = dict(engine="gemini", api_key="test-key-123")
    defaults.update(kw)
    return ModelConfig(**defaults)


def test_registry_dispatches_gemini():
    engine = create_engine(cfg())
    assert isinstance(engine, GeminiEngine)


def test_wav_bytes_are_valid_wav():
    samples = np.zeros(16000, dtype=np.float32)
    data = _float32_to_wav_bytes(samples)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 16000


def test_load_rejects_tts_model():
    engine = GeminiEngine(cfg(cloud_model="gemini-2.5-pro-preview-tts"))
    with pytest.raises(RuntimeError, match="text-to-speech"):
        engine.load()


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    engine = GeminiEngine(ModelConfig(engine="gemini", api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_transcribe_parses_response(monkeypatch):
    engine = GeminiEngine(cfg())
    engine.load()

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"text": "नमस्ते, hello world"}]}}]}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    audio = np.zeros(32000, dtype=np.float32)  # 2s
    result = engine.transcribe(audio, language="hi", initial_prompt="Vidysea, Pathlynks")

    assert result.text == "नमस्ते, hello world"
    assert result.duration_s == 2.0
    assert "gemini-2.5-flash-lite" in captured["url"]
    assert captured["headers"].get("X-goog-api-key") == "test-key-123"
    assert "python-urllib" not in captured["headers"].get("User-agent", "").lower()
    prompt_text = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "VERBATIM" in prompt_text
    assert "'hi'" in prompt_text  # language hint forwarded
    assert "Vidysea" in prompt_text  # vocabulary forwarded
    assert captured["body"]["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "audio/wav"


def test_transcribe_raises_on_empty_candidates(monkeypatch):
    engine = GeminiEngine(cfg())
    engine.load()

    class FakeResponse:
        def read(self):
            return json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: FakeResponse())
    with pytest.raises(RuntimeError, match="no transcription"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_hinglish_language_resolution():
    from whisperflow.stt.faster_whisper_engine import HINGLISH_SEED, resolve_language

    lang, prompt = resolve_language("hinglish", "Vidysea, Pathlynks")
    assert lang == "hi"
    assert prompt.startswith(HINGLISH_SEED)
    assert "Vidysea" in prompt
    # passthrough for everything else
    assert resolve_language("en", "x") == ("en", "x")
    assert resolve_language("", "") == ("", "")
    assert resolve_language("HINGLISH", "") == ("hi", HINGLISH_SEED)


def test_config_validation_requires_key_for_cloud(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from whisperflow.config import ConfigError, load_config

    bad = tmp_path / "config.toml"
    bad.write_text('[model]\nengine = "gemini"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="API key"):
        load_config(bad)
