# -*- coding: utf-8 -*-
"""Cloud STT provider registry — data only, no network."""

import pytest

from whisperflow.stt import providers


def test_groq_is_registered_and_openai_compatible():
    p = providers.get("groq")
    assert p.kind == "openai_compatible"
    assert p.base_url == "https://api.groq.com/openai/v1"
    assert p.default_model == "whisper-large-v3-turbo"
    assert p.api_key_env == "GROQ_API_KEY"
    assert p.cost_tier == "free"
    assert p.setup_steps  # non-empty guide


def test_gemini_is_registered_with_cheap_default():
    p = providers.get("gemini")
    assert p.kind == "gemini"
    assert p.default_model == "gemini-2.5-flash-lite"
    assert p.api_key_env == "GEMINI_API_KEY"


def test_openai_is_registered_and_paid():
    p = providers.get("openai")
    assert p.kind == "openai_compatible"
    assert p.base_url == "https://api.openai.com/v1"
    assert p.cost_tier == "paid"


def test_deepgram_is_registered():
    p = providers.get("deepgram")
    assert p.kind == "deepgram"
    assert p.api_key_env == "DEEPGRAM_API_KEY"


def test_local_is_registered_and_excluded_from_cloud_list():
    p = providers.get("local")
    assert p.kind == "local"
    ids = [x.id for x in providers.cloud_providers()]
    assert "local" not in ids
    assert "groq" in ids


def test_get_unknown_id_raises_clear_error():
    with pytest.raises(KeyError, match="unknown speech engine 'nonsense'"):
        providers.get("nonsense")


def test_is_cloud():
    assert providers.is_cloud("groq") is True
    assert providers.is_cloud("local") is False


def test_all_providers_includes_local_and_every_cloud_id():
    ids = {p.id for p in providers.all_providers()}
    assert ids == {"local", "groq", "gemini", "openai", "deepgram"}
