# tests/test_first_run.py
# -*- coding: utf-8 -*-
"""Pure decision logic backing the first-run chooser dialog."""

from whisperflow.ui.first_run import provider_already_has_key


def test_provider_already_has_key_true_when_env_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "some-key")
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("groq")) is True


def test_provider_already_has_key_false_when_unset(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("groq")) is False


def test_provider_already_has_key_false_for_local():
    from whisperflow.stt import providers

    assert provider_already_has_key(providers.get("local")) is False
