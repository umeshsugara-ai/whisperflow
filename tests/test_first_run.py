# tests/test_first_run.py
# -*- coding: utf-8 -*-
"""Pure decision logic backing the first-run chooser dialog."""

from whisperflow.ui.first_run import fallback_engine, provider_already_has_key


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


def test_fallback_engine_is_local_when_this_build_can_run_it():
    assert fallback_engine(True) == "local"


def test_fallback_engine_defers_on_cloud_only_build():
    # regression: skipping key entry / closing the chooser on a cloud-only
    # install used to save engine="local" — an engine that build can't run,
    # dead-ending straight into the startup recovery loop. Defer instead.
    assert fallback_engine(False) is None
