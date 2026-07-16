# -*- coding: utf-8 -*-
"""Shared cloud-engine HTTP plumbing (base.request_json / check_upload_size):
retry-on-blip, upload-size guard, and friendly 401/429 mapping."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from whisperflow.stt import base
from whisperflow.stt.base import check_upload_size, request_json


def _req() -> urllib.request.Request:
    return urllib.request.Request("https://api.example.com/v1/x", data=b"{}")


def _http_error(code: int, body: bytes = b"detail") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example.com/v1/x", code, "msg", {}, io.BytesIO(body))


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_request_json_returns_decoded_body(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeResponse({"text": "hi"}))
    assert request_json(_req(), provider_name="X") == {"text": "hi"}


def test_request_json_retries_once_on_network_blip(monkeypatch):
    calls = {"n": 0}

    def flaky_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return _FakeResponse({"text": "recovered"})

    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    assert request_json(_req(), provider_name="X")["text"] == "recovered"
    assert calls["n"] == 2


def test_request_json_gives_up_after_second_network_failure(monkeypatch):
    calls = {"n": 0}

    def dead_urlopen(req, timeout=0):
        calls["n"] += 1
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", dead_urlopen)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="X API unreachable"):
        request_json(_req(), provider_name="X")
    assert calls["n"] == 2  # exactly one retry, never a loop


def test_request_json_maps_401_to_friendly_key_message_with_signup_url(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=0: (_ for _ in ()).throw(_http_error(401))
    )
    with pytest.raises(RuntimeError, match=r"rejected your API key.*example-signup\.com"):
        request_json(_req(), provider_name="X", signup_url="https://example-signup.com")


def test_request_json_does_not_retry_http_errors(monkeypatch):
    calls = {"n": 0}

    def unauthorized(req, timeout=0):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr("urllib.request.urlopen", unauthorized)
    with pytest.raises(RuntimeError):
        request_json(_req(), provider_name="X")
    assert calls["n"] == 1  # a bad key won't get better on retry


def test_request_json_maps_429_to_rate_limit_message(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=0: (_ for _ in ()).throw(_http_error(429))
    )
    with pytest.raises(RuntimeError, match="rate limit"):
        request_json(_req(), provider_name="X")


def test_request_json_keeps_raw_message_for_other_http_errors(monkeypatch):
    # 403 stays raw on purpose: on Cloudflare-fronted APIs it can mean
    # bot-blocking, not a bad key — the body excerpt is the useful signal
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=0: (_ for _ in ()).throw(_http_error(403, b"cf-blocked")),
    )
    with pytest.raises(RuntimeError, match="X API error 403: cf-blocked"):
        request_json(_req(), provider_name="X")


# ---- check_upload_size ----


class _FakeProvider:
    display_name = "Fake (cloud)"
    max_upload_bytes = 100


def test_check_upload_size_raises_over_limit():
    with pytest.raises(RuntimeError, match="too long for Fake"):
        check_upload_size(101, _FakeProvider())


def test_check_upload_size_passes_under_limit():
    check_upload_size(100, _FakeProvider())  # must not raise


def test_check_upload_size_skips_when_no_limit_known():
    p = _FakeProvider()
    p.max_upload_bytes = 0
    check_upload_size(10**9, p)  # must not raise


# ---- verify_provider_key (live key check used by the key-entry UIs) ----


def test_verify_provider_key_local_needs_no_key():
    from whisperflow.stt.registry import verify_provider_key

    assert verify_provider_key("local", "") is None


def test_verify_provider_key_success_returns_none(monkeypatch):
    from whisperflow.stt.registry import verify_provider_key

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _FakeResponse({"text": ""}))
    assert verify_provider_key("groq", "good-key") is None


def test_verify_provider_key_bad_key_returns_friendly_message(monkeypatch):
    from whisperflow.stt.registry import verify_provider_key

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda req, timeout=0: (_ for _ in ()).throw(_http_error(401))
    )
    msg = verify_provider_key("groq", "typo-key")
    assert msg is not None and "rejected your API key" in msg
