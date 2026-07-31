# -*- coding: utf-8 -*-
"""Auto-updater decision core + IO shell (whisperflow.updater).

Pure-logic tests only — no Tk, no network: urllib is monkeypatched with the
same _FakeResponse pattern as test_stt_base.py, disk work runs on tmp_path.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from whisperflow import updater
from whisperflow.updater import (
    UpdateCandidate,
    cleanup_updates_dir,
    download_asset,
    download_path,
    evaluate_release,
    fetch_latest_release,
    is_download_valid,
    is_newer,
    parse_version,
    pick_setup_asset,
    read_last_run_version,
    write_last_run_version,
)


# ---- parse_version ----------------------------------------------------------


def test_parse_version_accepts_v_prefix_and_plain():
    assert parse_version("v1.0.3") == (1, 0, 3)
    assert parse_version("1.2") == (1, 2)
    assert parse_version("V2") == (2,)


def test_parse_version_rejects_malformed():
    # a weird tag must mean "not an update", never a crash or a download loop
    assert parse_version("latest") is None
    assert parse_version("v1.0.3-beta") is None
    assert parse_version("") is None
    assert parse_version("v1..2") is None


# ---- is_newer ---------------------------------------------------------------


def test_is_newer_strictly_newer_only():
    assert is_newer("v1.0.3", "1.0.2") is True
    assert is_newer("v1.0.2", "1.0.2") is False
    assert is_newer("v1.0.1", "1.0.2") is False


def test_is_newer_pads_shorter_tuples():
    assert is_newer("v1.0.1", "1.0") is True
    assert is_newer("v1.0", "1.0.1") is False
    assert is_newer("v1.0", "1.0.0") is False  # equal after padding


def test_is_newer_false_on_malformed_either_side():
    assert is_newer("latest", "1.0.2") is False
    assert is_newer("v1.0.3", "dev") is False


# ---- pick_setup_asset -------------------------------------------------------


def _asset(name="WhisperFlow-Setup.exe", url="https://gh/dl/setup.exe", size=123):
    return {"name": name, "browser_download_url": url, "size": size}


def test_pick_setup_asset_finds_installer_among_others():
    assets = [_asset(name="WhisperFlow-Source.zip"), _asset(), _asset(name="notes.txt")]
    assert pick_setup_asset(assets) == ("https://gh/dl/setup.exe", 123)


def test_pick_setup_asset_case_insensitive():
    assert pick_setup_asset([_asset(name="whisperflow-setup.EXE")]) is not None


def test_pick_setup_asset_absent_or_invalid():
    assert pick_setup_asset([]) is None
    assert pick_setup_asset([_asset(name="Other.exe")]) is None
    assert pick_setup_asset([_asset(size=0)]) is None
    assert pick_setup_asset([_asset(url="")]) is None


# ---- evaluate_release -------------------------------------------------------


def _release(tag="v9.9.9", assets=None):
    return {"tag_name": tag, "assets": [_asset()] if assets is None else assets}


def test_evaluate_release_yields_candidate():
    cand = evaluate_release(_release(), "1.0.2")
    assert cand == UpdateCandidate(version="9.9.9", url="https://gh/dl/setup.exe", size=123)


def test_evaluate_release_none_when_not_newer():
    assert evaluate_release(_release(tag="v1.0.2"), "1.0.2") is None
    assert evaluate_release(_release(tag="v0.9"), "1.0.2") is None


def test_evaluate_release_none_on_missing_keys():
    assert evaluate_release({}, "1.0.2") is None
    assert evaluate_release({"tag_name": "v9.9.9"}, "1.0.2") is None
    assert evaluate_release(_release(assets=[]), "1.0.2") is None


# ---- download_path / is_download_valid / cleanup ---------------------------


def test_download_path_embeds_version(tmp_path):
    a = download_path(tmp_path, "1.0.3")
    b = download_path(tmp_path, "1.0.4")
    assert a.name == "WhisperFlow-Setup-1.0.3.exe"
    assert a != b


def test_is_download_valid_checks_exact_size(tmp_path):
    f = tmp_path / "x.exe"
    f.write_bytes(b"abc")
    assert is_download_valid(f, 3) is True
    assert is_download_valid(f, 4) is False
    assert is_download_valid(tmp_path / "missing.exe", 3) is False


def test_cleanup_updates_dir_sweeps_parts_and_old_exes(tmp_path):
    keep = tmp_path / "WhisperFlow-Setup-1.0.4.exe"
    keep.write_bytes(b"new")
    old = tmp_path / "WhisperFlow-Setup-1.0.3.exe"
    old.write_bytes(b"old")
    part = tmp_path / "WhisperFlow-Setup-1.0.4.exe.part"
    part.write_bytes(b"half")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")

    cleanup_updates_dir(tmp_path, keep=keep)

    assert keep.exists()
    assert unrelated.exists()
    assert not old.exists()
    assert not part.exists()


def test_cleanup_updates_dir_tolerates_missing_dir(tmp_path):
    cleanup_updates_dir(tmp_path / "nope")  # must not raise


# ---- fetch_latest_release (urllib monkeypatched) ---------------------------


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self, n: int = -1) -> bytes:
        data, self._data = self._data, b""
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_latest_release_parses_json_and_sets_user_agent(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["ua"] = req.get_header("User-agent")
        return _FakeResponse({"tag_name": "v1.0.3"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert fetch_latest_release() == {"tag_name": "v1.0.3"}
    assert seen["ua"] and seen["ua"].startswith("WhisperFlow/")


def test_fetch_latest_release_silent_on_http_403(monkeypatch):
    err = urllib.error.HTTPError("u", 403, "rate limited", {}, io.BytesIO(b""))
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: (_ for _ in ()).throw(err))
    assert fetch_latest_release() is None


def test_fetch_latest_release_silent_on_network_error(monkeypatch):
    err = urllib.error.URLError("no dns")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: (_ for _ in ()).throw(err))
    assert fetch_latest_release() is None


# ---- download_asset ---------------------------------------------------------


class _StreamingFake:
    def __init__(self, data: bytes, fail_after: int | None = None) -> None:
        self._buf = io.BytesIO(data)
        self._fail_after = fail_after
        self._reads = 0

    def read(self, n: int = -1) -> bytes:
        self._reads += 1
        if self._fail_after is not None and self._reads > self._fail_after:
            raise OSError("connection dropped")
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_asset_success_is_atomic(tmp_path, monkeypatch):
    body = b"x" * 100
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _StreamingFake(body))
    dest = tmp_path / "WhisperFlow-Setup-1.0.3.exe"
    assert download_asset("https://gh/dl", dest, expected_size=100) is True
    assert dest.read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_download_asset_size_mismatch_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: _StreamingFake(b"short"))
    dest = tmp_path / "WhisperFlow-Setup-1.0.3.exe"
    assert download_asset("https://gh/dl", dest, expected_size=100) is False
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_download_asset_midstream_error_cleans_part(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=0: _StreamingFake(b"x" * 200000, fail_after=1),
    )
    dest = tmp_path / "WhisperFlow-Setup-1.0.3.exe"
    assert download_asset("https://gh/dl", dest, expected_size=200000) is False
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


# ---- last-run-version marker ------------------------------------------------


def test_last_run_version_round_trip(tmp_path):
    assert read_last_run_version(tmp_path) is None
    write_last_run_version(tmp_path, "1.0.3")
    assert read_last_run_version(tmp_path) == "1.0.3"


# ---- [updates] config round-trip -------------------------------------------


def test_updates_config_round_trips(tmp_path):
    from whisperflow.config import Config, load_config, save_config

    cfg = Config()
    cfg.updates.auto_check = False
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.updates.auto_check is False


def test_updates_config_defaults_true_when_section_absent(tmp_path):
    from whisperflow.config import Config, load_config, save_config, serialize_config

    cfg = Config()
    path = tmp_path / "config.toml"
    text = serialize_config(cfg)
    # simulate an old config written before [updates] existed
    text = text[: text.index("[updates]")]
    path.write_text(text, encoding="utf-8")
    assert load_config(path).updates.auto_check is True
