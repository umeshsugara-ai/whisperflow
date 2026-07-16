# -*- coding: utf-8 -*-
"""Local-inference pack download/verify/extract/activate — mocked network, no real download."""

import hashlib
import io
import sys
import urllib.error
import zipfile

import pytest

from whisperflow import localpack


def _fake_zip_bytes(filename: str = "faster_whisper/__init__.py", content: bytes = b"# fake\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def test_is_installed_false_when_marker_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(localpack, "PACK_DIR", tmp_path / "local-pack")
    assert localpack.is_installed() is False


def test_is_installed_true_when_marker_present(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    pack_dir.mkdir()
    (pack_dir / ".pack_complete").write_text("1", encoding="utf-8")
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)
    assert localpack.is_installed() is True


def test_ensure_installed_downloads_verifies_extracts(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)

    zip_bytes = _fake_zip_bytes()
    sha_hex = hashlib.sha256(zip_bytes).hexdigest()

    def fake_urlopen(url, timeout=0):
        class FakeResp:
            def read(self_inner):
                if url == localpack.pack_url():
                    return zip_bytes
                if url == localpack.pack_sha_url():
                    return sha_hex.encode("ascii")
                raise AssertionError(f"unexpected URL {url}")

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    progress_msgs = []
    localpack.ensure_installed(progress_cb=progress_msgs.append)

    assert localpack.is_installed() is True
    assert (pack_dir / "faster_whisper" / "__init__.py").read_bytes() == b"# fake\n"
    assert any("download" in m.lower() for m in progress_msgs)


def test_ensure_installed_noop_when_already_installed(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    pack_dir.mkdir()
    (pack_dir / ".pack_complete").write_text("1", encoding="utf-8")
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)

    def fail_urlopen(*a, **kw):
        raise AssertionError("should not be called — pack already installed")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    localpack.ensure_installed()  # must not raise, must not touch the network


def test_ensure_installed_raises_on_checksum_mismatch(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)

    zip_bytes = _fake_zip_bytes()
    wrong_sha = "0" * 64

    def fake_urlopen(url, timeout=0):
        class FakeResp:
            def read(self_inner):
                return zip_bytes if url == localpack.pack_url() else wrong_sha.encode("ascii")

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="checksum"):
        localpack.ensure_installed()
    assert localpack.is_installed() is False  # must NOT mark a bad download as complete
    assert not pack_dir.exists() or not (pack_dir / ".pack_complete").exists()


def test_ensure_installed_raises_on_download_failure(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)

    def fail_urlopen(url, timeout=0):
        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    with pytest.raises(RuntimeError, match="download"):
        localpack.ensure_installed()
    assert localpack.is_installed() is False  # must NOT mark a failed download as complete


def test_activate_prepends_pack_dir_to_syspath(tmp_path, monkeypatch):
    pack_dir = tmp_path / "local-pack"
    pack_dir.mkdir()
    (pack_dir / ".pack_complete").write_text("1", encoding="utf-8")
    monkeypatch.setattr(localpack, "PACK_DIR", pack_dir)
    original_path = list(sys.path)
    try:
        localpack.activate()
        assert str(pack_dir) in sys.path
        assert sys.path[0] == str(pack_dir)
        localpack.activate()  # idempotent — calling twice doesn't duplicate the entry
        assert sys.path.count(str(pack_dir)) == 1
    finally:
        sys.path[:] = original_path


def test_activate_raises_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(localpack, "PACK_DIR", tmp_path / "local-pack")
    with pytest.raises(RuntimeError, match="not installed"):
        localpack.activate()


def test_registry_local_dispatch_is_noop_when_faster_whisper_importable(monkeypatch):
    """The common case (dev checkout / full build): faster_whisper is
    already importable, so the pack machinery must never be touched."""
    from whisperflow.stt import registry

    called = {"ensure": False, "activate": False}
    monkeypatch.setattr(localpack, "ensure_installed", lambda *a, **kw: called.__setitem__("ensure", True))
    monkeypatch.setattr(localpack, "activate", lambda: called.__setitem__("activate", True))
    # faster_whisper IS installed in this dev/test environment (it's a real
    # project dependency — see requirements.txt), so this exercises the
    # real "already importable" branch, not a mock.
    registry._ensure_local_available()
    assert called == {"ensure": False, "activate": False}


def test_registry_local_dispatch_falls_back_to_pack_when_not_importable(monkeypatch):
    from whisperflow.stt import registry

    def fake_import_faster_whisper():
        raise ImportError("no module named faster_whisper")

    monkeypatch.setattr(registry, "_try_import_faster_whisper", fake_import_faster_whisper)
    monkeypatch.setattr(localpack, "is_installed", lambda: True)
    activated = {"called": False}
    monkeypatch.setattr(localpack, "activate", lambda: activated.__setitem__("called", True))
    registry._ensure_local_available()
    assert activated["called"] is True


def test_registry_local_dispatch_triggers_download_when_pack_missing(monkeypatch):
    """The pack isn't installed yet -> _ensure_local_available() should
    attempt to download it (not just raise), then activate it on success."""
    from whisperflow.stt import registry

    def fake_import_faster_whisper():
        raise ImportError("no module named faster_whisper")

    monkeypatch.setattr(registry, "_try_import_faster_whisper", fake_import_faster_whisper)
    monkeypatch.setattr(localpack, "is_installed", lambda: False)
    called = {"ensure": False, "activate": False}
    monkeypatch.setattr(localpack, "ensure_installed", lambda progress_cb=None: called.__setitem__("ensure", True))
    monkeypatch.setattr(localpack, "activate", lambda: called.__setitem__("activate", True))
    registry._ensure_local_available()
    assert called == {"ensure": True, "activate": True}


def test_registry_local_dispatch_raises_friendly_error_when_download_fails(monkeypatch):
    from whisperflow.stt import registry

    def fake_import_faster_whisper():
        raise ImportError("no module named faster_whisper")

    def fake_ensure_installed(progress_cb=None):
        raise RuntimeError("failed to download local-inference pack: simulated network failure")

    monkeypatch.setattr(registry, "_try_import_faster_whisper", fake_import_faster_whisper)
    monkeypatch.setattr(localpack, "is_installed", lambda: False)
    monkeypatch.setattr(localpack, "ensure_installed", fake_ensure_installed)
    with pytest.raises(RuntimeError, match="one-time download"):
        registry._ensure_local_available()
