# Phase C: Slim Installer — Light Cloud Base + Downloadable Local Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the ~1GB installer into a light (~150MB) cloud-ready base plus an optional ~800MB "local inference pack" that only downloads if/when a user actually picks the Local engine — so a cloud-only user's install/update is 6-7x smaller.

**Architecture:** A new PyInstaller build variant (`WF_BUILD=cloud`, env-driven) excludes `ctranslate2`/`faster_whisper`/the CUDA wheels from the frozen exe entirely. A new `whisperflow/localpack.py` module downloads a pre-built pack zip (published as a separate GitHub release asset, built from the identical Python/PyInstaller environment) into `%LOCALAPPDATA%\WhisperFlow\local-pack\` on first local-engine use, verifies its SHA-256 against a sidecar file, and prepends it to `sys.path` before `faster_whisper` is imported. `registry.py`'s local-engine dispatch tries a plain import first (a no-op on the existing full/dev build) and only engages the pack machinery as a fallback when `faster_whisper` isn't already importable (i.e., only on a cloud-base build).

**Tech Stack:** stdlib `urllib.request`/`zipfile`/`hashlib` (no new pip dependency), existing PyInstaller/Inno Setup toolchain.

## Global Constraints

- No new pip dependencies for the download/verify/extract logic — stdlib only (`urllib.request`, `zipfile`, `hashlib`), matching this project's established "plain stdlib, no SDK" pattern (see the cloud STT engines from Phases A/B).
- The pack machinery must be a **no-op fallback**, never a forced path: on a dev checkout or the existing "full" build (where `faster_whisper` is already importable via the normal environment/bundle), local-engine dispatch behaves exactly as it does today — zero pack-download, zero `sys.path` mutation. Only a `WF_BUILD=cloud` frozen build, which deliberately excludes `faster_whisper`/`ctranslate2` from its bundle, ever triggers the pack path.
- `create_engine` must never crash with a raw `ImportError` when the pack is missing — it raises a clear, actionable `RuntimeError` (matches the existing pattern in `gemini_engine.py`'s `load()` — a plain, friendly message, not a stack-trace-only failure).
- **This phase is explicitly risk-gated per the design spec**: loading a native extension module (`ctranslate2`'s compiled `.pyd`) into a PyInstaller-frozen app at runtime is fragile. Tasks 1-6 below are ordinary TDD/code tasks (registry, download/verify logic, build scripts) — safe to build via the normal subagent-driven-development flow. **The actual runtime verification (Task 7) is NOT delegated to a subagent** — per this project's own established practice (the original installer build in an earlier phase was done directly, not subagent-driven, because it requires real ~10-20 minute PyInstaller/Inno builds, a real install, and a live smoke-test decision that gates whether this phase ships as designed or falls back to two installer variants). The plan's controller (you, reading this) performs Task 7 directly after Tasks 1-6 are reviewed and merged.
- If Task 7's smoke test fails (native module won't load correctly from the pack at runtime), the documented fallback is **two installer variants** (`WhisperFlow-Cloud-Setup.exe` ~150MB, `WhisperFlow-Full-Setup.exe` ~1GB, both already buildable from the `WF_BUILD` switch built in Task 4) — this requires zero additional code, only a build-script/release decision, so Tasks 1-6 are not wasted work even if the on-demand pack path is abandoned.

---

### Task 1: `whisperflow/localpack.py` — download, verify, extract, activate

**Files:**
- Create: `whisperflow/localpack.py`
- Test: `tests/test_localpack.py`

**Interfaces:**
- Consumes: `whisperflow.config.data_dir()` (existing).
- Produces: `PACK_DIR` (module-level `Path`, `data_dir() / "local-pack"`), `PACK_VERSION` (module-level `str` constant, bumped by a maintainer alongside `installer/whisperflow.iss`'s `AppVersion` whenever the pack's contents change), `pack_url() -> str` and `pack_sha_url() -> str` (the release asset + its `.sha256` sidecar), `is_installed() -> bool`, `ensure_installed(progress_cb: Callable[[str], None] | None = None) -> None` (no-op if already installed; downloads the zip + sidecar, verifies SHA-256, extracts, writes a completion marker file; raises `RuntimeError` with a clear message on a checksum mismatch or download failure — never leaves a partially-extracted pack marked as installed), `activate() -> None` (prepends `PACK_DIR` to `sys.path` if not already present; raises `RuntimeError` if `is_installed()` is `False` — callers must call `ensure_installed()` first).

**Design note on the sidecar file:** rather than hardcoding a SHA-256 into this module's source (which would need a code change every time the pack is rebuilt), the checksum is published as a second, small release asset (`whisperflow-local-pack-v<VERSION>.zip.sha256`, plain text, one line: the hex digest) alongside the zip — a standard GitHub Releases pattern. `ensure_installed()` downloads both and compares.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_localpack.py
# -*- coding: utf-8 -*-
"""Local-inference pack download/verify/extract/activate — mocked network, no real download."""

import hashlib
import io
import sys
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_localpack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.localpack'`

- [ ] **Step 3: Implement**

```python
# whisperflow/localpack.py
"""On-demand local-inference pack — the ctranslate2/faster_whisper/CUDA-DLL
payload that a `WF_BUILD=cloud` frozen installer deliberately excludes to
stay small. Downloaded once, only if/when the user picks the Local engine,
from a separate GitHub release asset built in the identical Python/
PyInstaller environment as the app itself (so the native .pyd's ABI
matches). A no-op on a dev checkout or a `WF_BUILD=full` build, where
faster_whisper is already importable via the normal environment/bundle —
see registry.py's local-engine dispatch, which only reaches this module
as a fallback.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from whisperflow.config import data_dir

log = logging.getLogger(__name__)

# Bump alongside installer/whisperflow.iss's AppVersion whenever the pack's
# contents (ctranslate2/faster_whisper/CUDA DLL versions) change.
PACK_VERSION = "1.0.1"
_RELEASE_BASE = f"https://github.com/umeshsugara-ai/whisperflow/releases/download/v{PACK_VERSION}"
_PACK_NAME = f"whisperflow-local-pack-v{PACK_VERSION}.zip"

PACK_DIR = data_dir() / "local-pack"
_MARKER = ".pack_complete"


def pack_url() -> str:
    return f"{_RELEASE_BASE}/{_PACK_NAME}"


def pack_sha_url() -> str:
    return f"{_RELEASE_BASE}/{_PACK_NAME}.sha256"


def is_installed() -> bool:
    return (PACK_DIR / _MARKER).exists()


def ensure_installed(progress_cb: Callable[[str], None] | None = None) -> None:
    """Download, verify, and extract the pack if it isn't already installed.
    Idempotent — a no-op when is_installed() is already True."""
    if is_installed():
        return
    report = progress_cb or (lambda msg: None)

    report(f"Downloading local-inference pack ({PACK_VERSION}, ~800MB, one-time)...")
    with urllib.request.urlopen(pack_url(), timeout=120) as resp:
        zip_bytes = resp.read()
    with urllib.request.urlopen(pack_sha_url(), timeout=30) as resp:
        expected_sha = resp.read().decode("ascii").strip().split()[0]

    report("Verifying download...")
    actual_sha = hashlib.sha256(zip_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"local-inference pack checksum mismatch (expected {expected_sha}, got {actual_sha}) "
            "— the download may be corrupted; try again"
        )

    report("Extracting...")
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    import io

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(PACK_DIR)

    (PACK_DIR / _MARKER).write_text(actual_sha, encoding="utf-8")
    report("Local-inference pack installed.")


def activate() -> None:
    """Make the pack's faster_whisper/ctranslate2/CUDA DLLs importable.
    Must be called before the first `import faster_whisper`. Idempotent."""
    if not is_installed():
        raise RuntimeError(
            "local-inference pack is not installed — call ensure_installed() first"
        )
    pack_str = str(PACK_DIR)
    if pack_str not in sys.path:
        sys.path.insert(0, pack_str)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_localpack.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Run the full suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass (162 prior + 7 new).

- [ ] **Step 6: Commit**

```bash
git add whisperflow/localpack.py tests/test_localpack.py
git commit -m "Add localpack — on-demand download/verify/extract/activate for local-inference deps"
```

---

### Task 2: `registry.py` — fallback dispatch for a cloud-base build

**Files:**
- Modify: `whisperflow/stt/registry.py`
- Test: `tests/test_localpack.py` (append) or a new `tests/test_registry_local_fallback.py` — use `tests/test_localpack.py` since it's the same subsystem

**Interfaces:**
- Consumes: `whisperflow.localpack.is_installed`/`ensure_installed`/`activate` (Task 1).
- Produces: `create_engine`'s behavior for `kind == "local"` is unchanged in the two common cases (dev checkout, full build) and gains a guarded, friendly-error fallback for the cloud-base case.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_localpack.py`:

```python
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


def test_registry_local_dispatch_raises_friendly_error_when_pack_missing(monkeypatch):
    from whisperflow.stt import registry

    def fake_import_faster_whisper():
        raise ImportError("no module named faster_whisper")

    monkeypatch.setattr(registry, "_try_import_faster_whisper", fake_import_faster_whisper)
    monkeypatch.setattr(localpack, "is_installed", lambda: False)
    with pytest.raises(RuntimeError, match="one-time download"):
        registry._ensure_local_available()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_localpack.py -k registry -v`
Expected: FAIL with `AttributeError: module 'whisperflow.stt.registry' has no attribute '_ensure_local_available'`

- [ ] **Step 3: Implement**

In `whisperflow/stt/registry.py`, add a small importable indirection function (so tests can monkeypatch it cleanly) and the guard function, then call the guard from `create_engine`:

```python
def _try_import_faster_whisper() -> None:
    """Indirection so tests can simulate 'not importable' without actually
    uninstalling the package. Raises ImportError exactly like a real
    failed import would."""
    import faster_whisper  # noqa: F401


def _ensure_local_available() -> None:
    """No-op on a dev checkout or WF_BUILD=full frozen build, where
    faster_whisper is already importable. Falls back to the on-demand
    local-inference pack only when it isn't (a WF_BUILD=cloud build)."""
    try:
        _try_import_faster_whisper()
        return
    except ImportError:
        pass

    from whisperflow import localpack

    if not localpack.is_installed():
        raise RuntimeError(
            "Local (on-device) mode needs a one-time download (~800MB) that hasn't "
            "happened yet — open Settings and pick Local again, or switch to a free "
            "cloud engine like Groq in the meantime."
        )
    localpack.activate()
```

Then in `create_engine`, add the guard before dispatch:

```python
def create_engine(cfg: ModelConfig) -> SttEngine:
    provider = providers.get(cfg.engine)
    if provider.kind == "local":
        _ensure_local_available()
    module_path, class_name = _ENGINE_BY_KIND[provider.kind].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_localpack.py -v`
Expected: PASS (10 passed — 7 from Task 1 + 3 new)

- [ ] **Step 5: Run the full suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass, including the existing `tests/test_openai_compatible_engine.py`/`tests/test_deepgram_engine.py`/`tests/test_gemini_engine.py` registry-dispatch tests (must still pass unchanged — this task only adds a guard for `kind == "local"`, every other kind's dispatch path is untouched).

- [ ] **Step 6: Commit**

```bash
git add whisperflow/stt/registry.py tests/test_localpack.py
git commit -m "registry.py: fall back to the on-demand local-inference pack when faster_whisper isn't bundled"
```

---

### Task 3: `app.py` — surface the pack download on the Home status strip

**Files:**
- Modify: `app.py` — `build_controller()` (add a pack-download-status check mirroring the existing `_model_needs_download` pattern)
- Test: `tests/test_bootstrap_config.py` (append — this file already tests other `app.py`-level pure helper functions the same way, e.g. `test_any_cloud_api_key_available_detects_non_gemini_key`)

**Interfaces:**
- Consumes: `whisperflow.localpack.is_installed`, `whisperflow.stt.registry._try_import_faster_whisper` (Task 2).
- Produces: `_local_pack_needs_download(model_cfg) -> bool` (mirrors the existing `_model_needs_download(model_cfg) -> bool` at `app.py` — read it first, right above `build_controller`, to match its exact style), plus a WARNING log line in `build_controller` before `create_engine(cfg.model)` is called, so the pending pack download shows on the Home screen's status strip exactly like the existing "Downloading the speech model" warning does.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bootstrap_config.py`:

```python
def test_local_pack_needs_download_false_for_cloud_engine():
    import app
    from whisperflow.config import ModelConfig

    assert app._local_pack_needs_download(ModelConfig(engine="groq")) is False


def test_local_pack_needs_download_false_when_faster_whisper_importable(monkeypatch):
    import app

    # real dev/test environment — faster_whisper IS a project dependency,
    # so this exercises the actual "already available" branch.
    from whisperflow.config import ModelConfig

    assert app._local_pack_needs_download(ModelConfig(engine="local")) is False


def test_local_pack_needs_download_true_when_not_importable_and_pack_missing(monkeypatch):
    import app
    from whisperflow.config import ModelConfig
    from whisperflow.stt import registry

    def fake_import():
        raise ImportError("simulated")

    monkeypatch.setattr(registry, "_try_import_faster_whisper", fake_import)
    monkeypatch.setattr("whisperflow.localpack.is_installed", lambda: False)
    assert app._local_pack_needs_download(ModelConfig(engine="local")) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_bootstrap_config.py -k local_pack -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute '_local_pack_needs_download'`

- [ ] **Step 3: Implement**

In `app.py`, read the existing `_model_needs_download` function in full first (it's right above `build_controller`). Add a sibling function directly after it:

```python
def _local_pack_needs_download(model_cfg) -> bool:
    """True when engine="local" but faster_whisper isn't importable AND the
    on-demand pack hasn't been downloaded yet (WF_BUILD=cloud installs only
    — a no-op check on a dev checkout or a WF_BUILD=full build)."""
    if model_cfg.engine != "local":
        return False
    from whisperflow.stt import registry

    try:
        registry._try_import_faster_whisper()
        return False
    except ImportError:
        pass
    from whisperflow import localpack

    return not localpack.is_installed()
```

Then in `build_controller`, add the warning check right before `engine = create_engine(cfg.model)` (immediately after the existing `_model_needs_download` warning block):

```python
    if _local_pack_needs_download(cfg.model):
        log.warning(
            "Downloading the local-inference pack (~800MB, one-time) — "
            "please keep the app open; dictation starts when it finishes."
        )
    engine = create_engine(cfg.model)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest tests/test_bootstrap_config.py -v`
Expected: PASS (all tests in the file, including the 3 new ones)

- [ ] **Step 5: Run the full suite**

Run: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q`
Expected: all tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_bootstrap_config.py
git commit -m "app.py: surface the local-inference pack download on the Home status strip"
```

---

### Task 4: `installer/whisperflow.spec` — `WF_BUILD=cloud|full` switch

**Files:**
- Modify: `installer/whisperflow.spec`

**Interfaces:** none consumed from prior tasks — this is a PyInstaller spec file, not Python application code. Produces: the spec now reads the `WF_BUILD` environment variable (`cloud` or `full`, default `full` — so an unset env var behaves exactly as today, a required backward-compat property) and excludes `ctranslate2`/`faster_whisper`/`onnxruntime`/the nvidia CUDA wheel binaries when `WF_BUILD=cloud`.

**Design note:** this task has no automated test (PyInstaller spec files aren't pytest-collectible Python) — verification is a real build + `dist/` size check, done as part of Task 7 (the maintainer-run risk-verification gate), not delegated here. Still follow TDD's spirit: read the CURRENT `installer/whisperflow.spec` in full first (it's short, ~80 lines) since this task edits it directly, not via a diff against a stale assumption.

- [ ] **Step 1: Read the current spec file in full**

`Read installer/whisperflow.spec` — note its current structure: `datas`, `binaries` (with the CUDA-DLL-collection block and its documented exclusions), `hiddenimports`, and the `Analysis(...)` call's `excludes=["pytest"]` list.

- [ ] **Step 2: Add the `WF_BUILD` switch**

Add this near the top of the file, right after the existing `REPO = os.path.dirname(SPECPATH)` line:

```python
WF_BUILD = os.environ.get("WF_BUILD", "full")  # "cloud" (slim) or "full" (default, current behavior)
if WF_BUILD not in ("cloud", "full"):
    raise SystemExit(f"WF_BUILD must be 'cloud' or 'full', got {WF_BUILD!r}")
```

- [ ] **Step 3: Skip the CUDA-DLL binaries collection entirely for a cloud build**

Find the existing block that starts `binaries = []` through the `except ImportError: print(...)` block (collecting `ctranslate2`/`onnxruntime` dynamic libs and the nvidia CUDA DLLs). Wrap the WHOLE block in `if WF_BUILD == "full":` — a cloud build needs none of these binaries since `ctranslate2` itself won't be bundled:

```python
binaries = []
if WF_BUILD == "full":
    binaries += collect_dynamic_libs("ctranslate2")
    binaries += collect_dynamic_libs("onnxruntime")

    # CUDA runtime for ctranslate2 GPU inference (cublas + cudnn). These come from
    # the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip wheels in the build venv —
    # without them the frozen app dies with "cublas64_12.dll is not found".
    # On CPU-only machines the DLLs are simply never loaded.
    # Skipped (verified unused by ctranslate2's whisper pipeline on 2026-07-15 —
    # smoke test reaches encode/decode fine without them; saves ~865MB):
    #   cudnn_engines_precompiled (graph-API fusion engines), cudnn_adv (RNN ops),
    #   *.alt.dll (old-driver nvrtc variant), nvblas (BLAS drop-in shim)
    _CUDA_SKIP = ("cudnn_engines_precompiled", "cudnn_adv", ".alt.", "nvblas")
    try:
        import glob

        import nvidia

        for _base in nvidia.__path__:
            for _dll in glob.glob(os.path.join(_base, "*", "bin", "*.dll")):
                if not any(s in os.path.basename(_dll) for s in _CUDA_SKIP):
                    binaries.append((_dll, "."))
    except ImportError:
        print("WARNING: nvidia CUDA wheels not installed — frozen build will be CPU-only")
```

- [ ] **Step 4: Also skip the faster-whisper VAD-model data files for a cloud build**

Find the existing `datas += collect_data_files("faster_whisper")` line. Guard it the same way:

```python
datas = [(os.path.join(REPO, "assets", "app.ico"), "assets")]
if WF_BUILD == "full":
    # faster-whisper bundles the silero VAD onnx model as package data
    datas += collect_data_files("faster_whisper")
```

- [ ] **Step 5: Explicitly exclude the heavy packages from PyInstaller's static analysis for a cloud build**

PyInstaller's import scanner can still try to bundle `ctranslate2`/`faster_whisper`/`onnxruntime`/`nvidia` even when nothing in the app imports them unconditionally (they're all imported lazily, inside function bodies, but PyInstaller's static scan is conservative). Add them to `excludes` for a cloud build. Find the `Analysis(...)` call's `excludes=["pytest"]` argument and change it to:

```python
    excludes=["pytest"] + ([] if WF_BUILD == "full" else [
        "ctranslate2", "faster_whisper", "onnxruntime", "nvidia",
        "torch", "tokenizers",  # transitive faster_whisper/ctranslate2 deps
    ]),
```

- [ ] **Step 6: Rename the output for a cloud build so both variants can coexist in `dist/`**

Find the `name="WhisperFlow"` arguments in both the `EXE(...)` and `COLLECT(...)` calls (two occurrences). Change both to:

```python
    name="WhisperFlow" if WF_BUILD == "full" else "WhisperFlow-Cloud",
```

- [ ] **Step 7: Commit**

```bash
git add installer/whisperflow.spec
git commit -m "installer spec: add WF_BUILD=cloud|full switch — cloud excludes ctranslate2/faster_whisper/CUDA (~150MB vs ~1GB)"
```

(No pytest run for this task — spec files aren't Python modules pytest can import. The full suite should still pass unaffected since nothing in `whisperflow/` changed; running it is optional here but doesn't hurt: `C:/Users/Lenovo/AppData/Local/Microsoft/WindowsApps/python.exe -m pytest -q` should still show all prior tests passing.)

---

### Task 5: `scripts/build_installer.ps1` — build variants + pack packaging

**Files:**
- Modify: `scripts/build_installer.ps1`

**Interfaces:** none — PowerShell build script, not tested by pytest. Reads `WF_BUILD` (Task 4) and adds a `-LocalPack` switch that packages the local-inference dependencies (not the whole app) into the zip `localpack.py` (Task 1) downloads at runtime.

- [ ] **Step 1: Read the current script in full**

`Read scripts/build_installer.ps1` — note the existing structure: build-venv setup, PyInstaller freeze, Inno Setup compile, and the final size-report block.

- [ ] **Step 2: Add a `-Full` / default-cloud parameter and a `-LocalPack` mode**

Replace the script's top (`$ErrorActionPreference = "Stop"` through the `Set-Location $repo` line stays; everything after gets restructured) with:

```powershell
param(
    [switch]$Full,       # build the full (current, ~1GB) installer instead of the slim cloud one
    [switch]$LocalPack   # build+zip the local-inference pack for a GitHub release, instead of the installer
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Output "== 0/2 Build venv =="
$python = "$repo\.venv-build\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $base = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
    if (-not (Test-Path $base)) { $base = "python.exe" }
    & $base -m venv "$repo\.venv-build"
}
# nvidia wheels = CUDA runtime DLLs bundled into the exe (GPU support)
& $python -m pip install --quiet -r requirements.txt pyinstaller nvidia-cublas-cu12 nvidia-cudnn-cu12

if ($LocalPack) {
    Write-Output "== Building local-inference pack zip =="
    $packVer = (Select-String -Path whisperflow\localpack.py -Pattern 'PACK_VERSION = "([^"]+)"').Matches[0].Groups[1].Value
    $packStage = "$repo\build\local-pack-stage"
    Remove-Item -Recurse -Force $packStage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $packStage | Out-Null

    # Copy the SAME packages the "full" frozen build bundles, from the SAME
    # venv — this is what makes the native .pyd ABI-compatible with a
    # WF_BUILD=cloud exe built from this same venv.
    $sitePkgs = "$repo\.venv-build\Lib\site-packages"
    foreach ($pkg in @("faster_whisper", "ctranslate2", "tokenizers", "nvidia")) {
        $src = "$sitePkgs\$pkg"
        if (Test-Path $src) { Copy-Item -Recurse $src "$packStage\$pkg" }
    }

    $zipPath = "$repo\dist\whisperflow-local-pack-v$packVer.zip"
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
    Compress-Archive -Path "$packStage\*" -DestinationPath $zipPath
    $sha = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    Set-Content -Path "$zipPath.sha256" -Value $sha -NoNewline -Encoding ascii

    $mb = [math]::Round((Get-Item $zipPath).Length / 1MB)
    Write-Output ""
    Write-Output "DONE: $zipPath ($mb MB), sha256=$sha"
    Write-Output "Publish both files as release assets: gh release upload vX.Y.Z `"$zipPath`" `"$zipPath.sha256`""
    exit 0
}

$env:WF_BUILD = if ($Full) { "full" } else { "cloud" }
Write-Output "== 1/2 PyInstaller freeze (WF_BUILD=$env:WF_BUILD) =="
& $python -m PyInstaller installer\whisperflow.spec --noconfirm --distpath dist --workpath build
$exeName = if ($Full) { "WhisperFlow" } else { "WhisperFlow-Cloud" }
if (-not (Test-Path "$repo\dist\$exeName\$exeName.exe")) {
    Write-Error "PyInstaller did not produce dist\$exeName\$exeName.exe"
}

Write-Output "== 2/2 Inno Setup compile =="
$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"  # winget --scope user
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Write-Error "Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup --scope user"
}
& $iscc "installer\whisperflow.iss"

$setup = "$repo\installer\Output\WhisperFlow-Setup.exe"
if (Test-Path $setup) {
    $mb = [math]::Round((Get-Item $setup).Length / 1MB)
    Write-Output ""
    Write-Output "DONE: $setup ($mb MB)"
    Write-Output "Distribute via: gh release create vX.Y.Z `"$setup`""
} else {
    Write-Error "Inno Setup did not produce $setup"
}
```

**Note for the implementer:** the `installer\whisperflow.iss` Inno script currently hardcodes `WhisperFlow.exe`/`WhisperFlow` as its source app name — this task does NOT modify `whisperflow.iss` (that's covered by Task 7's manual verification, since whether the Inno script needs a cloud-specific variant depends on what Task 7's smoke test finds; don't speculatively rewrite the `.iss` file here). This task's job is only to make the PowerShell script capable of producing a `WF_BUILD=cloud` frozen `dist/` output and the pack zip — wiring that into Inno Setup for a real installer is Task 7's job.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_installer.ps1
git commit -m "build_installer.ps1: add -Full/-LocalPack modes for the slim cloud-base build"
```

---

### Task 6: README — document the cloud-base install size and maintainer release process

**Files:**
- Modify: `README.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Add a short note to the existing "Which speech engine should I pick?" section**

Find that section (added in Phase B) and add this paragraph right after its provider comparison table:

```markdown
> **Install size:** the default installer only includes cloud engines
> (~150MB). If you pick **Local**, WhisperFlow downloads a one-time
> ~800MB local-inference pack automatically the first time you use it —
> you'll see a status message while it downloads, same as the speech
> model download.
```

- [ ] **Step 2: Add a maintainer note near the existing "Building the installer (maintainer)" line**

Find that existing line (in the "Option A — the .exe installer" section) and replace it with:

```markdown
**Building the installer (maintainer):** install [Inno Setup 6](https://jrsoftware.org/isinfo.php), then run `powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1` for the slim cloud-base build (default, ~150MB) or `-Full` for the all-in-one build (~1GB) → `installer\Output\WhisperFlow-Setup.exe`. Build the local-inference pack separately with `-LocalPack` → `dist\whisperflow-local-pack-vX.Y.Z.zip` (+ `.sha256`) — publish both alongside the installer: `gh release upload vX.Y.Z installer\Output\WhisperFlow-Setup.exe dist\whisperflow-local-pack-vX.Y.Z.zip dist\whisperflow-local-pack-vX.Y.Z.zip.sha256`. Bump `PACK_VERSION` in `whisperflow/localpack.py` to match the release tag whenever the pack's contents change.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: document the slim cloud-base install and the pack-release maintainer process"
```

---

### Task 7 (maintainer-run, NOT subagent-dispatched): risk-gate smoke test + fallback decision

**This task is performed directly by the plan's controller after Tasks 1-6 are reviewed and merged — it is explicitly excluded from subagent-driven-development's per-task dispatch loop**, per this plan's Global Constraints. It mirrors how the original installer build (an earlier phase of this project) was done: real PyInstaller/Inno builds take 10-20 minutes each and the pass/fail decision here determines whether Phase C ships as designed (on-demand pack) or falls back to two static installer variants — both are legitimate, plan-anticipated outcomes, not a plan failure either way.

**Steps (for the controller to execute directly, not via Agent dispatch):**

1. Build the cloud-base variant: `powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1` (no `-Full` flag) → confirm `dist\WhisperFlow-Cloud\WhisperFlow-Cloud.exe` exists and `dist\WhisperFlow-Cloud\` is roughly ~150MB (`(Get-ChildItem dist\WhisperFlow-Cloud -Recurse | Measure-Object Length -Sum).Sum / 1MB`).
2. Build the local-pack zip: `powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -LocalPack` → confirm `dist\whisperflow-local-pack-v1.0.1.zip` and its `.sha256` sidecar exist.
3. **For the smoke test only** (not for a real release), serve the pack zip + sha256 from a location `localpack.py`'s `pack_url()`/`pack_sha_url()` can reach — simplest: temporarily push a real pre-release to GitHub (`gh release create v1.0.1-pack-test dist\whisperflow-local-pack-v1.0.1.zip dist\whisperflow-local-pack-v1.0.1.zip.sha256`), OR monkeypatch `PACK_VERSION`/the release URLs to point at a local `python -m http.server` serving the zip — pick whichever is faster to set up.
4. Run `dist\WhisperFlow-Cloud\WhisperFlow-Cloud.exe` from a directory outside the repo (e.g. copy `dist\WhisperFlow-Cloud\` to a temp folder first, matching how the original installer smoke test in an earlier phase ran the frozen exe from `C:\` rather than the repo, to catch any accidental repo-relative-path assumption).
5. In the running app, go to Settings → Speech engine → pick **Local** → Save → restart.
6. Watch `%LOCALAPPDATA%\WhisperFlow\logs\whisperflow.log` (or the equivalent path if a `--config` override was used) for: the pack-download warning line from Task 3, successful extraction, then the SAME `"ready — hotkey ..."` line that every prior phase's smoke test has confirmed reaching.
7. **Decision:**
   - **If it reaches `ready — hotkey` with local dictation working** (test an actual hold-to-talk dictation): Phase C ships as designed. Publish the real release: `gh release create v1.0.2 installer\Output\WhisperFlow-Setup.exe dist\whisperflow-local-pack-v1.0.1.zip dist\whisperflow-local-pack-v1.0.1.zip.sha256` (bump the installer version in `installer\whisperflow.iss` first, matching the pattern from every prior release in this project).
   - **If the native module fails to load from the pack** (e.g. `ctranslate2`'s `.pyd` raises an ABI/DLL error even with the identical build venv): fall back to two static installer variants. This needs zero new code — build both `-Full` and the default cloud variant, wire `installer\whisperflow.iss` to have (or duplicate into) a `WhisperFlow-Cloud-Setup.exe`/`WhisperFlow-Full-Setup.exe` naming split (a small `.iss` edit, decide the exact approach once you see whatever error the pack path produced), and publish both as separate release assets. Document the outcome in `docs/superpowers/specs/2026-07-15-cloud-stt-providers-design.md`'s Phase C section (append a short "Outcome" note — do not silently rewrite the original risk framing) so it's clear which path shipped and why.
8. Whichever outcome: update `docs/superpowers/specs/2026-07-15-cloud-stt-providers-design.md` with the Phase C outcome, and clean up any test/pre-release GitHub release created in step 3 (`gh release delete v1.0.1-pack-test --yes` if one was created).

---

## Self-Review Notes (completed during plan authoring)

- **Spec coverage:** slim `WF_BUILD=cloud|full` build variant ✓ Task 4, downloadable local-inference pack with SHA-256 verification ✓ Task 1, on-demand fetch triggered by picking Local ✓ Tasks 2-3, `localpack.py`'s `is_installed()`/`ensure_installed()`/`activate()` interface ✓ Task 1 (exact names from the spec), startup guard raising a friendly error instead of crashing ✓ Task 2, `build_installer.ps1` `-Full`/pack-building support ✓ Task 5 (spec said `-LocalPack`, matches), README + release docs ✓ Task 6, the explicit risk-gate + two-variant fallback ✓ Task 7. `tests: localpack.is_installed/ensure_installed/activate against a fake pack dir (monkeypatched download); SHA mismatch aborts cleanly; create_engine(local) without pack raises the friendly guided error, not ImportError` — all three ✓ Tasks 1-2.
- **Placeholder scan:** none found — every step has literal code or, for Tasks 4-5 (PyInstaller spec / PowerShell, not pytest-testable), an exact file edit plus an explicit note on why no automated test applies there (matching how this project has always treated build-tooling changes — verified by a real build, not a unit test). Task 7 is deliberately NOT a bite-sized TDD task (it can't be — it's a real, possibly multi-hour infrastructure verification with a genuine two-outcome decision point) and is explicitly marked as such rather than dressed up as an ordinary task.
- **Type consistency:** `localpack.is_installed() -> bool`, `ensure_installed(progress_cb=None) -> None`, `activate() -> None` used identically across Tasks 1-3, matching the spec's own named interface. `registry._try_import_faster_whisper()`/`_ensure_local_available()` used identically across Tasks 2-3.
