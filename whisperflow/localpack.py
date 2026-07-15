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
import io
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Callable

from whisperflow.config import data_dir

# Bump to match the release tag (and installer/whisperflow.iss's AppVersion)
# whenever the pack's contents (ctranslate2/faster_whisper/CUDA DLL versions) change.
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
    try:
        with urllib.request.urlopen(pack_url(), timeout=120) as resp:
            zip_bytes = resp.read()
        with urllib.request.urlopen(pack_sha_url(), timeout=30) as resp:
            expected_sha = resp.read().decode("ascii").strip().split()[0]
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"failed to download local-inference pack: {exc}") from exc

    report("Verifying download...")
    actual_sha = hashlib.sha256(zip_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"local-inference pack checksum mismatch (expected {expected_sha}, got {actual_sha}) "
            "— the download may be corrupted; try again"
        )

    report("Extracting...")
    PACK_DIR.mkdir(parents=True, exist_ok=True)

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
