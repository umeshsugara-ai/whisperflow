"""VS Code-style silent auto-update for installed (frozen) builds.

Background loop: ~60s after launch and every 24h after, ask GitHub for the
latest release; if it's newer than this build, silently pre-download the
installer into data_dir()/updates/. Only a COMPLETE, size-verified download
flips the "update ready" state the UI surfaces (tray item, Home strip,
pill toast). Installing is always a manual click: the app launches the
downloaded Setup.exe with /VERYSILENT (per-user install, no UAC) and quits
cleanly; the installer's PrepareToInstall kills any stragglers, upgrades in
place, and its [Run] entry relaunches the app. User settings live in
%LOCALAPPDATA%\\WhisperFlow, which the installer never touches.

Every network/disk failure here is a silent no-op retried next cycle —
log.debug only, because anything at WARNING or above lands on the Home
warning strip and an offline laptop must not nag.

Future work (deliberately out of scope): SHA-256/Authenticode verification
of the downloaded exe (today: HTTPS + exact size check), delta updates.

Dev checkouts never update themselves — Updater.start() no-ops unless
config.is_frozen(); `git pull` is the dev update channel.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from whisperflow import __version__
from whisperflow.config import data_dir, is_frozen

log = logging.getLogger(__name__)

GITHUB_LATEST_URL = "https://api.github.com/repos/umeshsugara-ai/whisperflow/releases/latest"
ASSET_NAME = "WhisperFlow-Setup.exe"
FIRST_CHECK_DELAY_S = 60.0
CHECK_INTERVAL_S = 24 * 3600.0
LAST_RUN_VERSION_FILE = "last_run_version.txt"


# ---- pure decision core (no I/O — unit-tested) ------------------------------


def parse_version(tag: str) -> tuple[int, ...] | None:
    """"v1.0.3" / "1.0.3" -> (1, 0, 3); anything non-numeric ("latest",
    "v1.0.3-beta", "") -> None, which every caller treats as "not an
    update" — a weird tag must never trigger a download loop."""
    tag = tag.strip()
    if tag[:1] in ("v", "V"):
        tag = tag[1:]
    if not tag:
        return None
    parts = tag.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def is_newer(remote_tag: str, current: str) -> bool:
    """True only when remote parses AND is strictly newer. Tuples are padded
    with zeros so "1.0" vs "1.0.1" compares by value, not length."""
    r, c = parse_version(remote_tag), parse_version(current)
    if r is None or c is None:
        return False
    width = max(len(r), len(c))
    return r + (0,) * (width - len(r)) > c + (0,) * (width - len(c))


def pick_setup_asset(assets: list[dict]) -> tuple[str, int] | None:
    """(download_url, size) of the installer asset, or None. The asset name
    is version-less, so the URL must come from the release JSON — never
    constructed."""
    for a in assets:
        if str(a.get("name", "")).lower() == ASSET_NAME.lower():
            url = a.get("browser_download_url") or ""
            size = a.get("size") or 0
            if url and size > 0:
                return url, size
            return None
    return None


@dataclass(frozen=True)
class UpdateCandidate:
    version: str  # normalized, no leading "v"
    url: str
    size: int


def evaluate_release(release: dict, current_version: str) -> UpdateCandidate | None:
    """The single decision function: release JSON in, candidate out (or
    None for "nothing to do" — older, malformed, or no installer asset)."""
    tag = str(release.get("tag_name", ""))
    if not is_newer(tag, current_version):
        return None
    asset = pick_setup_asset(release.get("assets") or [])
    if asset is None:
        return None
    url, size = asset
    return UpdateCandidate(version=tag.lstrip("vV"), url=url, size=size)


def download_path(updates_dir: Path, version: str) -> Path:
    """Versioned filename so a stale 1.0.3 download can't be mistaken for a
    fresh 1.0.4 one."""
    return updates_dir / f"WhisperFlow-Setup-{version}.exe"


def is_download_valid(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


# ---- thin IO shell (every failure -> debug log + None/False) ----------------


def _request(url: str) -> urllib.request.Request:
    # GitHub's API rejects requests without a User-Agent.
    return urllib.request.Request(url, headers={
        "User-Agent": f"WhisperFlow/{__version__}",
        "Accept": "application/vnd.github+json",
    })


def fetch_latest_release(timeout: float = 15.0) -> dict | None:
    try:
        with urllib.request.urlopen(_request(GITHUB_LATEST_URL), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — offline/rate-limited is normal life
        log.debug("update check skipped: %s", exc)
        return None


def download_asset(url: str, dest: Path, expected_size: int, timeout: float = 30.0) -> bool:
    """Stream to dest's .part sibling, verify the exact size, then atomically
    rename (same idiom as save_config's tmp+os.replace)."""
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as resp, open(part, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)
        if part.stat().st_size != expected_size:
            raise OSError(f"size mismatch: got {part.stat().st_size}, expected {expected_size}")
        os.replace(part, dest)
        return True
    except Exception as exc:  # noqa: BLE001 — disk full / dropped connection / AV lock
        log.debug("update download failed (will retry next cycle): %s", exc)
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def cleanup_updates_dir(updates_dir: Path, keep: Path | None = None) -> None:
    """Sweep half-finished .part files and superseded installers. Per-file
    try/except: a file locked by a running installer must not abort the
    sweep."""
    if not updates_dir.is_dir():
        return
    for p in updates_dir.iterdir():
        if p == keep:
            continue
        if p.suffix == ".part" or (p.name.startswith("WhisperFlow-Setup-") and p.suffix == ".exe"):
            try:
                p.unlink()
            except OSError:
                pass


def launch_installer(path: Path, expected_size: int) -> bool:
    """Re-validate AT CLICK TIME (antivirus may have eaten the file since it
    downloaded), then spawn the silent upgrade detached — the installer will
    kill this very process, so it must not be our child in any job sense."""
    if not is_download_valid(path, expected_size):
        return False
    try:
        subprocess.Popen(  # noqa: S603 — our own downloaded, size-verified installer
            [str(path), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        return True
    except OSError as exc:
        log.debug("could not launch the update installer: %s", exc)
        return False


def read_last_run_version(data_path: Path) -> str | None:
    try:
        return (data_path / LAST_RUN_VERSION_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_last_run_version(data_path: Path, version: str) -> None:
    try:
        (data_path / LAST_RUN_VERSION_FILE).write_text(version, encoding="utf-8")
    except OSError:
        pass  # a missed marker just means no "updated ✓" toast next time


# ---- orchestrator -----------------------------------------------------------


class Updater:
    """Owns the wf-updater background thread and the "update ready" state.

    `ready_version` is read cross-thread by the tray's callable menu labels
    (the house idiom — plain attribute reads, no locking needed for a str
    swap). `on_ready(version)` fires from the updater thread exactly when a
    complete, size-verified installer is on disk.
    """

    def __init__(self, cfg, on_ready: Callable[[str], None]) -> None:
        self.cfg = cfg
        self.on_ready = on_ready
        self.ready_version: str | None = None
        self._ready_path: Path | None = None
        self._ready_size: int = 0
        self._stop = threading.Event()

    def start(self) -> None:
        if not is_frozen():
            return  # dev checkout: `git pull` is the update channel
        threading.Thread(target=self._loop, daemon=True, name="wf-updater").start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        if self._stop.wait(FIRST_CHECK_DELAY_S):
            return
        while not self._stop.is_set():
            try:
                self.check_once()
            except Exception as exc:  # noqa: BLE001 — the loop must survive anything
                log.debug("update check crashed (loop continues): %s", exc)
            if self._stop.wait(CHECK_INTERVAL_S):
                return

    def check_once(self) -> None:
        if not self.cfg.updates.auto_check:  # re-read each cycle: config reload applies live
            return
        release = fetch_latest_release()
        if release is None:
            return
        updates_dir = data_dir() / "updates"
        cand = evaluate_release(release, __version__)
        if cand is None:
            # up to date (or the release was pulled / user updated by hand):
            # anything still lying in updates/ is garbage now
            cleanup_updates_dir(updates_dir)
            return
        updates_dir.mkdir(parents=True, exist_ok=True)
        dest = download_path(updates_dir, cand.version)
        if not is_download_valid(dest, cand.size):
            # half-finished downloads always restart from scratch — no resume
            cleanup_updates_dir(updates_dir)
            if not download_asset(cand.url, dest, cand.size):
                return  # retry next cycle
        self._ready_path = dest
        self._ready_size = cand.size
        self.ready_version = cand.version
        self.on_ready(cand.version)

    def install_ready_update(self) -> bool:
        """Launch the downloaded installer. True → caller must quit the app
        cleanly (exit 0, so the crash watchdog stands down)."""
        path, size = self._ready_path, self._ready_size
        if path is None or not launch_installer(path, size):
            # e.g. antivirus deleted the file since it downloaded — clear the
            # state (menu item disappears) and let the next cycle re-download.
            # This one warning is allowed on the Home strip: the user just
            # clicked and deserves feedback.
            self.ready_version = None
            self._ready_path = None
            log.warning("The downloaded update is missing or damaged — it will be re-downloaded automatically.")
            return False
        return True
