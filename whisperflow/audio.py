"""Microphone capture for dictation.

sounddevice InputStream at 16 kHz mono float32 — fed directly to
faster-whisper as a numpy array (no temp WAV, no ffmpeg).

Anti-Wispr details:
- the input device is re-resolved at EVERY recording start (a Bluetooth
  headset connecting mid-session is picked up immediately, and the device
  name is exposed so the UI can show which mic is live);
- hard max-duration cap; too-short and silent recordings are flagged so the
  pipeline can skip transcription instead of hallucinating.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from whisperflow.config import AudioConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def level_fraction(peak: float) -> float:
    """Map a raw input peak to a 0..1 UI level — shared by the overlay
    waveform and the Settings mic-test bar so both tell the same story
    (the 20x gain keeps faint laptop mics visible)."""
    return min(1.0, peak * 20.0)

# Virtual mics that exist even when their companion app isn't streaming — they
# deliver pure silence, and Windows loves silently making them the default
# (this bit us on 2026-07-07 with "Microphone (Camo)"). We can't refuse to use
# them, but we can warn loudly so "why is nothing transcribing" is answerable.
_VIRTUAL_MIC_HINTS = ("camo", "steam streaming", "droidcam", "iriun", "virtual audio")


@dataclass
class Recording:
    samples: np.ndarray  # float32 mono @16k
    device_name: str
    duration_s: float
    rms: float
    too_short: bool
    silent: bool


def list_input_devices(devices=None, hostapis=None) -> list[str]:
    """Unique input-device names for the Settings mic picker.

    Windows exposes each physical mic once per host API — MME truncates
    names to 31 chars while WASAPI carries the full name — so prefer the
    WASAPI rows and dedupe case-insensitively: one row per real mic.
    Falls back to all-API dedup on platforms without WASAPI. The
    devices/hostapis params exist for unit tests; production callers pass
    nothing and get a live sounddevice query.
    """
    try:
        if devices is None:
            devices = sd.query_devices()
        if hostapis is None:
            hostapis = sd.query_hostapis()
    except Exception:  # noqa: BLE001 — a broken audio stack must not kill Settings
        log.warning("could not enumerate input devices", exc_info=True)
        return []

    def rows(wasapi_only: bool) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for dev in devices:
            if dev["max_input_channels"] <= 0:
                continue
            api = hostapis[dev["hostapi"]]["name"].lower()
            if wasapi_only and "wasapi" not in api:
                continue
            key = dev["name"].lower()
            if key not in seen:
                seen.add(key)
                names.append(dev["name"])
        return names

    return rows(True) or rows(False)


def resolve_device(preference: str) -> tuple[int | None, str]:
    """Return (device_index_or_None_for_default, human_name).

    preference == "default" -> system default input device (index None).
    Anything else -> case-insensitive substring match over input devices;
    falls back to default with a warning if nothing matches.
    """
    if preference and preference.lower() != "default":
        needle = preference.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
                return idx, dev["name"]
        log.warning("audio device %r not found; using system default", preference)
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        return None, sd.query_devices(default_idx)["name"]
    return None, "system default"


def device_warning(preference: str, resolved_name: str) -> str:
    """Human-readable warning when the resolved mic looks wrong, else "".

    Pure (no sounddevice calls) so it's unit-testable anywhere:
    - the pinned device wasn't found and we silently fell back to the default;
    - the mic in use is a known always-silent virtual device (Camo & friends).
    """
    lowered = resolved_name.lower()
    if (
        preference
        and preference.lower() != "default"
        and preference.lower() not in lowered
    ):
        return f'mic "{preference}" not found — using "{resolved_name}" instead'
    for hint in _VIRTUAL_MIC_HINTS:
        if hint in lowered:
            return (
                f'"{resolved_name}" is a virtual mic — it records silence unless '
                "its companion app is streaming. Pick your real mic in "
                "Settings → Microphone"
            )
    return ""


class Recorder:
    """Start/stop microphone capture; returns a Recording on stop."""

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._stream: sd.InputStream | None = None
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._device_name = ""
        self._max_samples = int(cfg.max_seconds * SAMPLE_RATE)
        self._sample_count = 0
        self.on_max_duration: callable | None = None  # set by controller
        self.last_peak: float = 0.0  # live input level for UI feedback
        self.device_warning: str = ""  # refreshed on every start(); read by the mic test
        self._warned_devices: set[str] = set()  # WARN once per device, then debug
        self._max_notified = False  # on_max_duration fired for THIS recording
        # live-chunking state (see take_pending()): peak-based voice activity —
        # the peak is already computed for the UI, so this adds no per-block work
        self._voice_peak = max(cfg.silence_rms * 8.0, 0.004)
        self._last_voice: float = 0.0  # monotonic time of last voiced block
        self._voiced_since_drain = False

    def set_config(self, cfg: AudioConfig) -> None:
        """Swap the audio config live (Settings save / tray file reload).
        Derived thresholds are recomputed; the device change takes effect on
        the next recording start (start() re-resolves it every time)."""
        self.cfg = cfg
        self._max_samples = int(cfg.max_seconds * SAMPLE_RATE)
        self._voice_peak = max(cfg.silence_rms * 8.0, 0.004)

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def captured_seconds(self) -> float:
        with self._lock:
            return self._sample_count / SAMPLE_RATE

    @property
    def pending_seconds(self) -> float:
        """Audio buffered since start (or the last take_pending()) — the
        chunking watermark, derived from the block list itself so it can
        never drift out of sync with the actual buffer."""
        with self._lock:
            return sum(len(b) for b in self._blocks) / SAMPLE_RATE

    @property
    def voiced_since_drain(self) -> bool:
        return self._voiced_since_drain

    def seconds_since_voice(self) -> float:
        """Seconds since the last block that looked like speech (inf if none)."""
        if self._last_voice <= 0.0:
            return float("inf")
        return time.monotonic() - self._last_voice

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("audio status: %s", status)
        # live level for UI feedback (no lock: single float write is atomic)
        peak = float(np.abs(indata[:, 0]).max())
        self.last_peak = peak
        with self._lock:
            # voice flag updates inside the lock so take_pending() can't clear
            # a flag belonging to a block that lands in the NEXT buffer
            if peak > self._voice_peak:
                self._last_voice = time.monotonic()
                self._voiced_since_drain = True
            if self._sample_count >= self._max_samples:
                return  # cap reached: drop further blocks
            self._blocks.append(indata[:, 0].copy())
            self._sample_count += frames
            if self._sample_count >= self._max_samples and not self._max_notified and self.on_max_duration:
                self._max_notified = True  # once per recording; hook stays wired for the next one
                threading.Thread(target=self.on_max_duration, daemon=True).start()

    def start(self) -> str:
        """Begin capture. Returns the active device name."""
        if self._stream is not None:
            raise RuntimeError("already recording")
        device_idx, self._device_name = resolve_device(self.cfg.device)
        self.device_warning = device_warning(self.cfg.device, self._device_name)
        if self.device_warning:
            if self._device_name not in self._warned_devices:
                self._warned_devices.add(self._device_name)
                log.warning("%s", self.device_warning)
            else:
                log.debug("%s", self.device_warning)
        with self._lock:
            self._blocks = []
            self._sample_count = 0
            self._last_voice = 0.0
            self._voiced_since_drain = False
        self._max_notified = False
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_idx,
            callback=self._callback,
            latency="low",  # minimize device spin-up so the first word isn't clipped
        )
        self._stream.start()
        log.info("recording started on %r", self._device_name)
        return self._device_name

    def _finalize(self, blocks: list[np.ndarray]) -> Recording:
        """Turn raw blocks into a Recording (rms/auto-gain/flag logic)."""
        samples = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        duration_s = len(samples) / SAMPLE_RATE
        # silence decision uses the ORIGINAL level, before any gain
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0

        # Auto-gain: laptop mics at low Windows input volume produce faint
        # audio (peaks ~0.005) that VAD discards as silence. If there IS
        # signal but it's quiet, normalize to a healthy peak before STT.
        peak = float(np.abs(samples).max()) if len(samples) else 0.0
        if 0.0 < peak < 0.30 and rms > 0.0:
            gain = min(0.85 / peak, 40.0)  # cap gain so pure noise isn't blown up
            samples = samples * gain
            log.info("auto-gain applied: peak %.4f -> %.2f (gain %.1fx)", peak, min(peak * gain, 0.85), gain)

        return Recording(
            samples=samples,
            device_name=self._device_name,
            duration_s=duration_s,
            rms=rms,
            too_short=duration_s < self.cfg.min_seconds,
            silent=rms < self.cfg.silence_rms,
        )

    def stop(self) -> Recording:
        """End capture and return the buffered audio."""
        if self._stream is None:
            raise RuntimeError("not recording")
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

        with self._lock:
            blocks, self._blocks = self._blocks, []

        rec = self._finalize(blocks)
        log.info(
            "recording stopped: %.2fs, rms=%.5f, too_short=%s, silent=%s",
            rec.duration_s,
            rec.rms,
            rec.too_short,
            rec.silent,
        )
        return rec

    def take_pending(self) -> list[np.ndarray] | None:
        """Swap out the audio buffered so far WITHOUT stopping the stream —
        the live-chunking handoff. Deliberately CHEAP (a list swap under the
        lock): the controller calls this while holding its own state lock,
        so the heavy concatenate/rms/gain work is deferred to
        build_recording(), which the worker thread runs lock-free.
        None when idle or nothing is buffered."""
        if self._stream is None:
            return None
        with self._lock:
            blocks, self._blocks = self._blocks, []
            self._voiced_since_drain = False
        return blocks or None

    def build_recording(self, blocks: list[np.ndarray]) -> Recording:
        """Finalize blocks from take_pending() into a Recording (worker thread)."""
        rec = self._finalize(blocks)
        log.info("chunk drained: %.2fs, rms=%.5f, silent=%s", rec.duration_s, rec.rms, rec.silent)
        return rec

    def cancel(self) -> None:
        """End capture and discard the buffer."""
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()
        with self._lock:
            self._blocks = []
        log.info("recording cancelled")
