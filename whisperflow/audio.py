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
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from whisperflow.config import AudioConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class Recording:
    samples: np.ndarray  # float32 mono @16k
    device_name: str
    duration_s: float
    rms: float
    too_short: bool
    silent: bool


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

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.debug("audio status: %s", status)
        # live level for UI feedback (no lock: single float write is atomic)
        self.last_peak = float(np.abs(indata[:, 0]).max())
        with self._lock:
            if self._sample_count >= self._max_samples:
                return  # cap reached: drop further blocks
            self._blocks.append(indata[:, 0].copy())
            self._sample_count += frames
            if self._sample_count >= self._max_samples and self.on_max_duration:
                # notify once, outside the lock would be nicer but callback is short
                cb, self.on_max_duration = self.on_max_duration, None
                threading.Thread(target=cb, daemon=True).start()

    def start(self) -> str:
        """Begin capture. Returns the active device name."""
        if self._stream is not None:
            raise RuntimeError("already recording")
        device_idx, self._device_name = resolve_device(self.cfg.device)
        with self._lock:
            self._blocks = []
            self._sample_count = 0
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

    def stop(self) -> Recording:
        """End capture and return the buffered audio."""
        if self._stream is None:
            raise RuntimeError("not recording")
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

        with self._lock:
            blocks, self._blocks = self._blocks, []

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

        rec = Recording(
            samples=samples,
            device_name=self._device_name,
            duration_s=duration_s,
            rms=rms,
            too_short=duration_s < self.cfg.min_seconds,
            silent=rms < self.cfg.silence_rms,
        )
        log.info(
            "recording stopped: %.2fs, rms=%.5f, too_short=%s, silent=%s",
            duration_s,
            rms,
            rec.too_short,
            rec.silent,
        )
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
