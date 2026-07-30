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

# Auto-gain (see Recorder._finalize): quiet-but-real audio is normalized up to
# GAIN_TARGET peak, never amplifying more than GAIN_CAP. When even the cap
# can't reach the target (peak < GAIN_TARGET/GAIN_CAP) the input is so faint
# it's indistinguishable from a collapsed mic level — sending it to STT
# produces whisper hallucinations typed into the user's window (live incident
# 2026-07-28: 40x-amplified noise floor -> 207 chars of garbage + a raw
# "<|hi|>" token injected). In that capped regime the silence bar is raised
# by CAPPED_SILENCE_FACTOR; lowering [audio].silence_rms scales both.
GAIN_TARGET = 0.85
GAIN_CAP = 40.0
CAPPED_SILENCE_FACTOR = 4.0

# a mic-level collapse shows up as a RUN of capped low-level recordings; one
# alone is just a quiet room. Warn once per streak (lands on the Home strip).
LOW_LEVEL_STREAK = 3

# How long a pinned device that failed to open is skipped in favour of the
# system-default route. Some host-API rows fail persistently for a whole app
# session (seen live: the WASAPI row of a mic that a freshly-started process
# opens fine — PortAudio caches its device list at init, so a long-running
# process can hold a stale endpoint), and retrying it on EVERY dictation cost
# a failed open plus the retry sleep before a single word was captured. A
# cooldown rather than a permanent skip keeps the ordinary "device was busy
# for a moment" case self-healing.
DEVICE_RETRY_COOLDOWN_S = 60.0


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
    # True only when the capped-gain guard CHANGED the verdict: input was in
    # the "would have passed the plain silence gate, but the mic level is
    # collapsed" band. Plain silence stays False — the UI uses this to say
    # "mic level near zero" instead of the generic "no speech".
    low_level: bool = False


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


def resolve_device(preference: str, devices=None, hostapis=None) -> tuple[int | None, str]:
    """Return (device_index_or_None_for_default, human_name).

    preference == "default" -> system default input device (index None).
    Anything else -> case-insensitive substring match over input devices,
    preferring the WASAPI row of a mic that appears under several host
    APIs; falls back to default with a warning if nothing matches.

    The WASAPI preference matters: the Settings picker shows WASAPI names
    (list_input_devices), but a first-match scan lands on the same mic's
    DirectSound row (MME's 31-char truncation stops the needle matching
    its row) — and DirectSound is the API that fails with PaErrorCode
    -9999 after sleep/lock, which made a pinned, healthy mic error out.
    The devices/hostapis params exist for unit tests.
    """
    if devices is None:
        devices = sd.query_devices()
    if preference and preference.lower() != "default":
        needle = preference.lower()
        matches = [
            (idx, dev)
            for idx, dev in enumerate(devices)
            if dev["max_input_channels"] > 0 and needle in dev["name"].lower()
        ]
        if matches:
            try:
                apis = hostapis if hostapis is not None else sd.query_hostapis()
                for idx, dev in matches:
                    if "wasapi" in apis[dev["hostapi"]]["name"].lower():
                        return idx, dev["name"]
            except Exception:  # noqa: BLE001 — host-API lookup is best-effort
                pass
            idx, dev = matches[0]
            return idx, dev["name"]
        log.warning("audio device %r not found; using system default", preference)
    default_idx = sd.default.device[0]
    if default_idx is not None and default_idx >= 0:
        return None, sd.query_devices(default_idx)["name"]
    return None, "system default"


def host_api_name(device_idx: int | None) -> str:
    """Host API behind a device index ("Windows WASAPI", "MME", …), or "?" if
    it can't be determined. Diagnostics only: several host APIs expose the
    SAME mic under the same name, so a failure log that prints only the name
    can't tell you which backend actually failed."""
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        if device_idx is None:
            device_idx = sd.default.device[0]
        return hostapis[devices[device_idx]["hostapi"]]["name"]
    except Exception:  # noqa: BLE001 — never let a log line break recording
        return "?"


def uses_wasapi(device_idx: int | None, devices=None, hostapis=None) -> bool:
    """True when the given input device (None = system default) is hosted by
    WASAPI. Streams opened on WASAPI devices need
    `sd.WasapiSettings(auto_convert=True)`: PortAudio's WASAPI backend only
    accepts the device's native mix rate (usually 48 kHz) and rejects our
    16 kHz request with -9997 "Invalid sample rate" — the live cause of a
    pinned, healthy mic failing to open (2026-07-27). MME/DirectSound
    resample on their own and must NOT receive WASAPI settings.
    The devices/hostapis params exist for unit tests."""
    try:
        if devices is None:
            devices = sd.query_devices()
        if hostapis is None:
            hostapis = sd.query_hostapis()
        if device_idx is None:
            device_idx = sd.default.device[0]
        if device_idx is None or device_idx < 0 or device_idx >= len(devices):
            return False
        api = hostapis[devices[device_idx]["hostapi"]]["name"]
        return "wasapi" in api.lower()
    except Exception:  # noqa: BLE001 — a best-effort probe must never block recording
        return False


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

    def __init__(self, cfg: AudioConfig, max_seconds: float | None = None) -> None:
        self.cfg = cfg
        self._stream: sd.InputStream | None = None
        self._blocks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._device_name = ""
        # None = trust cfg.max_seconds. The app passes config.effective_max_seconds(),
        # which holds the cap down when live chunking is off (see that function).
        self._max_samples = int((cfg.max_seconds if max_seconds is None else max_seconds) * SAMPLE_RATE)
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
        self._low_level_streak = 0  # consecutive capped-gain low-level recordings
        self._device_cooldown_until = 0.0  # monotonic; see DEVICE_RETRY_COOLDOWN_S

    def set_config(self, cfg: AudioConfig, max_seconds: float | None = None) -> None:
        """Swap the audio config live (Settings save / tray file reload).
        Derived thresholds are recomputed; the device change takes effect on
        the next recording start (start() re-resolves it every time).
        `max_seconds` overrides cfg.max_seconds exactly as in __init__ — it must
        be passed on every call, since toggling live chunking in Settings changes
        the cap in both directions."""
        self.cfg = cfg
        self._max_samples = int((cfg.max_seconds if max_seconds is None else max_seconds) * SAMPLE_RATE)
        self._voice_peak = max(cfg.silence_rms * 8.0, 0.004)
        # picking a mic in Settings is an explicit "try this one" — honour it
        # immediately instead of making the user wait out an old cooldown
        self._device_cooldown_until = 0.0

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
        preference = self.cfg.device
        if device_idx is not None and time.monotonic() < self._device_cooldown_until:
            # this row failed recently — go straight to the route that worked
            # instead of paying a doomed open plus the retry sleep every time
            device_idx = None
            self._device_name = resolve_device("default")[1]
            # we CHOSE this route, so judge the warning as a default-mic run:
            # otherwise the pinned name won't match the default's (truncated)
            # name and the Home strip cries "mic not found" about a mic that
            # is present and working
            preference = "default"
            log.debug("pinned device still in cooldown; using system default")
        self.device_warning = device_warning(preference, self._device_name)
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

        def _open(idx: int | None) -> None:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=idx,
                callback=self._callback,
                latency="low",  # minimize device spin-up so the first word isn't clipped
                # WASAPI rejects non-native rates (-9997) without auto_convert;
                # see uses_wasapi(). None for MME/DirectSound-hosted devices.
                extra_settings=sd.WasapiSettings(auto_convert=True) if uses_wasapi(idx) else None,
            )
            self._stream.start()

        try:
            _open(device_idx)
        except sd.PortAudioError as exc:
            # transient host-API failures (DirectSound -9999 after sleep/
            # lock, device briefly claimed by another app): one retry via
            # the system default route instead of dying on the first try
            self._stream = None
            self._device_cooldown_until = time.monotonic() + DEVICE_RETRY_COOLDOWN_S
            log.warning(
                "mic open failed on %r [index %s, %s] (%s) — using the system "
                "default for the next %.0fs",
                self._device_name, device_idx, host_api_name(device_idx), exc,
                DEVICE_RETRY_COOLDOWN_S,
            )
            time.sleep(0.3)
            try:
                _open(None)
                self._device_name = resolve_device("default")[1]
                # the warning was computed for the pinned device we just gave
                # up on — re-judge it for the mic actually in use
                self.device_warning = device_warning("default", self._device_name)
                log.info("fell back to system default [%s]", host_api_name(None))
            except sd.PortAudioError as exc2:
                self._stream = None
                raise RuntimeError(
                    f"microphone unavailable ({self._device_name}) — check it isn't "
                    f"in use by another app, then try again"
                ) from exc2
        log.info("recording started on %r", self._device_name)
        return self._device_name

    def _finalize(self, blocks: list[np.ndarray]) -> Recording:
        """Turn raw blocks into a Recording (rms/auto-gain/flag logic)."""
        samples = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
        duration_s = len(samples) / SAMPLE_RATE
        # silence decision uses the ORIGINAL level, before any gain
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        peak = float(np.abs(samples).max()) if len(samples) else 0.0

        # Capped-gain guard: when even GAIN_CAP can't lift the peak to
        # GAIN_TARGET, "signal" this faint is a collapsed mic level, and
        # amplified noise floor sent to whisper comes back as hallucinated
        # text typed into the user's window. In that regime the silence bar
        # is CAPPED_SILENCE_FACTOR higher. low_level marks exactly the band
        # where this guard flipped the verdict — plain silence stays False.
        capped = 0.0 < peak < GAIN_TARGET / GAIN_CAP
        low_level = capped and (
            self.cfg.silence_rms <= rms < self.cfg.silence_rms * CAPPED_SILENCE_FACTOR
        )
        silent = rms < self.cfg.silence_rms or low_level
        self._track_low_level(low_level, silent)

        # Auto-gain: laptop mics at low Windows input volume produce faint
        # audio (peaks ~0.005) that VAD discards as silence. If there IS
        # signal but it's quiet, normalize to a healthy peak before STT.
        # Silent recordings are never amplified — they never reach STT.
        if not silent and 0.0 < peak < 0.30:
            gain = min(GAIN_TARGET / peak, GAIN_CAP)
            samples = samples * gain
            log.info(
                "auto-gain applied: peak %.4f -> %.2f (gain %.1fx)",
                peak, min(peak * gain, GAIN_TARGET), gain,
            )

        return Recording(
            samples=samples,
            device_name=self._device_name,
            duration_s=duration_s,
            rms=rms,
            too_short=duration_s < self.cfg.min_seconds,
            silent=silent,
            low_level=low_level,
        )

    def _track_low_level(self, low_level: bool, silent: bool) -> None:
        """Mic-health streak: a collapsed input level shows up as a RUN of
        low_level recordings (a mix of in-band and below-band chunks, so
        plain silence is NEUTRAL — only real speech resets). Warn exactly
        once per streak; a healthy recording re-arms the warning. Called
        from _finalize, which runs on both the worker thread (live chunks)
        and the hotkey thread (stop) — hence the lock."""
        with self._lock:
            if low_level:
                self._low_level_streak += 1
                if self._low_level_streak == LOW_LEVEL_STREAK:
                    log.warning(
                        "Mic level near zero for %d recordings — Windows input volume "
                        "or another app (e.g. Zoom auto-adjust) may have lowered it. "
                        "Open Settings → Test mic.",
                        LOW_LEVEL_STREAK,
                    )
            elif not silent:
                self._low_level_streak = 0

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

    def close(self) -> None:
        """Force-release the device, swallowing errors — the idempotent
        cleanup a `finally` can always call. stop() is the normal path; this
        exists so a failed stop() can't leave the mic held open, which would
        make every later recording fail to claim the device until restart."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001 — releasing is best-effort by definition
            log.debug("stream close failed during forced release", exc_info=True)

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
