"""Audio capture test.

--check     : automated — record for --duration seconds (silence is fine),
              assert buffer shape/dtype/rate, print device + RMS, exit 0.
(default)   : manual — record --duration seconds while you SPEAK, save wav,
              print RMS so you can confirm the mic actually heard you.

    python scripts/test_audio.py --duration 1 --check
    python scripts/test_audio.py --duration 5
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from whisperflow.audio import SAMPLE_RATE, Recorder  # noqa: E402
from whisperflow.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--check", action="store_true", help="automated assertions, no wav output")
    args = ap.parse_args()

    cfg = load_config()
    rec = Recorder(cfg.audio)
    device = rec.start()
    print(f"recording on: {device}")
    if not args.check:
        print(f"SPEAK NOW for {args.duration:.0f}s...")
    # wait until the requested amount of audio has actually been captured
    # (the device needs a moment to spin up before frames start arriving)
    deadline = time.time() + args.duration + 5.0
    while rec.captured_seconds < args.duration and time.time() < deadline:
        time.sleep(0.05)
    result = rec.stop()

    print(f"device      : {result.device_name}")
    print(f"samples     : {result.samples.shape} dtype={result.samples.dtype}")
    print(f"duration    : {result.duration_s:.2f}s")
    print(f"rms         : {result.rms:.5f}")
    print(f"too_short   : {result.too_short}   silent: {result.silent}")

    if args.check:
        expected = int(args.duration * SAMPLE_RATE)
        assert result.samples.dtype == np.float32, "dtype must be float32"
        assert result.samples.ndim == 1, "must be mono 1-D"
        # allow 25% slack for stream start/stop latency
        assert expected * 0.75 <= len(result.samples) <= expected * 1.25, (
            f"sample count {len(result.samples)} not within 25% of expected {expected}"
        )
        print("CHECK PASS")
        return 0

    out = Path(__file__).parent / "test_recording.wav"
    pcm = (np.clip(result.samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    print(f"saved: {out} — play it back to confirm quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
