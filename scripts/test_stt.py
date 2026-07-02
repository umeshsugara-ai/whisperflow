"""STT engine test.

--smoke : automated — load configured model, transcribe 1s synthetic silence
          (must not raise), print model/device/timing, exit 0.
--live  : manual — record N seconds from the mic, transcribe, print raw text.
          Try English, Hindi, and Hinglish. Use --model small to prove the
          registry swap works.

    python scripts/test_stt.py --smoke
    python scripts/test_stt.py --live --duration 6 [--model small] [--language hi]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from whisperflow.config import load_config  # noqa: E402
from whisperflow.stt.registry import FASTER_WHISPER_MODELS, create_engine, resolve_model_id  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--live", action="store_true")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--model", default="", help="override [model].name (e.g. small)")
    ap.add_argument("--language", default="", help="override [model].language (e.g. hi)")
    args = ap.parse_args()

    cfg = load_config()
    if args.model:
        cfg.model.name = args.model
    if args.language:
        cfg.model.language = args.language

    # registry sanity (part of the automated verify)
    for name in ("large-v3-turbo", "large-v3", "medium", "small"):
        assert name in FASTER_WHISPER_MODELS, f"registry missing {name}"
    assert resolve_model_id("org/custom-model") == "org/custom-model", "passthrough broken"
    print("registry OK:", ", ".join(FASTER_WHISPER_MODELS))

    engine = create_engine(cfg.model)
    print(f"loading model: {cfg.model.name} -> {engine.model_id} on {cfg.model.device}")
    t0 = time.perf_counter()
    engine.load()
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    if args.smoke:
        silence = np.zeros(16_000, dtype=np.float32)  # 1s @16k
        result = engine.transcribe(silence)
        print(f"smoke transcribe: text={result.text!r} lang={result.language} " f"took={result.transcribe_seconds:.2f}s")
        print("SMOKE PASS")
        return 0

    # --live
    from whisperflow.audio import Recorder

    rec = Recorder(cfg.audio)
    device = rec.start()
    print(f"SPEAK NOW ({args.duration:.0f}s) on {device}...")
    deadline = time.time() + args.duration + 5.0
    while rec.captured_seconds < args.duration and time.time() < deadline:
        time.sleep(0.05)
    recording = rec.stop()
    print(f"recorded {recording.duration_s:.1f}s (rms={recording.rms:.4f})")
    if recording.silent:
        print("WARNING: recording is silent — check your mic")

    result = engine.transcribe(recording.samples, language=cfg.model.language)
    print(f"\nlanguage : {result.language} (p={result.language_probability:.2f})")
    print(f"time     : {result.transcribe_seconds:.2f}s for {result.duration_s:.1f}s audio")
    print(f"text     : {result.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
