"""Manual smoke test: transcribe 1s of silence through every configured
cloud provider with a real API key. NOT part of the pytest suite (hits
real network APIs, costs real quota/money for paid providers).

Usage:
    set GROQ_API_KEY=...      (or GEMINI_API_KEY / OPENAI_API_KEY / DEEPGRAM_API_KEY)
    python scripts/test_cloud_stt.py groq
    python scripts/test_cloud_stt.py          # tries every provider with a key present
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from whisperflow.config import ModelConfig
from whisperflow.stt.providers import cloud_providers
from whisperflow.stt.registry import create_engine


def try_provider(provider_id: str) -> None:
    from whisperflow.stt import providers

    provider = providers.get(provider_id)
    key = os.environ.get(provider.api_key_env, "")
    if not key:
        print(f"SKIP {provider_id}: ${provider.api_key_env} not set")
        return
    cfg = ModelConfig(engine=provider_id)
    engine = create_engine(cfg)
    try:
        engine.load()
        # 1s of near-silence (small noise so it's not pure zeros, some APIs
        # reject dead-silent audio as invalid)
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, language="en")
        print(f"OK   {provider_id}: transcribed {result.duration_s:.1f}s in "
              f"{result.transcribe_seconds:.2f}s -> {result.text!r}")
    except Exception as exc:  # noqa: BLE001 — smoke test, report and continue
        print(f"FAIL {provider_id}: {exc}")


def main() -> None:
    targets = sys.argv[1:] or [p.id for p in cloud_providers()]
    for provider_id in targets:
        try_provider(provider_id)


if __name__ == "__main__":
    main()
