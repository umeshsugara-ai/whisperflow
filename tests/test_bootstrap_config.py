"""bootstrap_config() and its API-key detection: a cloud recommendation must
wire the matching provider's api_key_env, not stay stuck on the Gemini
default (see final-review-fix-report.md, Important #2)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
app = importlib.import_module("app")


def test_any_cloud_api_key_available_detects_non_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert app._any_cloud_api_key_available() is False

    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert app._any_cloud_api_key_available() is True


def test_bootstrap_config_wires_api_key_env_for_recommended_cloud_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")

    from whisperflow import sysinfo
    from whisperflow.sysinfo import Recommendation

    def fake_recommend(specs, has_api_key=False, local_available=True):
        assert has_api_key is True  # proves the GROQ-only env was detected
        return Recommendation(
            engine="groq",
            name="whisper-large-v3-turbo",
            device="cpu",
            compute_type="int8",
            reason="weak machine, groq key available",
            alternatives=[],
        )

    monkeypatch.setattr(sysinfo, "recommend", fake_recommend)
    monkeypatch.setattr(sysinfo, "probe", lambda: sysinfo.SystemSpecs(gpu_name=None, vram_mb=0, ram_gb=4.0, cpu_cores=2))

    cfg_path = tmp_path / "config.toml"
    cfg = app.bootstrap_config(cfg_path)

    assert cfg.model.engine == "groq"
    assert cfg.model.cloud_model == "whisper-large-v3-turbo"
    assert cfg.model.api_key_env == "GROQ_API_KEY"

    from whisperflow.config import load_config

    reloaded = load_config(cfg_path)
    assert reloaded.model.api_key_env == "GROQ_API_KEY"
