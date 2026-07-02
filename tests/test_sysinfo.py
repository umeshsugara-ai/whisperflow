# -*- coding: utf-8 -*-
"""Model-recommendation ladder + startup mismatch checks (fake specs)."""

from whisperflow import sysinfo
from whisperflow.config import ModelConfig
from whisperflow.sysinfo import Recommendation, SystemSpecs, recommend, startup_check


def specs(vram_mb=0, gpu=None, ram_gb=16.0, cores=8) -> SystemSpecs:
    return SystemSpecs(gpu_name=gpu, vram_mb=vram_mb, ram_gb=ram_gb, cpu_cores=cores)


def test_big_gpu_gets_turbo():
    rec = recommend(specs(vram_mb=8192, gpu="RTX 4060 Laptop"))
    assert (rec.engine, rec.name, rec.device) == ("local", "large-v3-turbo", "cuda")


def test_mid_gpu_gets_medium():
    rec = recommend(specs(vram_mb=4096, gpu="GTX 1650"))
    assert rec.name == "medium"
    assert rec.device == "cuda"


def test_tiny_gpu_gets_small_cuda():
    rec = recommend(specs(vram_mb=2048, gpu="MX450"))
    assert rec.name == "small"
    assert rec.device == "cuda"


def test_no_gpu_decent_cpu_gets_small_cpu():
    rec = recommend(specs(vram_mb=0, ram_gb=16, cores=8))
    assert (rec.engine, rec.name, rec.device, rec.compute_type) == ("local", "small", "cpu", "int8")


def test_weak_machine_with_key_gets_cloud():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=True)
    assert rec.engine == "gemini"
    assert "audio leaves the machine" in rec.reason


def test_weak_machine_without_key_gets_small_with_cloud_alternative():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert rec.engine == "local"
    assert rec.name == "small"
    assert any("gemini" in a for a in rec.alternatives)


def test_gpu_owner_with_key_gets_cloud_as_alternative_not_default():
    rec = recommend(specs(vram_mb=8192, gpu="RTX 4060"), has_api_key=True)
    assert rec.engine == "local"  # local stays default when hardware allows
    assert any("gemini" in a for a in rec.alternatives)


def test_startup_check_flags_cuda_without_gpu():
    cfg = ModelConfig(device="cuda")
    warning = startup_check(cfg, specs(vram_mb=0))
    assert warning and "no NVIDIA GPU" in warning


def test_startup_check_flags_big_model_small_vram():
    cfg = ModelConfig(name="large-v3-turbo", device="cuda")
    warning = startup_check(cfg, specs(vram_mb=2048, gpu="MX450"))
    assert warning and "recommend" in warning


def test_startup_check_silent_when_matched():
    cfg = ModelConfig(name="large-v3-turbo", device="cuda")
    assert startup_check(cfg, specs(vram_mb=8192, gpu="RTX 4060")) is None


# ---- autostart ----


def test_autostart_command_is_two_quoted_paths_pythonw_and_app():
    cmd = sysinfo.autostart_command()
    assert cmd.count('"') == 4  # "<pythonw>" "<app.py>"
    assert "pythonw.exe" in cmd.lower()
    assert cmd.rstrip('"').endswith("app.py")


def test_autostart_enable_query_disable_roundtrip(monkeypatch):
    # throwaway value name so the user's real WhisperFlow Run entry is untouched
    monkeypatch.setattr(sysinfo, "_RUN_VALUE", "WhisperFlowPytest")
    try:
        assert sysinfo.is_autostart_enabled() is False
        sysinfo.enable_autostart()
        assert sysinfo.is_autostart_enabled() is True
    finally:
        sysinfo.disable_autostart()
    assert sysinfo.is_autostart_enabled() is False
    sysinfo.disable_autostart()  # idempotent: no-op when already absent
