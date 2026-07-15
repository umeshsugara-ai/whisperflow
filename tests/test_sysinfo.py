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
    assert rec.engine == "groq"
    assert "audio leaves the machine" in rec.reason


def test_weak_machine_without_key_gets_small_with_cloud_alternative():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert rec.engine == "local"
    assert rec.name == "small"
    assert any("groq" in a for a in rec.alternatives)


def test_no_gpu_weak_cpu_without_key_still_mentions_groq_as_free_option():
    # even with NO key, the free-tier cloud option should be surfaced —
    # unlike the old gemini-only behavior, groq needs no pre-existing key
    # to be worth recommending (it's free to sign up for).
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert any("groq" in a.lower() for a in rec.alternatives)


def test_gpu_owner_with_key_gets_cloud_as_alternative_not_default():
    rec = recommend(specs(vram_mb=8192, gpu="RTX 4060"), has_api_key=True)
    assert rec.engine == "local"  # local stays default when hardware allows
    assert any("groq" in a for a in rec.alternatives)


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


def test_autostart_command_store_python_uses_wscript(monkeypatch):
    monkeypatch.setattr(sysinfo, "_is_store_python", lambda: True)
    cmd = sysinfo.autostart_command()
    assert "wscript.exe" in cmd.lower()
    assert cmd.rstrip('"').lower().endswith("run.vbs")
    assert "//B" in cmd
    assert "pythonw" not in cmd.lower()  # the silently-failing alias must never be registered


def test_autostart_command_non_store_uses_pythonw(monkeypatch):
    monkeypatch.setattr(sysinfo, "_is_store_python", lambda: False)
    cmd = sysinfo.autostart_command()
    assert "pythonw.exe" in cmd.lower()
    assert cmd.endswith(" --autostart")
    assert '"' + str(sysinfo._APP_ROOT / "app.py") + '"' in cmd


def test_autostart_enable_query_disable_roundtrip(monkeypatch):
    # throwaway value name so the user's real WhisperFlow Run entry is untouched
    monkeypatch.setattr(sysinfo, "_RUN_VALUE", "WhisperFlowPytest")
    try:
        assert sysinfo.is_autostart_enabled() is False
        assert sysinfo.get_autostart_command() is None
        sysinfo.enable_autostart()
        assert sysinfo.is_autostart_enabled() is True
        assert sysinfo.get_autostart_command() == sysinfo.autostart_command()
    finally:
        sysinfo.disable_autostart()
    assert sysinfo.is_autostart_enabled() is False
    sysinfo.disable_autostart()  # idempotent: no-op when already absent


def test_ensure_autostart_first_run_registers_and_writes_sentinel(monkeypatch, tmp_path):
    monkeypatch.setattr(sysinfo, "_RUN_VALUE", "WhisperFlowPytest")
    sentinel = tmp_path / ".autostart_initialized"
    try:
        sysinfo.ensure_autostart(sentinel)
        assert sysinfo.get_autostart_command() == sysinfo.autostart_command()
        assert sentinel.exists()
    finally:
        sysinfo.disable_autostart()


def test_ensure_autostart_heals_stale_entry(monkeypatch, tmp_path):
    import winreg

    monkeypatch.setattr(sysinfo, "_RUN_VALUE", "WhisperFlowPytest")
    sentinel = tmp_path / ".autostart_initialized"
    sentinel.write_text("1", encoding="utf-8")
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, sysinfo._RUN_KEY) as key:
            winreg.SetValueEx(
                key, "WhisperFlowPytest", 0, winreg.REG_SZ, '"broken\\pythonw.exe" "app.py"'
            )
        sysinfo.ensure_autostart(sentinel)
        assert sysinfo.get_autostart_command() == sysinfo.autostart_command()
    finally:
        sysinfo.disable_autostart()


def test_show_event_roundtrip(monkeypatch):
    import ctypes

    # throwaway event name so a running WhisperFlow instance is untouched
    monkeypatch.setattr(sysinfo, "_SHOW_EVENT", "Global\\WhisperFlowPytestShowEvent")
    assert sysinfo.signal_show_event() is False  # no instance listening
    handle = sysinfo.create_show_event()
    assert handle
    try:
        assert sysinfo.signal_show_event() is True
        assert sysinfo.wait_show_event(handle, 0) is True
        assert sysinfo.wait_show_event(handle, 0) is False  # auto-reset consumed it
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def test_ensure_autostart_respects_opt_out(monkeypatch, tmp_path):
    monkeypatch.setattr(sysinfo, "_RUN_VALUE", "WhisperFlowPytest")
    sentinel = tmp_path / ".autostart_initialized"
    sentinel.write_text("1", encoding="utf-8")  # user disabled via tray after first run
    sysinfo.ensure_autostart(sentinel)
    assert sysinfo.get_autostart_command() is None  # must NOT re-register


# ---- pure config builders ----


def test_build_recommended_config_local():
    rec = Recommendation(
        engine="local", name="large-v3-turbo", device="cuda", compute_type="int8_float16",
        reason="test", alternatives=[],
    )
    cfg = sysinfo.build_recommended_config(rec)
    assert cfg.model.engine == "local"
    assert cfg.model.name == "large-v3-turbo"
    assert cfg.model.device == "cuda"
    assert cfg.model.compute_type == "int8_float16"


def test_build_recommended_config_cloud_sets_api_key_env_from_registry():
    rec = Recommendation(
        engine="groq", name="whisper-large-v3-turbo", device="cpu", compute_type="int8",
        reason="test", alternatives=[],
    )
    cfg = sysinfo.build_recommended_config(rec)
    assert cfg.model.engine == "groq"
    assert cfg.model.cloud_model == "whisper-large-v3-turbo"
    assert cfg.model.api_key_env == "GROQ_API_KEY"


def test_build_config_for_engine_local_reuses_recommend_ladder():
    cfg = sysinfo.build_config_for_engine("local", specs(vram_mb=8192, gpu="RTX 4060"))
    assert cfg.model.engine == "local"
    assert cfg.model.name == "large-v3-turbo"  # matches the big-GPU ladder branch
    assert cfg.model.device == "cuda"


def test_build_config_for_engine_cloud_uses_provider_default():
    cfg = sysinfo.build_config_for_engine("openai", specs(vram_mb=0, ram_gb=4, cores=2))
    assert cfg.model.engine == "openai"
    assert cfg.model.cloud_model == "gpt-4o-transcribe"
    assert cfg.model.api_key_env == "OPENAI_API_KEY"
