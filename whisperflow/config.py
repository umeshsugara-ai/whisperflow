"""Config loading/validation for WhisperFlow.

TOML at the app root (next to app.py). stdlib tomllib, dataclass views,
validation with actionable error messages, and in-place reload support.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_ROOT / "config.toml"

VALID_DEVICES = {"cuda", "cpu"}
VALID_COMPUTE_TYPES = {"int8_float16", "float16", "int8", "float32"}
VALID_CLEANUP_TIERS = {"off", "rules", "llm", "gemini"}
VALID_INJECT_METHODS = {"auto", "type", "paste"}
VALID_ENGINES = {"local", "gemini"}


@dataclass
class ModelConfig:
    engine: str = "local"  # local (faster-whisper, on-device) | gemini (BYOK cloud)
    name: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 1
    vad: bool = True
    language: str = ""  # "" = auto-detect
    # cloud engine (BYOK) settings — only used when engine != "local"
    cloud_model: str = "gemini-2.5-flash"
    api_key: str = ""  # inline key (prefer api_key_env)
    api_key_env: str = "GEMINI_API_KEY"  # env var read when api_key is empty

    def resolve_api_key(self) -> str:
        import os

        return self.api_key or os.environ.get(self.api_key_env, "")


@dataclass
class HotkeyConfig:
    combo: str = "ctrl+windows"
    tap_threshold_ms: int = 350
    double_tap_ms: int = 0  # 0 = off; >0 enables double-tap-to-start


@dataclass
class AudioConfig:
    device: str = "default"
    max_seconds: float = 120.0
    min_seconds: float = 0.3
    silence_rms: float = 0.0005


@dataclass
class CleanupConfig:
    tier: str = "rules"
    llm_model: str = "qwen2.5:3b-instruct"
    llm_url: str = "http://localhost:11434"
    llm_timeout_s: float = 3.0
    gemini_model: str = "gemini-2.5-flash-lite"  # cheapest tier — text polish doesn't need more
    extra_fillers: list[str] = field(default_factory=list)


@dataclass
class InjectConfig:
    method: str = "auto"
    paste_threshold_chars: int = 500
    type_interval_ms: int = 5


@dataclass
class Replacement:
    from_: str
    to: str


@dataclass
class DictionaryConfig:
    vocabulary: list[str] = field(default_factory=list)
    replacements: list[Replacement] = field(default_factory=list)


@dataclass
class OverlayConfig:
    always_visible: bool = True  # keep the pill on-screen at rest (Wispr-style)
    show_hint: bool = True  # briefly show "Alt+Win to talk" on first show


@dataclass
class HistoryConfig:
    max_entries: int = 500


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    path: Path = DEFAULT_CONFIG_PATH


class ConfigError(ValueError):
    """Raised when config.toml has an invalid value."""


def _validate(cfg: Config) -> None:
    m = cfg.model
    if m.engine not in VALID_ENGINES:
        raise ConfigError(f"[model].engine must be one of {sorted(VALID_ENGINES)}, got {m.engine!r}")
    if m.engine != "local" and not m.resolve_api_key():
        raise ConfigError(
            f"[model].engine = {m.engine!r} needs an API key: set [model].api_key "
            f"or the {m.api_key_env} environment variable"
        )
    if m.device not in VALID_DEVICES:
        raise ConfigError(f"[model].device must be one of {sorted(VALID_DEVICES)}, got {m.device!r}")
    if m.compute_type not in VALID_COMPUTE_TYPES:
        raise ConfigError(
            f"[model].compute_type must be one of {sorted(VALID_COMPUTE_TYPES)}, got {m.compute_type!r}"
        )
    if m.beam_size < 1:
        raise ConfigError(f"[model].beam_size must be >= 1, got {m.beam_size}")
    if cfg.cleanup.tier not in VALID_CLEANUP_TIERS:
        raise ConfigError(f"[cleanup].tier must be one of {sorted(VALID_CLEANUP_TIERS)}, got {cfg.cleanup.tier!r}")
    if cfg.inject.method not in VALID_INJECT_METHODS:
        raise ConfigError(f"[inject].method must be one of {sorted(VALID_INJECT_METHODS)}, got {cfg.inject.method!r}")
    if not cfg.hotkey.combo.strip():
        raise ConfigError("[hotkey].combo must not be empty")
    if cfg.audio.max_seconds <= cfg.audio.min_seconds:
        raise ConfigError("[audio].max_seconds must be greater than [audio].min_seconds")


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate config.toml. Missing keys fall back to defaults."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)

    def section(name: str) -> dict:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"[{name}] must be a table")
        return value

    d = section("dictionary")
    replacements = [
        Replacement(from_=r["from"], to=r["to"])
        for r in d.get("replacements", [])
        if isinstance(r, dict) and "from" in r and "to" in r
    ]

    cfg = Config(
        model=ModelConfig(**{k: v for k, v in section("model").items() if k in ModelConfig.__dataclass_fields__}),
        hotkey=HotkeyConfig(**{k: v for k, v in section("hotkey").items() if k in HotkeyConfig.__dataclass_fields__}),
        audio=AudioConfig(**{k: v for k, v in section("audio").items() if k in AudioConfig.__dataclass_fields__}),
        cleanup=CleanupConfig(
            **{k: v for k, v in section("cleanup").items() if k in CleanupConfig.__dataclass_fields__}
        ),
        inject=InjectConfig(**{k: v for k, v in section("inject").items() if k in InjectConfig.__dataclass_fields__}),
        dictionary=DictionaryConfig(vocabulary=list(d.get("vocabulary", [])), replacements=replacements),
        history=HistoryConfig(
            **{k: v for k, v in section("history").items() if k in HistoryConfig.__dataclass_fields__}
        ),
        overlay=OverlayConfig(
            **{k: v for k, v in section("overlay").items() if k in OverlayConfig.__dataclass_fields__}
        ),
        path=cfg_path,
    )
    _validate(cfg)
    return cfg
