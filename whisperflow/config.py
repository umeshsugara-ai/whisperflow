"""Config loading/validation/saving for WhisperFlow.

TOML at the app root (next to app.py). stdlib tomllib, dataclass views,
validation with actionable error messages, in-place reload support, and a
serializer so the Settings/Dictionary UI can persist changes (stdlib has no
TOML writer; we regenerate the canonical commented template with current
values so a GUI save stays as self-documenting as the shipped file).
"""

from __future__ import annotations

import os
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
    paste_threshold_chars: int = 1500
    type_interval_ms: int = 5
    modifier_release_timeout_ms: int = 2000  # wait for Ctrl/Alt/Shift/Win release before injecting; 0 = off
    clipboard_restore_delay_ms: int = 600  # paste mode: how long the target gets to read the clipboard


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
class StartupConfig:
    auto_register: bool = True  # on first run, register autostart in the HKCU Run key


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
    startup: StartupConfig = field(default_factory=StartupConfig)
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

    cfg = _build_config(section, replacements, d, cfg_path)
    _validate(cfg)
    return cfg


def _build_config(section, replacements, d, cfg_path) -> Config:
    return Config(
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
        startup=StartupConfig(
            **{k: v for k, v in section("startup").items() if k in StartupConfig.__dataclass_fields__}
        ),
        path=cfg_path,
    )


# ---- serialization (Settings/Dictionary UI persistence) --------------------


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return repr(v)  # int | float


def serialize_config(cfg: Config) -> str:
    """The full commented config.toml text with `cfg`'s current values.

    Pure function; round-trips with load_config for every dataclass field.
    User-added comments are replaced by the canonical ones — the price of a
    GUI save (targeted line-editing breaks on arrays-of-tables).
    """
    t = _toml_value
    m, hk, a, c, i, dic, h, o, s = (
        cfg.model, cfg.hotkey, cfg.audio, cfg.cleanup, cfg.inject,
        cfg.dictionary, cfg.history, cfg.overlay, cfg.startup,
    )
    replacement_blocks = "".join(
        f"\n[[dictionary.replacements]]\nfrom = {t(r.from_)}\nto = {t(r.to)}\n"
        for r in dic.replacements
    )
    return f'''# WhisperFlow configuration
# Edit freely; use the tray menu "Reload config" to apply most changes without restart.
# (This file is regenerated when you save from the Settings/Dictionary screens.)

[model]
# Not sure what fits your machine? Run:  python app.py --recommend
engine = {t(m.engine)}           # local (fully on-device, private) | gemini (BYOK cloud — audio goes to Google)

# --- local engine ---
# Registry names: large-v3-turbo | large-v3 | medium | small
# OR any raw HuggingFace CTranslate2 repo id, e.g. "Systran/faster-whisper-medium"
name = {t(m.name)}
device = {t(m.device)}            # cuda | cpu
compute_type = {t(m.compute_type)}  # int8_float16 (CUDA cc>=7.0) | float16 | int8 (cpu)
beam_size = {t(m.beam_size)}
vad = {t(m.vad)}
language = {t(m.language)}      # "" = auto-detect | "en" | "hi" (Devanagari output) |
                           # "hinglish" = Roman-script Hindi+English mix ("kya tum sun rahe ho")

# --- cloud engine (only when engine = "gemini"; bring your own key) ---
cloud_model = {t(m.cloud_model)}   # audio-input model; gemini-2.5-pro for higher accuracy
api_key = {t(m.api_key)}                        # prefer the env var below instead of pasting a key here
api_key_env = {t(m.api_key_env)}      # env var read when api_key is empty

[hotkey]
combo = {t(hk.combo)}      # keys joined by +; parsed by the `keyboard` library
# Other good options: "ctrl+windows" | "windows+space" (conflicts with the
# keyboard-layout switcher if you use multiple input languages)
# Avoid "alt+space" — it's the Windows system-menu shortcut.
tap_threshold_ms = {t(hk.tap_threshold_ms)}     # release faster than this = toggle mode; held longer = hold-to-talk
double_tap_ms = {t(hk.double_tap_ms)}        # 0 = off; >0 = double-tap the combo to START dictation
                           # (Wispr-style). A later single tap stops it.

[audio]
device = {t(a.device)}         # "default" = system default (re-checked every recording), or a name substring
max_seconds = {t(a.max_seconds)}          # hard recording cap
min_seconds = {t(a.min_seconds)}          # discard shorter recordings
silence_rms = {t(a.silence_rms)}       # below this RMS the recording is treated as silence (no transcription)

[cleanup]
tier = {t(c.tier)}             # off | rules | llm (Ollama, local) | gemini (cloud text-polish, BYOK)
llm_model = {t(c.llm_model)}
llm_url = {t(c.llm_url)}
llm_timeout_s = {t(c.llm_timeout_s)}
gemini_model = {t(c.gemini_model)}  # cheapest Gemini tier — enough for text polish, keeps cost minimal
extra_fillers = {t(c.extra_fillers)}         # additional filler words to strip in rules tier, e.g. ["basically"]

[inject]
method = {t(i.method)}            # auto | type | paste
paste_threshold_chars = {t(i.paste_threshold_chars)}  # in auto mode, text longer than this uses clipboard-paste
type_interval_ms = {t(i.type_interval_ms)}       # delay between typed chunks
modifier_release_timeout_ms = {t(i.modifier_release_timeout_ms)}  # wait (ms) for Ctrl/Alt/Shift/Win to be physically released
                           # before injecting — held modifiers corrupt typed text. 0 = off.
clipboard_restore_delay_ms = {t(i.clipboard_restore_delay_ms)}    # paste mode: time the target app gets to read the clipboard
                           # before the previous clipboard content is restored

[dictionary]
# Vocabulary biases recognition (fed to whisper as initial_prompt)
vocabulary = {t(dic.vocabulary)}

# Replacement rules applied to the transcript AFTER STT (case-insensitive match)
{replacement_blocks}
[history]
max_entries = {t(h.max_entries)}          # history.jsonl trimmed to this many entries

[overlay]
always_visible = {t(o.always_visible)}      # keep the pill on-screen at rest so you always see the app is alive
show_hint = {t(o.show_hint)}           # briefly show the hotkey (e.g. "● Alt+Win") on first show, then
                           # settle to a compact resting pill that expands on hover

[startup]
auto_register = {t(s.auto_register)}       # on first run, register WhisperFlow to start at Windows login
                           # (HKCU Run key, windowless). Toggle anytime via tray → "Start on
                           # Windows login", or run: python app.py --install/--uninstall-autostart
'''


def save_config(cfg: Config, path: Path | None = None) -> None:
    """Validate, back up the existing file, then atomically write cfg as TOML."""
    _validate(cfg)
    target = Path(path) if path else cfg.path
    text = serialize_config(cfg)
    if target.exists():
        target.with_name(target.name + ".bak").write_text(
            target.read_text(encoding="utf-8"), encoding="utf-8"
        )
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
