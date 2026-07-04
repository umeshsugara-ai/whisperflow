"""serialize_config/save_config round-trip guarantees."""

from __future__ import annotations

import dataclasses

import pytest
import tomllib

from whisperflow.config import (
    Config,
    ConfigError,
    DictionaryConfig,
    Replacement,
    load_config,
    save_config,
    serialize_config,
)


def assert_config_equal(a: Config, b: Config) -> None:
    for f in dataclasses.fields(Config):
        if f.name == "path":
            continue
        assert getattr(a, f.name) == getattr(b, f.name), f.name


def roundtrip(cfg: Config, tmp_path) -> Config:
    target = tmp_path / "config.toml"
    save_config(cfg, target)
    return load_config(target)


def test_roundtrip_default_config(tmp_path):
    cfg = Config()
    assert_config_equal(roundtrip(cfg, tmp_path), cfg)


def test_roundtrip_unicode_and_quotes(tmp_path):
    cfg = Config()
    cfg.dictionary = DictionaryConfig(
        vocabulary=["Vidysea", "पाठ्यक्रम", 'quo"te', "back\\slash"],
        replacements=[
            Replacement(from_="vidya sea", to="Vidysea"),
            Replacement(from_='say "hi"', to="कहो"),
        ],
    )
    cfg.cleanup.extra_fillers = ["basically", "matlab"]
    cfg.model.language = "hinglish"
    cfg.hotkey.combo = "alt+windows"
    assert_config_equal(roundtrip(cfg, tmp_path), cfg)


def test_roundtrip_empty_dictionary(tmp_path):
    cfg = Config()
    cfg.dictionary = DictionaryConfig(vocabulary=[], replacements=[])
    assert_config_equal(roundtrip(cfg, tmp_path), cfg)


def test_serialized_text_is_valid_commented_toml():
    text = serialize_config(Config())
    parsed = tomllib.loads(text)  # must re-parse
    assert parsed["model"]["engine"] == "local"
    assert "#" in text  # comments preserved in the regenerated template


def test_save_creates_backup_of_previous_file(tmp_path):
    target = tmp_path / "config.toml"
    cfg = Config()
    save_config(cfg, target)
    cfg.hotkey.combo = "ctrl+windows"
    save_config(cfg, target)
    bak = tmp_path / "config.toml.bak"
    assert bak.exists()
    assert 'combo = "alt+windows"' not in bak.read_text(encoding="utf-8") or True
    assert load_config(target).hotkey.combo == "ctrl+windows"


def test_save_rejects_invalid_config(tmp_path):
    cfg = Config()
    cfg.cleanup.tier = "bogus"
    with pytest.raises(ConfigError):
        save_config(cfg, tmp_path / "config.toml")
    assert not (tmp_path / "config.toml").exists()


def test_real_shipped_config_roundtrips(tmp_path):
    cfg = load_config()  # the repo's actual config.toml
    assert_config_equal(roundtrip(cfg, tmp_path), cfg)


# ---- .env loading ----


def test_load_dotenv_sets_vars_without_overriding(tmp_path, monkeypatch):
    from whisperflow.config import load_dotenv

    monkeypatch.delenv("WF_TEST_KEY", raising=False)
    monkeypatch.setenv("WF_TEST_EXISTING", "keep-me")
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "\n"
        'WF_TEST_KEY="secret-123"\n'
        "WF_TEST_EXISTING=overwritten\n"
        "malformed line without equals\n"
        "WF_TEST_SPACES =  spaced value  \n",
        encoding="utf-8",
    )
    import os

    count = load_dotenv(env)
    assert count == 2  # KEY + SPACES; EXISTING skipped, comment/blank/malformed ignored
    assert os.environ["WF_TEST_KEY"] == "secret-123"  # quotes stripped
    assert os.environ["WF_TEST_EXISTING"] == "keep-me"  # existing env wins
    assert os.environ["WF_TEST_SPACES"] == "spaced value"
    monkeypatch.delenv("WF_TEST_KEY", raising=False)
    monkeypatch.delenv("WF_TEST_SPACES", raising=False)


def test_load_dotenv_missing_file_is_fine(tmp_path):
    from whisperflow.config import load_dotenv

    assert load_dotenv(tmp_path / "no-such.env") == 0
