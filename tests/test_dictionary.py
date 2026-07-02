# -*- coding: utf-8 -*-
"""Dictionary: vocabulary prompt + replacement rules + processor pipeline."""

from whisperflow.config import CleanupConfig, DictionaryConfig, Replacement
from whisperflow.dictionary import apply_replacements, vocabulary_prompt
from whisperflow.processing import build_processor


def dict_cfg():
    return DictionaryConfig(
        vocabulary=["Vidysea", "Pathlynks"],
        replacements=[
            Replacement(from_="vidya sea", to="Vidysea"),
            Replacement(from_="path links", to="Pathlynks"),
        ],
    )


def test_vocabulary_prompt_joined():
    assert vocabulary_prompt(dict_cfg()) == "Vidysea, Pathlynks"


def test_replacements_case_insensitive():
    out = apply_replacements("open Vidya Sea and PATH LINKS now", dict_cfg())
    assert "Vidysea" in out
    assert "Pathlynks" in out


def test_replacement_whole_phrase_only():
    # "path links" inside a longer token must not match
    out = apply_replacements("sympathlinkstest stays", dict_cfg())
    assert "sympathlinkstest" in out


def test_processor_off_tier_returns_raw():
    process = build_processor(CleanupConfig(tier="off"), dict_cfg(), llm_available=False)
    text, tier = process("um hello vidya sea", "en")
    assert tier == "off"
    assert "um" in text  # off = untouched except dictionary
    assert "Vidysea" in text  # replacements still apply


def test_processor_rules_tier():
    process = build_processor(CleanupConfig(tier="rules"), dict_cfg(), llm_available=False)
    text, tier = process("um hello path links", "en")
    assert tier == "rules"
    assert "um" not in text.lower().split()
    assert "Pathlynks" in text


def test_processor_llm_unreachable_falls_back_to_rules():
    cfg = CleanupConfig(tier="llm", llm_url="http://localhost:1")  # nothing listens here
    process = build_processor(cfg, dict_cfg())  # health_check runs -> False
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "um" not in text.lower().split()
    assert "hello world" in text.lower()


def test_processor_gemini_tier_with_mock(monkeypatch):
    from whisperflow import processing

    seen = {}

    def fake_clean(text, model, api_key):
        seen["model"] = model
        return "Kya tum sun rahe ho?"

    monkeypatch.setattr(processing.gemini_llm, "clean", fake_clean)
    cfg = CleanupConfig(tier="gemini")
    process = build_processor(cfg, dict_cfg(), gemini_api_key="k")
    text, tier = process("kiatum sunrayo", "hi")
    assert tier == "gemini"
    assert text == "Kya tum sun rahe ho?"
    # defaults to the cheapest tier when no override given
    assert seen["model"] == "gemini-2.5-flash-lite"


def test_processor_gemini_without_key_falls_back():
    cfg = CleanupConfig(tier="gemini")
    process = build_processor(cfg, dict_cfg(), gemini_api_key="")
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "hello world" in text.lower()


def test_processor_gemini_failure_falls_back(monkeypatch):
    from whisperflow import processing

    def boom(*a, **k):
        raise TimeoutError("gemini down")

    monkeypatch.setattr(processing.gemini_llm, "clean", boom)
    cfg = CleanupConfig(tier="gemini")
    process = build_processor(cfg, dict_cfg(), gemini_api_key="k")
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "hello world" in text.lower()


def test_processor_llm_mid_run_failure_falls_back(monkeypatch):
    from whisperflow import processing

    cfg = CleanupConfig(tier="llm", llm_url="http://localhost:1", llm_timeout_s=0.1)

    def boom(*args, **kwargs):
        raise TimeoutError("ollama died mid-run")

    monkeypatch.setattr(processing.ollama_llm, "clean", boom)
    process = build_processor(cfg, dict_cfg(), llm_available=True)  # pretend it was up at startup
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "hello world" in text.lower()
