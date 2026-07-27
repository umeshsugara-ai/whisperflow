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


def test_processor_groq_tier_with_mock(monkeypatch):
    from whisperflow import processing

    seen = {}

    def fake_clean(text, model, api_key):
        seen["model"] = model
        seen["key"] = api_key
        return "Kya tum sun rahe ho?"

    monkeypatch.setattr(processing.groq_llm, "clean", fake_clean)
    cfg = CleanupConfig(tier="groq")
    process = build_processor(cfg, dict_cfg(), groq_api_key="gk")
    text, tier = process("kiatum sunrayo", "hi")
    assert tier == "groq"
    assert text == "Kya tum sun rahe ho?"
    assert seen["model"] == "llama-3.3-70b-versatile"  # free default; 8b dropped words in live testing
    assert seen["key"] == "gk"


def test_processor_groq_without_key_falls_back():
    cfg = CleanupConfig(tier="groq")
    process = build_processor(cfg, dict_cfg(), groq_api_key="")
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "hello world" in text.lower()


def test_processor_groq_failure_falls_back(monkeypatch):
    from whisperflow import processing

    def boom(*a, **k):
        raise TimeoutError("groq down")

    monkeypatch.setattr(processing.groq_llm, "clean", boom)
    cfg = CleanupConfig(tier="groq")
    process = build_processor(cfg, dict_cfg(), groq_api_key="gk")
    text, tier = process("um hello world", "en")
    assert tier == "rules-fallback"
    assert "hello world" in text.lower()


def test_groq_clean_request_shape(monkeypatch):
    """The wire format: system+user chat messages, Bearer auth, temp 0."""
    import io
    import json as _json
    import urllib.request

    from whisperflow.cleanup import groq_llm

    captured = {}

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = _json.loads(req.data.decode("utf-8"))
        reply = _json.dumps(
            {"choices": [{"message": {"content": " polished text "}}]}
        ).encode("utf-8")

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp(reply)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = groq_llm.clean("raw text", model="llama-3.1-8b-instant", api_key="gk")
    assert out == "polished text"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["auth"] == "Bearer gk"
    assert captured["body"]["model"] == "llama-3.1-8b-instant"
    assert captured["body"]["temperature"] == 0.0
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["system", "user"]
    assert captured["body"]["messages"][1]["content"] == "raw text"
    # the polish prompt must allow spelling fixes but forbid rephrasing
    system = captured["body"]["messages"][0]["content"]
    assert "mis-transcriptions" in system
    assert "Do NOT rephrase" in system


def test_groq_clean_requires_key():
    import pytest

    from whisperflow.cleanup import groq_llm

    with pytest.raises(ValueError):
        groq_llm.clean("text", model="m", api_key="")


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
