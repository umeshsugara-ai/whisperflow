# -*- coding: utf-8 -*-
"""Rule-based cleanup fixtures — EN + Hinglish. The tier must strip
disfluencies but NEVER paraphrase."""

from whisperflow.cleanup import rules


def test_english_fillers_removed():
    raw = "um so I think, uh, we should ship it you know, tomorrow"
    out = rules.clean(raw)
    assert "um" not in out.lower().split()
    assert "uh" not in out.lower().split()
    assert "you know" not in out.lower()
    assert "ship it" in out
    assert "tomorrow" in out


def test_hinglish_fillers_removed():
    raw = "matlab kal meeting hai yaar, toh deck ready rakhna"
    out = rules.clean(raw)
    assert "matlab" not in out.lower()
    assert "yaar" not in out.lower()
    assert "kal meeting hai" in out.lower()
    assert "deck ready rakhna" in out.lower()


def test_repeated_words_collapsed():
    assert "the cat" in rules.clean("the the cat").lower()
    assert rules.clean("very very good").lower().count("very") == 1


def test_capitalization_and_terminal_period():
    out = rules.clean("hello world")
    assert out == "Hello world."


def test_existing_punctuation_untouched():
    out = rules.clean("Hello world!")
    assert out == "Hello world!"


def test_devanagari_untouched_by_capitalization():
    raw = "नमस्ते दुनिया, यह एक परीक्षण है"
    out = rules.clean(raw)
    assert "नमस्ते दुनिया" in out
    # no ASCII terminal period forced onto Devanagari text
    assert not out.endswith(".")


def test_content_words_never_changed():
    raw = "deploy the pathlynks build to production at five pm"
    out = rules.clean(raw)
    for word in ("deploy", "pathlynks", "build", "production", "five", "pm"):
        assert word in out.lower()


def test_extra_fillers_from_config():
    out = rules.clean("basically we need more time", extra_fillers=["basically"])
    assert "basically" not in out.lower()
    assert "we need more time" in out.lower()


def test_empty_and_whitespace():
    assert rules.clean("") == ""
    assert rules.clean("   ") == ""


def test_filler_word_inside_another_word_survives():
    # "um" inside "umbrella", "summit" must never be stripped
    out = rules.clean("the summit umbrella plan")
    assert "summit" in out.lower()
    assert "umbrella" in out.lower()
