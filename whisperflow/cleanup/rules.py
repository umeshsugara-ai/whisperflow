"""Rule-based cleanup tier — deterministic, zero VRAM, <1ms.

A light touch, not a rewrite: whisper large-v3-turbo already emits decent
punctuation and capitalization, so this tier only strips fillers, collapses
stutter-repeats, normalizes whitespace, and tidies sentence casing. It must
NEVER paraphrase — that is the #1 documented complaint against Wispr Flow.
"""

from __future__ import annotations

import re

# Standalone fillers, English + Hinglish. Matched as whole words,
# optionally followed by a comma. Kept conservative on purpose: words like
# "like" or "matlab" are only fillers in some positions, so we only strip
# the unambiguous comma-delimited/leading forms.
DEFAULT_FILLERS = [
    "um",
    "uh",
    "umm",
    "uhh",
    "hmm",
    "erm",
    "you know",
    "i mean",
    "matlab",
    "mtlb",
    "yaar",
    "haan toh",
    "toh basically",
]


def _filler_pattern(fillers: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(f) for f in sorted(fillers, key=len, reverse=True))
    # filler surrounded by boundaries, eaten together with a trailing comma/space
    return re.compile(rf"(?i)(?:(?<=^)|(?<=[\s,.!?]))(?:{alternatives})(?:\s*,)?(?=\s|[.!?,]|$)")


_REPEAT_WORD = re.compile(r"(?i)\b(\w+)(\s+\1)+\b")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_DUP_COMMA = re.compile(r",\s*,+")
_ORPHAN_PUNCT_START = re.compile(r"^\s*[,.;:]\s*")


def clean(text: str, extra_fillers: list[str] | None = None) -> str:
    """Apply rule-based cleanup. Input and output are both 'what was said' —
    only disfluencies and mechanical noise are removed."""
    if not text.strip():
        return text.strip()

    fillers = DEFAULT_FILLERS + [f for f in (extra_fillers or []) if f.strip()]
    out = _filler_pattern(fillers).sub("", text)

    out = _REPEAT_WORD.sub(r"\1", out)  # "the the" -> "the"
    out = _DUP_COMMA.sub(",", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = _ORPHAN_PUNCT_START.sub("", out)
    out = out.strip()

    if not out:
        return out

    # sentence-initial capitalization (ASCII only — never touch Devanagari)
    def cap(match: re.Match) -> str:
        return match.group(1) + match.group(2).upper()

    out = re.sub(r"(^|[.!?]\s+)([a-z])", cap, out)

    # terminal punctuation if the text ends mid-air (latin scripts only)
    if re.search(r"[A-Za-z0-9]$", out):
        out += "."
    return out
