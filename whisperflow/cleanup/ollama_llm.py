"""Optional LLM cleanup tier via a local Ollama server.

Strictly bounded: punctuation + filler removal ONLY, no rephrasing — and a
hard timeout after which the caller falls back to the rules tier for that
dictation. If Ollama isn't running at startup, the tier is unavailable and
the app degrades to rules with a notice (nothing is bundled or installed).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You clean up voice-dictation transcripts. Fix punctuation and "
    "capitalization, and remove filler words (um, uh, matlab, yaar). "
    "Do NOT rephrase, translate, summarize, or change any other words. "
    "Preserve the language exactly as spoken, including Hindi and Hinglish. "
    "Output ONLY the cleaned text with no commentary."
)


def health_check(base_url: str, timeout_s: float = 2.0) -> bool:
    """True if an Ollama server answers at base_url."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def clean(text: str, model: str, base_url: str, timeout_s: float = 3.0) -> str:
    """LLM cleanup. Raises on any failure — the caller handles fallback."""
    payload = json.dumps(
        {
            "model": model,
            "system": SYSTEM_PROMPT,
            "prompt": text,
            "stream": False,
            "options": {"temperature": 0.0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    cleaned = (body.get("response") or "").strip()
    if not cleaned:
        raise ValueError("Ollama returned an empty response")
    return cleaned
