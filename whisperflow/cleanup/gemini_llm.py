"""Cloud LLM cleanup tier via the user's own Gemini API key.

Sends the transcript TEXT (never audio) to Gemini for context-aware polish —
fixing mis-heard Hinglish words, punctuation, and casing without rephrasing.
Strictly bounded like the Ollama tier: any failure/timeout falls back to the
rules tier for that dictation. Opt-in via [cleanup].tier = "gemini".
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "You polish voice-dictation transcripts. Fix punctuation, capitalization, "
    "obvious mis-transcriptions (especially romanized Hindi/Hinglish words), "
    "and remove filler words (um, uh, matlab, yaar). Do NOT rephrase, "
    "translate, summarize, or change the meaning. Preserve the language and "
    "script exactly as spoken (Hinglish stays in Latin script). "
    "Output ONLY the corrected text.\n\nTranscript:\n{text}"
)


def clean(text: str, model: str, api_key: str, timeout_s: float = 4.0) -> str:
    """Polish `text` via Gemini. Raises on any failure — caller falls back."""
    if not api_key:
        raise ValueError("no Gemini API key configured")
    payload = json.dumps(
        {
            "contents": [{"parts": [{"text": PROMPT.format(text=text)}]}],
            "generationConfig": {"temperature": 0.0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL.format(model=model),
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    try:
        cleaned = " ".join(
            p.get("text", "") for p in body["candidates"][0]["content"]["parts"]
        ).strip()
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"Gemini returned no text (feedback: {body.get('promptFeedback')})") from None
    if not cleaned:
        raise ValueError("Gemini returned an empty response")
    return cleaned
