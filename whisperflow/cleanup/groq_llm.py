"""Cloud LLM cleanup tier via the user's existing Groq API key.

The zero-extra-setup polish path: anyone using the recommended Groq STT
engine already has GROQ_API_KEY configured, and Groq's free LLM tier is fast
enough (~300ms) for live chunking — no Ollama install, no 2GB model
download, no second key. Sends transcript TEXT only (never audio).
Strictly bounded like the other LLM tiers: any failure/timeout falls back
to the rules tier for that dictation. Opt-in via [cleanup].tier = "groq".
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You polish voice-dictation transcripts. Fix punctuation, capitalization, "
    "obvious mis-transcriptions (especially romanized Hindi/Hinglish words, "
    "e.g. 'nighi'->'nahi', 'liqha'->'likha'), and remove filler words "
    "(um, uh, matlab, yaar). Do NOT rephrase, translate, summarize, or "
    "change the meaning or word order. Preserve the language and script "
    "exactly as spoken (Hinglish stays in Latin script). "
    "Output ONLY the corrected text with no commentary."
)


def clean(text: str, model: str, api_key: str, timeout_s: float = 4.0) -> str:
    """Polish `text` via Groq chat completions. Raises on any failure —
    the caller handles the rules fallback."""
    if not api_key:
        raise ValueError("no Groq API key configured")
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Groq sits behind Cloudflare, which 403s urllib's default agent
            "User-Agent": "WhisperFlow/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    cleaned = (body["choices"][0]["message"]["content"] or "").strip()
    if not cleaned:
        raise ValueError("Groq returned an empty response")
    return cleaned
