"""Text-processing pipeline: cleanup tier + dictionary replacements.

Builds the controller's `process_text` hook from config. Tier resolution:
- "off"   -> raw text untouched
- "rules" -> rule-based cleanup
- "llm"   -> Ollama cleanup; ANY failure (down, timeout, empty) falls back
             to rules for that dictation — dictation must never block on
             an optional dependency.
Dictionary replacements apply after cleanup in every tier.
The RAW text is preserved upstream (controller/history), never here.
"""

from __future__ import annotations

import logging
from typing import Callable

from whisperflow.cleanup import gemini_llm, ollama_llm, rules
from whisperflow.config import CleanupConfig, DictionaryConfig
from whisperflow.dictionary import apply_replacements

log = logging.getLogger(__name__)


def build_processor(
    cleanup_cfg: CleanupConfig,
    dict_cfg: DictionaryConfig,
    llm_available: bool | None = None,
    gemini_api_key: str = "",
    gemini_model: str = "",
) -> Callable[[str, str], tuple[str, str]]:
    """Return process_text(raw, language) -> (final_text, tier_used)."""

    if llm_available is None and cleanup_cfg.tier == "llm":
        llm_available = ollama_llm.health_check(cleanup_cfg.llm_url)
        if not llm_available:
            log.warning(
                "cleanup tier is 'llm' but Ollama is unreachable at %s — degrading to rules",
                cleanup_cfg.llm_url,
            )
    if cleanup_cfg.tier == "gemini":
        if gemini_api_key:
            log.warning("cleanup tier 'gemini': transcript TEXT (not audio) will be sent to Google")
        else:
            log.warning("cleanup tier is 'gemini' but no API key is configured — degrading to rules")

    def _rules(raw: str) -> str:
        return rules.clean(raw, cleanup_cfg.extra_fillers)

    def process(raw: str, language: str) -> tuple[str, str]:
        tier_used = cleanup_cfg.tier
        if cleanup_cfg.tier == "off":
            text = raw
        elif cleanup_cfg.tier == "llm" and llm_available:
            try:
                text = ollama_llm.clean(
                    raw,
                    model=cleanup_cfg.llm_model,
                    base_url=cleanup_cfg.llm_url,
                    timeout_s=cleanup_cfg.llm_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("llm cleanup failed (%s); falling back to rules", exc)
                text = _rules(raw)
                tier_used = "rules-fallback"
        elif cleanup_cfg.tier == "gemini" and gemini_api_key:
            try:
                text = gemini_llm.clean(
                    raw,
                    model=gemini_model or cleanup_cfg.gemini_model,
                    api_key=gemini_api_key,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("gemini cleanup failed (%s); falling back to rules", exc)
                text = _rules(raw)
                tier_used = "rules-fallback"
        else:
            text = _rules(raw)
            if cleanup_cfg.tier in ("llm", "gemini"):
                tier_used = "rules-fallback"

        return apply_replacements(text, dict_cfg), tier_used

    return process
