"""Personal dictionary: vocabulary bias + post-STT replacement rules.

- vocabulary  -> joined into whisper's initial_prompt so domain terms
                 (Vidysea, Pathlynks, ...) are recognized correctly;
- replacements -> case-insensitive whole-phrase fixes applied AFTER
                 transcription+cleanup ("vidya sea" -> "Vidysea").
"""

from __future__ import annotations

import re

from whisperflow.config import DictionaryConfig


def vocabulary_prompt(cfg: DictionaryConfig) -> str:
    return ", ".join(v.strip() for v in cfg.vocabulary if v.strip())


def apply_replacements(text: str, cfg: DictionaryConfig) -> str:
    out = text
    for rule in cfg.replacements:
        if not rule.from_.strip():
            continue
        pattern = re.compile(rf"(?i)\b{re.escape(rule.from_)}\b")
        out = pattern.sub(rule.to, out)
    return out
