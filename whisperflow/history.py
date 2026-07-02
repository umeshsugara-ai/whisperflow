"""Dictation history — the non-destructive contract.

Every dictation appends one JSONL entry containing BOTH the raw transcript
(exactly what the STT heard) and the injected text (after cleanup +
dictionary). The raw text is therefore always recoverable, no matter what
the cleanup tier did — the anti-Wispr guarantee.

Entry: {ts, raw, injected, tier, method, language, duration_s, latency_ms}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class History:
    def __init__(self, path: Path, max_entries: int = 500) -> None:
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()

    def append(
        self,
        raw: str,
        injected: str,
        tier: str,
        method: str,
        language: str,
        duration_s: float,
        latency_ms: float,
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "raw": raw,
            "injected": injected,
            "tier": tier,
            "method": method,
            "language": language,
            "duration_s": round(duration_s, 2),
            "latency_ms": round(latency_ms, 1),
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._trim()

    def _trim(self) -> None:
        """Keep at most max_entries lines (called under lock)."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        if len(lines) > self.max_entries:
            self.path.write_text(
                "\n".join(lines[-self.max_entries :]) + "\n", encoding="utf-8"
            )

    def entries(self, limit: int = 50) -> list[dict]:
        """Most recent entries, newest last."""
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                return []
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping corrupt history line")
        return out

    def last(self) -> dict | None:
        items = self.entries(limit=1)
        return items[0] if items else None

    def clear(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)
