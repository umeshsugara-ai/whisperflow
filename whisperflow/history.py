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
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

STATS_SCHEMA = 1


def empty_stats() -> dict:
    return {
        "schema": STATS_SCHEMA,
        "total_words": 0,
        "total_dictations": 0,
        "total_speaking_s": 0.0,
        "days": {},
    }


def count_words(text: str) -> int:
    return len(text.split())


def accumulate(stats: dict, entry: dict) -> dict:
    """Fold one history entry into the lifetime rollup (mutates + returns stats)."""
    stats["total_words"] += count_words(entry.get("injected") or entry.get("raw", ""))
    stats["total_dictations"] += 1
    stats["total_speaking_s"] = round(
        stats["total_speaking_s"] + float(entry.get("duration_s", 0.0) or 0.0), 2
    )
    day = str(entry.get("ts", ""))[:10]
    if day:
        stats["days"][day] = stats["days"].get(day, 0) + 1
    return stats


def compute_streak(days: dict[str, int], today: str) -> int:
    """Consecutive dictation days ending today (or yesterday — today's first
    dictation may simply not have happened yet)."""
    try:
        d = date.fromisoformat(today)
    except ValueError:
        return 0
    if today not in days:
        d -= timedelta(days=1)
    streak = 0
    while d.isoformat() in days:
        streak += 1
        d -= timedelta(days=1)
    return streak


def average_wpm(stats: dict) -> float:
    minutes = float(stats.get("total_speaking_s", 0.0)) / 60.0
    return stats.get("total_words", 0) / minutes if minutes > 0 else 0.0


class History:
    def __init__(self, path: Path, max_entries: int = 500) -> None:
        self.path = path
        self.stats_path = path.with_name("stats.json")
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
            # lifetime rollup survives the trim above — dashboard totals stay true
            stats = self._read_stats()
            if stats is None:
                stats = self._seed_stats()  # jsonl already contains this entry
            else:
                accumulate(stats, entry)
            self._write_stats(stats)

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
        """Delete the entry list. Lifetime stats are a rollup and stay intact."""
        with self._lock:
            self.path.unlink(missing_ok=True)

    # ---- lifetime stats rollup (stats.json) ----

    def stats(self) -> dict:
        """Lifetime totals, independent of the max_entries trim. Seeds from the
        current jsonl when stats.json is missing (legacy installs — best
        available backfill)."""
        with self._lock:
            stats = self._read_stats()
            if stats is None:
                stats = self._seed_stats()
                self._write_stats(stats)
            return stats

    def _read_stats(self) -> dict | None:
        """Parse stats.json; None when missing/corrupt (caller reseeds). Under lock."""
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if (
            isinstance(data, dict)
            and data.get("schema") == STATS_SCHEMA
            and isinstance(data.get("days"), dict)
        ):
            base = empty_stats()
            base.update(data)
            return base
        return None

    def _seed_stats(self) -> dict:
        stats = empty_stats()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return stats
        for line in lines:
            try:
                accumulate(stats, json.loads(line))
            except json.JSONDecodeError:
                pass
        return stats

    def _write_stats(self, stats: dict) -> None:
        try:
            self.stats_path.write_text(
                json.dumps(stats, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("could not write stats.json: %s", exc)
