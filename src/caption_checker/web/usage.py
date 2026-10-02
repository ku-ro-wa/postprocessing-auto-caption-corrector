"""The usage tally (ADR 0010): one line per upload, Correct run and Export --
a time and an event name, never a Session id or any content -- appended to a
file at the data root. Like the spend ledger it sits beside ``sessions/``, so
the retention sweep never touches it. No third-party analytics.
"""

from __future__ import annotations

import threading
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

USAGE_FILENAME = "usage.log"
EVENTS = ("upload", "correct", "export")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UsageLog:
    def __init__(self, root: Path, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self.path = Path(root) / USAGE_FILENAME
        self._clock = clock
        self._lock = threading.Lock()

    def record(self, event: str) -> None:
        if event not in EVENTS:
            raise ValueError(f"Unknown usage event {event!r}")
        line = f"{self._clock().isoformat()} {event}\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def daily_counts(self) -> dict[date, dict[str, int]]:
        """Events per UTC day, oldest day first. Lines that don't parse are
        skipped."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        days: dict[date, Counter[str]] = {}
        for line in text.splitlines():
            stamp, _, event = line.partition(" ")
            try:
                day = datetime.fromisoformat(stamp).astimezone(timezone.utc).date()
            except ValueError:
                continue
            if event in EVENTS:
                days.setdefault(day, Counter())[event] += 1
        return {day: dict(days[day]) for day in sorted(days)}
