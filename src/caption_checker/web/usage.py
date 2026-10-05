"""The usage tally (ADR 0010): one line per first visit, upload, Correct run,
Free tier refusal, Export and opened Example (#65) -- a time, an event name
and at most a short detail (the link's ref tag for a visit, the Limit for a
refusal), never a Session id or any content -- appended to a file at the
data root. Like the spend ledger it
sits beside ``sessions/``, so the retention sweep never touches it. No
third-party analytics.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

USAGE_FILENAME = "usage.log"
EVENTS = ("visit", "upload", "correct", "refused", "export", "example")
_DETAIL = re.compile(r"[a-z0-9_-]{1,32}")


def ref_tag(raw: str) -> str | None:
    """A link's ``?ref=`` value as a detail to tally, or None when it's
    missing or not a short tag -- so a visitor can't write arbitrary text
    into the log."""
    tag = raw.strip().lower()
    return tag if _DETAIL.fullmatch(tag) else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UsageLog:
    def __init__(self, root: Path, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self.path = Path(root) / USAGE_FILENAME
        self._clock = clock
        self._lock = threading.Lock()

    def record(self, event: str, detail: str | None = None) -> None:
        if event not in EVENTS:
            raise ValueError(f"Unknown usage event {event!r}")
        if detail is not None and not _DETAIL.fullmatch(detail):
            raise ValueError(f"Unsafe usage detail {detail!r}")
        line = f"{self._clock().isoformat()} {event}"
        line += f" {detail}\n" if detail else "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

    def daily_counts(self) -> dict[date, dict[str, int]]:
        """Events per UTC day, oldest day first. Lines that don't parse are
        skipped."""
        days: dict[date, Counter[str]] = {}
        for day, event, _ in self._entries():
            days.setdefault(day, Counter())[event] += 1
        return {day: dict(days[day]) for day in sorted(days)}

    def details(self, event: str) -> dict[str, int]:
        """All-time counts of ``event`` by detail, most first; lines without
        one count under ``"(none)"``."""
        counts = Counter(
            detail or "(none)" for _, e, detail in self._entries() if e == event
        )
        return dict(counts.most_common())

    def _entries(self) -> list[tuple[date, str, str]]:
        """(UTC day, event, detail) per line; lines that don't parse are
        skipped."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        entries = []
        for line in text.splitlines():
            stamp, _, rest = line.partition(" ")
            event, _, detail = rest.partition(" ")
            try:
                day = datetime.fromisoformat(stamp).astimezone(timezone.utc).date()
            except ValueError:
                continue
            if event in EVENTS:
                entries.append((day, event, detail))
        return entries
