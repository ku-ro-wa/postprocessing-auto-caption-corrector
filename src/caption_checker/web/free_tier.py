"""The Free tier (ADR 0008): Read-through runs paid by the server's own
OpenRouter key, metered by each Session's word Allowance and a global Daily
budget, both over a rolling 24 hours.

Both limits are checked before a run against its pre-run estimate, and a run
that starts always finishes. A run holds its words and estimate from
:meth:`FreeTier.reserve` until :meth:`FreeTier.settle`, so two concurrent runs
can't both pass a check only one of them fits within.

Every settled run appends one entry (time, Session, words, cost) to the spend
ledger, a JSON-lines file at the data root, under an in-process lock -- safe
because the deployment runs exactly one server process (ADR 0010). It sits
outside ``sessions/``, so deleting Transcripts never resets an Allowance.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

LEDGER_FILENAME = "spend-ledger.jsonl"
WINDOW = timedelta(hours=24)
#: A run may start if its words fit in what's left plus this share of the
#: daily Allowance.
GRACE = 0.2

Limit = Literal["allowance", "daily_budget"]


@dataclass(frozen=True)
class Limits:
    """The Free tier's settings. Defaults are ADR 0010's launch figures."""

    allowance_words: int = 10_000
    daily_budget_usd: float = 0.25
    donate_url: str | None = None


class LimitReached(Exception):
    """A Free tier run was refused before it started: ``limit`` says which,
    and ``wait`` how long until enough of it comes back for this run -- None
    when the run would never fit, or only runs still in flight hold it."""

    def __init__(self, limit: Limit, wait: timedelta | None = None) -> None:
        super().__init__(f"Free tier {limit.replace('_', ' ')} used up")
        self.limit = limit
        self.wait = wait


@dataclass(frozen=True)
class _Entry:
    at: datetime
    session_id: str
    words: int
    cost_usd: float
    transcript_words: int


@dataclass(eq=False)
class Reservation:
    """A run's hold on the limits while it's in flight. ``words`` is what
    the Allowance will be charged if the run produces a result: the
    Transcript's full word count, but never more than was left, so a run in
    the grace margin leaves the balance at zero rather than owing.
    ``transcript_words`` is that full count, kept for measuring real use."""

    session_id: str
    words: int
    estimate_usd: float
    transcript_words: int


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FreeTier:
    """The Free tier's limits and spend ledger. Loads the ledger at
    construction, dropping entries older than the window."""

    def __init__(
        self,
        root: Path,
        limits: Limits,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.limits = limits
        self.path = Path(root) / LEDGER_FILENAME
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: list[Reservation] = []
        self._entries = self._load()
        self._prune()

    # -- queries -------------------------------------------------------

    def words_left(self, session_id: str) -> int:
        with self._lock:
            return self._words_left(session_id)

    def next_return(self, session_id: str) -> tuple[timedelta, int] | None:
        """How long until the Session's oldest charged run in the window ages
        out, and the words it gives back; None when nothing is charged."""
        with self._lock:
            charged = [e for e in self._session_recent(session_id) if e.words]
            if not charged:
                return None
            oldest = min(charged, key=lambda e: e.at)
            return self._until_aged_out(oldest), oldest.words

    # -- a run -----------------------------------------------------------

    def reserve(self, session_id: str, words: int, *, estimate_usd: float) -> Reservation:
        """Check a run of ``words`` costing about ``estimate_usd`` against
        both limits and hold them for it, or raise :class:`LimitReached`."""
        with self._lock:
            left = self._words_left(session_id)
            if not self._allowance_fits(words, left):
                raise LimitReached("allowance", self._allowance_wait(session_id, words))
            if not self._budget_fits(estimate_usd, self._spent_usd()):
                raise LimitReached("daily_budget", self._budget_wait(estimate_usd))
            reservation = Reservation(session_id, min(words, left), estimate_usd, words)
            self._pending.append(reservation)
            return reservation

    def settle(self, reservation: Reservation, *, produced: bool, cost_usd: float) -> None:
        """Release ``reservation`` and record the run: the Daily budget is
        charged ``cost_usd`` whatever happened, the Allowance only when the
        run ``produced`` a result."""
        entry = _Entry(
            at=self._clock(),
            session_id=reservation.session_id,
            words=reservation.words if produced else 0,
            cost_usd=cost_usd,
            transcript_words=reservation.transcript_words,
        )
        with self._lock:
            self._pending.remove(reservation)
            self._entries.append(entry)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(_dump(entry) + "\n")

    # -- internals (callers hold the lock) ------------------------------

    def _recent(self) -> list[_Entry]:
        since = self._clock() - WINDOW
        return [e for e in self._entries if e.at > since]

    def _session_recent(self, session_id: str) -> list[_Entry]:
        return [e for e in self._recent() if e.session_id == session_id]

    def _until_aged_out(self, entry: _Entry) -> timedelta:
        return entry.at + WINDOW - self._clock()

    def _words_used(self, session_id: str) -> int:
        used = sum(e.words for e in self._session_recent(session_id))
        return used + sum(r.words for r in self._pending if r.session_id == session_id)

    def _words_left(self, session_id: str) -> int:
        return max(0, self.limits.allowance_words - self._words_used(session_id))

    def _spent_usd(self) -> float:
        return sum(e.cost_usd for e in self._recent()) + sum(
            r.estimate_usd for r in self._pending
        )

    def _allowance_fits(self, words: int, left: int) -> bool:
        return words <= left + GRACE * self.limits.allowance_words

    def _budget_fits(self, estimate_usd: float, spent_usd: float) -> bool:
        return spent_usd + estimate_usd <= self.limits.daily_budget_usd

    def _allowance_wait(self, session_id: str, words: int) -> timedelta | None:
        """Age the Session's runs out oldest first until ``words`` fits."""
        used = self._words_used(session_id)
        for entry in sorted(self._session_recent(session_id), key=lambda e: e.at):
            used -= entry.words
            if self._allowance_fits(words, max(0, self.limits.allowance_words - used)):
                return self._until_aged_out(entry)
        return None

    def _budget_wait(self, estimate_usd: float) -> timedelta | None:
        """Age every run out oldest first until ``estimate_usd`` fits."""
        spent = self._spent_usd()
        for entry in sorted(self._recent(), key=lambda e: e.at):
            spent -= entry.cost_usd
            if self._budget_fits(estimate_usd, spent):
                return self._until_aged_out(entry)
        return None

    def _load(self) -> list[_Entry]:
        if not self.path.is_file():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(_parse(line))
            except (ValueError, KeyError, TypeError):
                continue  # half a line left by a crash mid-append
        return entries

    def _prune(self) -> None:
        """Drop entries older than the window, from memory and the file."""
        self._entries = self._recent()
        if not self.path.is_file():
            return
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(_dump(e) + "\n" for e in self._entries), encoding="utf-8")
        os.replace(tmp, self.path)


def _dump(entry: _Entry) -> str:
    return json.dumps(
        {
            "at": entry.at.isoformat(),
            "session": entry.session_id,
            "words": entry.words,
            "cost_usd": entry.cost_usd,
            "transcript_words": entry.transcript_words,
        }
    )


def _parse(line: str) -> _Entry:
    data = json.loads(line)
    return _Entry(
        at=datetime.fromisoformat(data["at"]),
        session_id=str(data["session"]),
        words=int(data["words"]),
        cost_usd=float(data["cost_usd"]),
        transcript_words=int(data.get("transcript_words", data["words"])),
    )
