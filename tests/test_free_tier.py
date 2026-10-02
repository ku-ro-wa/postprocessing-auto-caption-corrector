from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from caption_checker.web.free_tier import (
    LEDGER_FILENAME,
    FreeTier,
    LimitReached,
    Limits,
)

START = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def _tier(
    root: Path, clock: _Clock, *, words: int = 10_000, budget: float = 2.0
) -> FreeTier:
    return FreeTier(
        root, Limits(allowance_words=words, daily_budget_usd=budget), clock=clock
    )


def _run(
    tier: FreeTier,
    session: str,
    words: int,
    *,
    estimate: float = 0.01,
    cost: float | None = None,
    produced: bool = True,
) -> None:
    reservation = tier.reserve(session, words, estimate_usd=estimate)
    tier.settle(reservation, produced=produced, cost_usd=estimate if cost is None else cost)


def _ledger(root: Path) -> list[dict]:
    lines = (root / LEDGER_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines]


class TestAllowance:
    def test_a_new_session_has_the_whole_allowance(self, tmp_path: Path, clock: _Clock) -> None:
        assert _tier(tmp_path, clock).words_left("s1") == 10_000

    def test_a_run_is_charged_its_full_word_count_and_reruns_again(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 3_000)
        _run(tier, "s1", 3_000)
        assert tier.words_left("s1") == 4_000
        assert tier.words_left("s2") == 10_000

    def test_grace_lets_a_run_cross_the_line_then_the_balance_floors_at_zero(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 9_000)
        # 1,000 left + 20% of 10,000 = 3,000 may start.
        _run(tier, "s1", 3_000)
        assert tier.words_left("s1") == 0

    def test_a_run_past_the_grace_is_refused(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 9_000)
        with pytest.raises(LimitReached) as exc:
            tier.reserve("s1", 3_001, estimate_usd=0.01)
        assert exc.value.limit == "allowance"

    def test_no_debt_carries_into_the_next_window(self, tmp_path: Path, clock: _Clock) -> None:
        # The run that crossed the line used only what was left, so when the
        # earlier run ages out the whole of it comes back.
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 9_000)
        clock.now += timedelta(hours=1)
        _run(tier, "s1", 3_000)
        clock.now = START + timedelta(hours=24, seconds=1)
        assert tier.words_left("s1") == 9_000

    def test_a_run_that_produced_no_result_charges_no_words(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 3_000, produced=False)
        assert tier.words_left("s1") == 10_000

    def test_the_allowance_is_rolling_24_hours(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 4_000)
        clock.now += timedelta(hours=23)
        assert tier.words_left("s1") == 6_000
        clock.now += timedelta(hours=1, seconds=1)
        assert tier.words_left("s1") == 10_000


class TestDailyBudget:
    def test_a_run_whose_estimate_would_pass_the_budget_is_refused(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock, budget=0.25)
        _run(tier, "s1", 100, cost=0.20)
        with pytest.raises(LimitReached) as exc:
            tier.reserve("s2", 100, estimate_usd=0.06)
        assert exc.value.limit == "daily_budget"
        tier.reserve("s2", 100, estimate_usd=0.05)

    def test_the_budget_is_charged_the_actual_cost_even_when_no_result(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock, budget=0.25)
        _run(tier, "s1", 100, estimate=0.01, cost=0.24, produced=False)
        with pytest.raises(LimitReached):
            tier.reserve("s2", 100, estimate_usd=0.02)

    def test_the_budget_is_rolling_24_hours(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock, budget=0.25)
        _run(tier, "s1", 100, cost=0.25)
        clock.now += timedelta(hours=24, seconds=1)
        tier.reserve("s2", 100, estimate_usd=0.2)


class TestConcurrentRuns:
    def test_a_run_in_flight_holds_its_words(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock)
        tier.reserve("s1", 8_000, estimate_usd=0.01)
        assert tier.words_left("s1") == 2_000
        with pytest.raises(LimitReached):
            tier.reserve("s1", 8_000, estimate_usd=0.01)

    def test_a_run_in_flight_holds_its_estimate(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock, budget=0.25)
        tier.reserve("s1", 100, estimate_usd=0.2)
        with pytest.raises(LimitReached):
            tier.reserve("s2", 100, estimate_usd=0.1)

    def test_of_two_runs_at_once_only_one_that_fits_starts(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        start = threading.Barrier(2)
        outcomes: list[str] = []

        def run() -> None:
            start.wait()
            try:
                tier.reserve("s1", 8_000, estimate_usd=0.01)
                outcomes.append("started")
            except LimitReached:
                outcomes.append("refused")

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(outcomes) == ["refused", "started"]

    def test_settling_releases_the_hold(self, tmp_path: Path, clock: _Clock) -> None:
        tier = _tier(tmp_path, clock, budget=0.25)
        reservation = tier.reserve("s1", 8_000, estimate_usd=0.2)
        tier.settle(reservation, produced=False, cost_usd=0.0)
        tier.reserve("s1", 8_000, estimate_usd=0.2)


class TestLedger:
    def test_one_entry_per_run_with_time_session_words_and_cost(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 3_000, cost=0.0123)
        assert _ledger(tmp_path) == [
            {
                "at": START.isoformat(),
                "session": "s1",
                "words": 3_000,
                "cost_usd": 0.0123,
                "transcript_words": 3_000,
            }
        ]

    def test_a_grace_run_records_its_full_words_beside_the_charge(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 9_000)
        _run(tier, "s1", 3_000)
        last = _ledger(tmp_path)[-1]
        assert (last["words"], last["transcript_words"]) == (1_000, 3_000)

    def test_totals_survive_a_restart(self, tmp_path: Path, clock: _Clock) -> None:
        _run(_tier(tmp_path, clock, budget=0.25), "s1", 4_000, cost=0.2)
        tier = _tier(tmp_path, clock, budget=0.25)
        assert tier.words_left("s1") == 6_000
        with pytest.raises(LimitReached):
            tier.reserve("s2", 100, estimate_usd=0.1)

    def test_startup_prunes_entries_older_than_the_window(
        self, tmp_path: Path, clock: _Clock
    ) -> None:
        tier = _tier(tmp_path, clock)
        _run(tier, "s1", 1_000)
        clock.now += timedelta(hours=12)
        _run(tier, "s1", 2_000)
        clock.now += timedelta(hours=13)
        _tier(tmp_path, clock)
        assert [e["words"] for e in _ledger(tmp_path)] == [2_000]

    def test_an_unreadable_line_is_skipped(self, tmp_path: Path, clock: _Clock) -> None:
        # A crash mid-append can leave half a line.
        _run(_tier(tmp_path, clock), "s1", 1_000)
        with (tmp_path / LEDGER_FILENAME).open("a") as f:
            f.write('{"at": "2026-10')
        assert _tier(tmp_path, clock).words_left("s1") == 9_000
