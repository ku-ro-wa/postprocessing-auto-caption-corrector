from __future__ import annotations

from datetime import date, datetime, timezone

from caption_checker.web.usage import USAGE_FILENAME, UsageLog, ref_tag

from fake_clock import FakeClock

START = datetime(2026, 10, 1, 23, 59, tzinfo=timezone.utc)


def test_records_one_line_per_event_with_time_and_event_only(tmp_path):
    log = UsageLog(tmp_path, clock=FakeClock(START))
    log.record("upload")
    log.record("correct")

    lines = (tmp_path / USAGE_FILENAME).read_text().splitlines()
    assert lines == ["2026-10-01T23:59:00+00:00 upload", "2026-10-01T23:59:00+00:00 correct"]


def test_daily_counts_group_by_day_and_event(tmp_path):
    clock = FakeClock(START)
    log = UsageLog(tmp_path, clock=clock)
    log.record("upload")
    log.record("upload")
    clock.now = datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc)
    log.record("export")

    assert log.daily_counts() == {
        date(2026, 10, 1): {"upload": 2},
        date(2026, 10, 2): {"export": 1},
    }


def test_daily_counts_empty_without_a_file_and_skips_garbled_lines(tmp_path):
    log = UsageLog(tmp_path)
    assert log.daily_counts() == {}

    (tmp_path / USAGE_FILENAME).write_text("garbage\n2026-10-01T09:00:00+00:00 upload\n")
    assert log.daily_counts() == {date(2026, 10, 1): {"upload": 1}}


def test_unknown_event_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        UsageLog(tmp_path).record("login")


def test_a_detail_follows_the_event_and_is_counted_by_details(tmp_path):
    log = UsageLog(tmp_path, clock=FakeClock(START))
    log.record("visit", "hn")
    log.record("visit")
    log.record("visit", "hn")

    assert (tmp_path / USAGE_FILENAME).read_text().splitlines()[0] == (
        "2026-10-01T23:59:00+00:00 visit hn"
    )
    assert log.daily_counts() == {date(2026, 10, 1): {"visit": 3}}
    assert log.details("visit") == {"hn": 2, "(none)": 1}


def test_an_unsafe_detail_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        UsageLog(tmp_path).record("visit", "two words")


def test_ref_tag_keeps_only_short_tags():
    assert ref_tag(" IH ") == "ih"
    assert ref_tag("show-hn_1") == "show-hn_1"
    assert ref_tag("") is None
    assert ref_tag("a b") is None
    assert ref_tag("a" * 33) is None
