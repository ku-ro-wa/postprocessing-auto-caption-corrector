from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from caption_checker.web.free_tier import LEDGER_FILENAME
from caption_checker.web.models import TranscriptRecord, record_from_dict, record_to_dict
from caption_checker.web.storage import Storage

from fake_clock import FakeClock

DAY = timedelta(hours=24)
START = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def _save_new(storage: Storage, session_id: str) -> TranscriptRecord:
    record = TranscriptRecord(
        id=os.urandom(8).hex(),
        session_id=session_id,
        filename="a.srt",
        format="srt",
        created_at=storage.clock().isoformat(),
    )
    storage.staging_path(session_id, record.id, "srt").write_text("", encoding="utf-8")
    storage.save_transcript(record)
    return record


def _backdate(path: Path, age: timedelta) -> None:
    then = (datetime.now(timezone.utc) - age).timestamp()
    os.utime(path, (then, then))


class TestLastActivity:
    def test_saving_a_transcript_stamps_its_last_activity(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())

        clock.now = START + timedelta(hours=3)
        storage.save_transcript(record)

        loaded = storage.load_transcript(record.session_id, record.id)
        assert loaded is not None
        assert loaded.last_activity == (START + timedelta(hours=3)).isoformat()

    def test_a_record_from_before_last_activity_counts_from_its_upload(self) -> None:
        record = TranscriptRecord(
            id="t", session_id="s", filename="a.srt", format="srt",
            created_at=START.isoformat(),
        )
        data = record_to_dict(record)
        del data["last_activity"]
        assert record_from_dict(data).last_activity == START.isoformat()


class TestSweep:
    def test_an_expired_transcript_is_gone_after_a_sweep(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())

        clock.now = START + DAY
        assert storage.sweep(DAY, keep_empty_sessions_for=DAY) == 1

        assert storage.load_transcript(record.session_id, record.id) is None
        assert storage.original_path(record.session_id, record.id) is None

    def test_a_fresh_transcript_survives(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())

        clock.now = START + DAY - timedelta(minutes=1)
        assert storage.sweep(DAY, keep_empty_sessions_for=DAY) == 0

        assert storage.load_transcript(record.session_id, record.id) is not None

    def test_activity_resets_the_clock(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())

        clock.now = START + timedelta(hours=20)
        storage.save_transcript(record)
        clock.now = START + timedelta(hours=30)
        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.load_transcript(record.session_id, record.id) is not None

    def test_the_retention_period_is_the_callers(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())

        clock.now = START + timedelta(hours=2)
        storage.sweep(timedelta(hours=1), keep_empty_sessions_for=timedelta(hours=1))

        assert storage.load_transcript(record.session_id, record.id) is None

    def test_only_expired_transcripts_go(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        session_id = storage.create_session()
        old = _save_new(storage, session_id)
        clock.now = START + timedelta(hours=12)
        new = _save_new(storage, session_id)

        clock.now = START + DAY
        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.load_transcript(session_id, old.id) is None
        assert storage.load_transcript(session_id, new.id) is not None
        assert storage.session_exists(session_id)

    def test_the_spend_ledger_is_never_touched(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        _save_new(storage, storage.create_session())
        ledger = tmp_path / LEDGER_FILENAME
        ledger.write_text('{"entry": 1}\n', encoding="utf-8")

        clock.now = START + timedelta(days=30)
        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert ledger.read_text(encoding="utf-8") == '{"entry": 1}\n'

    def test_a_session_left_empty_is_removed_once_idle(self, tmp_path: Path) -> None:
        # Real time: an empty Session's idleness comes from its directories.
        clock = FakeClock(datetime.now(timezone.utc) - DAY)
        storage = Storage(tmp_path, clock=clock)
        session_id = storage.create_session()
        _save_new(storage, session_id)
        clock.now += DAY
        storage.sweep(DAY, keep_empty_sessions_for=DAY)
        session_dir = tmp_path / "sessions" / session_id
        _backdate(session_dir / "transcripts", DAY + timedelta(minutes=1))
        _backdate(session_dir, DAY + timedelta(minutes=1))

        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert not storage.session_exists(session_id)

    def test_an_empty_session_used_recently_is_kept(self, tmp_path: Path) -> None:
        # Removing it would give the visitor a new Session, and with it a
        # fresh Allowance: an empty Session stays until it's been idle for
        # as long as a run's charge can count against its Allowance.
        storage = Storage(tmp_path)
        session_id = storage.create_session()
        record = _save_new(storage, session_id)
        storage.delete_transcript(session_id, record.id)

        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.session_exists(session_id)

    def test_an_empty_session_is_kept_for_the_longer_idle_period(
        self, tmp_path: Path
    ) -> None:
        storage = Storage(tmp_path)
        session_id = storage.create_session()
        _backdate(tmp_path / "sessions" / session_id, timedelta(hours=2))

        storage.sweep(timedelta(hours=1), keep_empty_sessions_for=DAY)

        assert storage.session_exists(session_id)

    def test_an_upload_still_being_staged_survives(self, tmp_path: Path) -> None:
        storage = Storage(tmp_path)
        session_id = storage.create_session()
        storage.staging_path(session_id, "staged", "srt").write_text("", encoding="utf-8")

        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.original_path(session_id, "staged") is not None

    def test_a_long_abandoned_staged_upload_is_removed(self, tmp_path: Path) -> None:
        storage = Storage(tmp_path)
        session_id = storage.create_session()
        original = storage.staging_path(session_id, "staged", "srt")
        original.write_text("", encoding="utf-8")
        _backdate(original.parent, DAY)

        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.original_path(session_id, "staged") is None


class TestRacesWithDeletion:
    def test_a_save_after_deletion_leaves_nothing_behind(self, tmp_path: Path) -> None:
        # A Correct run that finishes after its Transcript was swept or
        # deleted mustn't bring back a record with no original to parse.
        storage = Storage(tmp_path)
        record = _save_new(storage, storage.create_session())
        storage.delete_transcript(record.session_id, record.id)

        storage.save_transcript(record)

        assert storage.load_transcript(record.session_id, record.id) is None
        assert storage.list_transcripts(record.session_id) == []

    def test_touch_stamps_activity_without_rewriting_the_record(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        record = _save_new(storage, storage.create_session())
        stale = storage.load_transcript(record.session_id, record.id)
        assert stale is not None
        record.priming_terms = "saved meanwhile"
        storage.save_transcript(record)

        clock.now = START + timedelta(hours=5)
        storage.touch(stale)

        loaded = storage.load_transcript(record.session_id, record.id)
        assert loaded is not None
        assert loaded.priming_terms == "saved meanwhile"
        assert loaded.last_activity == (START + timedelta(hours=5)).isoformat()

    def test_touching_a_deleted_transcript_leaves_nothing_behind(
        self, tmp_path: Path
    ) -> None:
        storage = Storage(tmp_path)
        record = _save_new(storage, storage.create_session())
        storage.delete_transcript(record.session_id, record.id)

        storage.touch(record)

        assert storage.load_transcript(record.session_id, record.id) is None


class TestSweepRobustness:
    def test_an_unreadable_record_does_not_stop_the_sweep(self, tmp_path: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path, clock=clock)
        session_id = storage.create_session()
        broken = _save_new(storage, session_id)
        expired = _save_new(storage, session_id)
        state = tmp_path / "sessions" / session_id / "transcripts" / broken.id / "state.json"
        state.write_text("{not json", encoding="utf-8")

        clock.now = START + DAY
        storage.sweep(DAY, keep_empty_sessions_for=DAY)

        assert storage.load_transcript(session_id, expired.id) is None
        assert state.exists()
