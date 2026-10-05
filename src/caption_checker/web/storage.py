"""Filesystem persistence for Sessions and Transcripts.

Layout, per ADR-0003 (persisted, session-scoped):

    <root>/sessions/<session_id>/transcripts/<transcript_id>/original.<ext>
    <root>/sessions/<session_id>/transcripts/<transcript_id>/state.json

A Session directory's existence *is* the session. Nothing about a visitor's
OpenRouter key is stored here: the browser keeps it and sends it with each
Correct (ADR 0010), and a ``session.json`` an older version left with a key
in it is never read. A Transcript's Cues are never duplicated into
``state.json`` — they're re-parsed from ``original.<ext>`` on load, per the
spec's "or a pointer to the stored original file, re-parsed on load."

Nothing here lasts (ADR 0010): :meth:`Storage.sweep` deletes each Transcript
a retention period after its ``last_activity``, which every save stamps, and
then any Session left empty. It only walks ``sessions/``, so the Free tier's
spend ledger beside it is never touched.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from caption_checker.parser import parse
from caption_checker.web.models import (
    TranscriptRecord,
    TranscriptSummary,
    record_from_dict,
    record_to_dict,
)

SESSION_ID_BYTES = 16

logger = logging.getLogger(__name__)


def default_data_dir() -> Path:
    override = os.environ.get("CAPTION_CHECKER_DATA_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "caption-checker" / "web"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Storage:
    def __init__(self, root: Path, *, clock: Callable[[], datetime] = _utcnow) -> None:
        self.root = Path(root)
        self.clock = clock

    # -- Sessions ----------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id

    def session_exists(self, session_id: str) -> bool:
        return self._session_dir(session_id).is_dir()

    def create_session(self) -> str:
        session_id = secrets.token_hex(SESSION_ID_BYTES)
        self._session_dir(session_id).mkdir(parents=True, exist_ok=True)
        return session_id

    # -- Transcripts ---------------------------------------------------

    def _transcript_dir(self, session_id: str, transcript_id: str) -> Path:
        return self._session_dir(session_id) / "transcripts" / transcript_id

    def staging_path(self, session_id: str, transcript_id: str, suffix: str) -> Path:
        """Where an upload's bytes land before the parse is validated. Also
        the Transcript's permanent original-file location, so a successful
        parse needs no extra move."""
        transcript_dir = self._transcript_dir(session_id, transcript_id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        return transcript_dir / f"original.{suffix}"

    def original_path(self, session_id: str, transcript_id: str) -> Path | None:
        transcript_dir = self._transcript_dir(session_id, transcript_id)
        if not transcript_dir.is_dir():
            return None
        matches = sorted(transcript_dir.glob("original.*"))
        return matches[0] if matches else None

    def load_cues(self, session_id: str, transcript_id: str) -> list:
        original = self.original_path(session_id, transcript_id)
        if original is None:
            raise FileNotFoundError(f"No stored original for transcript {transcript_id}")
        return parse(original)

    def save_transcript(self, record: TranscriptRecord) -> None:
        """Persist ``record``, stamping it with activity now: every change a
        reviewer makes is saved, so a save resets the retention clock."""
        record.last_activity = self.clock().isoformat()
        transcript_dir = self._transcript_dir(record.session_id, record.id)
        if self.original_path(record.session_id, record.id) is None:
            # Deleted (or swept) while a request held it: leave it deleted
            # rather than bring back a record with no original to parse.
            return
        self._write_json(transcript_dir / "state.json", record_to_dict(record))

    def touch(self, record: TranscriptRecord) -> None:
        """Stamp ``record``'s Transcript with activity now, rewriting only its
        stored ``last_activity`` -- so a request that only reads, like
        Export, can't write back a record another request has changed."""
        stored = self.load_transcript(record.session_id, record.id)
        if stored is not None:
            self.save_transcript(stored)
            record.last_activity = stored.last_activity

    def load_transcript(self, session_id: str, transcript_id: str) -> TranscriptRecord | None:
        data = self._read_json(self._transcript_dir(session_id, transcript_id) / "state.json")
        if data is None:
            return None
        return record_from_dict(data)

    def find_transcript(self, transcript_id: str) -> TranscriptRecord | None:
        """The Transcript ``transcript_id`` in whichever Session holds it,
        for the operator's tools; a visitor's requests go by their own
        Session."""
        for state in self.root.glob(f"sessions/*/transcripts/{transcript_id}/state.json"):
            data = self._read_json(state)
            if data is not None:
                return record_from_dict(data)
        return None

    def list_transcripts(self, session_id: str) -> list[TranscriptSummary]:
        transcripts_dir = self._session_dir(session_id) / "transcripts"
        if not transcripts_dir.is_dir():
            return []
        summaries = []
        for transcript_dir in sorted(transcripts_dir.iterdir()):
            data = self._read_json(transcript_dir / "state.json")
            if data is None:
                continue
            record = record_from_dict(data)
            summaries.append(
                TranscriptSummary(
                    id=record.id,
                    filename=record.filename,
                    format=record.format,
                    created_at=record.created_at,
                    flag_count=len(record.flags),
                    reviewed_count=record.reviewed_count,
                    is_example=record.is_example,
                )
            )
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def delete_transcript(self, session_id: str, transcript_id: str) -> None:
        transcript_dir = self._transcript_dir(session_id, transcript_id)
        shutil.rmtree(transcript_dir, ignore_errors=True)

    def discard_staged(self, session_id: str, transcript_id: str) -> None:
        """Remove a partially-created Transcript directory after a failed
        upload parse, so no record is left behind for an invalid file."""
        self.delete_transcript(session_id, transcript_id)

    # -- Retention -----------------------------------------------------

    def sweep(self, retention: timedelta, *, keep_empty_sessions_for: timedelta) -> int:
        """Delete every Transcript whose last activity is ``retention`` or
        more ago, and return how many went. A Transcript directory with no
        record yet (an upload being staged, or one a crash abandoned) goes
        once its directory is that old; one whose record can't be read is
        logged and left.

        Then remove each Session left without Transcripts that has been idle
        -- nothing added or deleted -- for ``keep_empty_sessions_for``. Its
        visitor gets a new Session, and with it a new Allowance, so the caller
        keeps an empty Session at least as long as a run's charge counts
        against its Allowance."""
        now = self.clock()
        sessions_dir = self.root / "sessions"
        if not sessions_dir.is_dir():
            return 0
        swept = 0
        for session_dir in sessions_dir.iterdir():
            transcripts_dir = session_dir / "transcripts"
            for transcript_dir in transcripts_dir.iterdir() if transcripts_dir.is_dir() else ():
                try:
                    expired = now - self._last_activity(transcript_dir) >= retention
                except (OSError, ValueError, KeyError, TypeError):
                    logger.exception("Can't read %s; leaving it", transcript_dir)
                    continue
                if expired:
                    shutil.rmtree(transcript_dir, ignore_errors=True)
                    swept += 1
            self._remove_if_idle(session_dir, now - keep_empty_sessions_for)
        return swept

    def _remove_if_idle(self, session_dir: Path, idle_since: datetime) -> None:
        """Remove ``session_dir`` if it holds no Transcripts and nothing in it
        has changed since ``idle_since``. ``rmdir`` only removes an empty
        directory, so an upload staged meanwhile keeps the Session."""
        transcripts_dir = session_dir / "transcripts"
        try:
            dirs = [d for d in (session_dir, transcripts_dir) if d.is_dir()]
            if max(self._mtime(d) for d in dirs) > idle_since:
                return
            if transcripts_dir.is_dir():
                transcripts_dir.rmdir()
            session_dir.rmdir()
        except OSError:
            pass  # not empty, or a request is using it

    def _last_activity(self, transcript_dir: Path) -> datetime:
        data = self._read_json(transcript_dir / "state.json")
        if data is None:
            return self._mtime(transcript_dir)
        return datetime.fromisoformat(record_from_dict(data).last_activity)

    @staticmethod
    def _mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
