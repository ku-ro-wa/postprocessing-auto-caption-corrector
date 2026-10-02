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
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

from caption_checker.parser import parse
from caption_checker.web.models import (
    TranscriptRecord,
    TranscriptSummary,
    record_from_dict,
    record_to_dict,
)

SESSION_ID_BYTES = 16


def default_data_dir() -> Path:
    override = os.environ.get("CAPTION_CHECKER_DATA_DIR")
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "caption-checker" / "web"


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

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
        transcript_dir = self._transcript_dir(record.session_id, record.id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(transcript_dir / "state.json", record_to_dict(record))

    def load_transcript(self, session_id: str, transcript_id: str) -> TranscriptRecord | None:
        data = self._read_json(self._transcript_dir(session_id, transcript_id) / "state.json")
        if data is None:
            return None
        return record_from_dict(data)

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
                )
            )
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def delete_transcript(self, session_id: str, transcript_id: str) -> None:
        transcript_dir = self._transcript_dir(session_id, transcript_id)
        if not transcript_dir.is_dir():
            return
        for child in transcript_dir.iterdir():
            child.unlink()
        transcript_dir.rmdir()

    def discard_staged(self, session_id: str, transcript_id: str) -> None:
        """Remove a partially-created Transcript directory after a failed
        upload parse, so no record is left behind for an invalid file."""
        self.delete_transcript(session_id, transcript_id)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
