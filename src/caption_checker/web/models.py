"""Web-layer records: Session, Transcript, Review Decision.

These sit on top of the core ``caption_checker.models`` (``Flag``) and
``caption_checker.corrector`` (``Correction``) rather than replacing them —
a Transcript here is a persisted wrapper around one file's Flags/Corrections
plus the reviewer's per-Flag Review Decisions. See ``CONTEXT.md`` for the
glossary these names follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from caption_checker.corrector import Correction
from caption_checker.models import Flag, flag_to_dict

DecisionStatus = Literal["pending", "accepted", "rejected"]


@dataclass
class ReviewDecision:
    """A reviewer's disposition on one Flag. ``text`` is only meaningful
    when ``status == "accepted"`` — it's the exact replacement text Export
    writes back into the Cue."""

    status: DecisionStatus = "pending"
    text: str | None = None


@dataclass
class TranscriptRecord:
    """One uploaded Transcript: its Flags (from the automatic local scan),
    any Corrections (from an explicit ``correct`` run), and the reviewer's
    Review Decisions — one per Flag, aligned by list index."""

    id: str
    session_id: str
    filename: str
    format: str
    created_at: str
    flags: list[Flag] = field(default_factory=list)
    corrections: list[Correction | None] = field(default_factory=list)
    decisions: list[ReviewDecision] = field(default_factory=list)
    correct_error: str | None = None

    @property
    def reviewed_count(self) -> int:
        return sum(1 for d in self.decisions if d.status != "pending")

    @property
    def has_corrections(self) -> bool:
        return any(c is not None for c in self.corrections)


@dataclass
class TranscriptSummary:
    """Lightweight listing row for a Session's transcript index."""

    id: str
    filename: str
    format: str
    created_at: str
    flag_count: int
    reviewed_count: int


def flag_from_dict(data: dict) -> Flag:
    return Flag(
        span=data["span"],
        global_indices=list(data["global_indices"]),
        cue_index=data["cue_index"],
        start=timedelta(seconds=data["start"]),
        end=timedelta(seconds=data["end"]),
        detector=data["detector"],
        reason=data["reason"],
        candidates=list(data.get("candidates", [])),
        confidence=data.get("confidence", 0.0),
        context=data.get("context", ""),
    )


def correction_to_dict(correction: Correction | None) -> dict | None:
    if correction is None:
        return None
    return {
        "id": correction.id,
        "replacement": correction.replacement,
        "confidence": correction.confidence,
        "rationale": correction.rationale,
    }


def correction_from_dict(data: dict | None) -> Correction | None:
    if data is None:
        return None
    return Correction(
        id=data["id"],
        replacement=data["replacement"],
        confidence=data["confidence"],
        rationale=data.get("rationale", ""),
    )


def decision_to_dict(decision: ReviewDecision) -> dict:
    return {"status": decision.status, "text": decision.text}


def decision_from_dict(data: dict) -> ReviewDecision:
    return ReviewDecision(status=data.get("status", "pending"), text=data.get("text"))


def record_to_dict(record: TranscriptRecord) -> dict:
    return {
        "id": record.id,
        "session_id": record.session_id,
        "filename": record.filename,
        "format": record.format,
        "created_at": record.created_at,
        "flags": [flag_to_dict(f) for f in record.flags],
        "corrections": [correction_to_dict(c) for c in record.corrections],
        "decisions": [decision_to_dict(d) for d in record.decisions],
        "correct_error": record.correct_error,
    }


def record_from_dict(data: dict) -> TranscriptRecord:
    flags = [flag_from_dict(f) for f in data["flags"]]
    corrections = [correction_from_dict(c) for c in data.get("corrections", [])]
    decisions = [decision_from_dict(d) for d in data.get("decisions", [])]
    return TranscriptRecord(
        id=data["id"],
        session_id=data["session_id"],
        filename=data["filename"],
        format=data["format"],
        created_at=data["created_at"],
        flags=flags,
        corrections=corrections,
        decisions=decisions,
        correct_error=data.get("correct_error"),
    )
