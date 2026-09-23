"""Orchestration for the web layer: upload+scan, the LLM correct pass,
recording Review Decisions, and assembling an Export. Calls the same core
entry points the CLI uses (``parser.parse``/``serialize``, ``detect.detect``,
and the ``corrector`` module's ``Corrector`` seam) — no new core-pipeline
logic lives here.

Unlike the CLI's own ``correct`` command (``correct.py``, with its
interactive/threshold ``Reviewer`` and its own ``ReviewDecision``/cache/
bypass machinery for a single synchronous run), the web layer's job is to
call an LLM for a verdict and then hold each Flag's Review Decision open
across many separate HTTP requests — so only the ``Corrector`` protocol
itself (the actual "ask an LLM" seam) is reused here, not ``correct.py``'s
CLI-specific orchestration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from caption_checker.correct import CHUNK_SIZE
from caption_checker.corrector import (
    Corrector,
    CorrectorError,
    Correction,
    FlagContext,
    MissingAPIKeyError,
    OpenRouterCorrector,
)
from caption_checker.detect import detect
from caption_checker.models import DEFAULT_MODEL, DetectConfig, Flag
from caption_checker.parser import serialize
from caption_checker.vocab import Vocab, load_vocab
from caption_checker.web.models import ReviewDecision, TranscriptRecord
from caption_checker.web.storage import Storage

# Per the spec's "fixed defaults" decision: matches the CLI's defaults
# exactly (default OOV threshold, built-in vocab only). No tuning UI in this
# spec.
SCAN_CONFIG = DetectConfig()

SUPPORTED_FORMATS = ("srt", "vtt")


class InvalidTranscriptError(ValueError):
    """Raised when an upload isn't a parseable SRT/VTT file."""


@lru_cache(maxsize=1)
def _default_vocab() -> Vocab:
    return load_vocab(algo=SCAN_CONFIG.phonetic_algo)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upload_transcript(
    storage: Storage, session_id: str, filename: str, content: bytes
) -> TranscriptRecord:
    """Validate, persist, and automatically scan an uploaded file.

    Raises ``InvalidTranscriptError`` before any Transcript record is
    created when the upload isn't a parseable SRT/VTT.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise InvalidTranscriptError(
            f"Unsupported caption format {Path(filename).suffix!r} "
            f"(expected .srt or .vtt)"
        )

    transcript_id = uuid4().hex
    original_path = storage.staging_path(session_id, transcript_id, suffix)
    original_path.write_bytes(content)

    try:
        cues = storage.load_cues(session_id, transcript_id)
    except Exception as exc:
        storage.discard_staged(session_id, transcript_id)
        raise InvalidTranscriptError(f"Couldn't parse {filename} as {suffix}: {exc}") from exc

    if not cues:
        storage.discard_staged(session_id, transcript_id)
        raise InvalidTranscriptError(f"{filename} has no cues to review.")

    flags = detect(cues, vocab=_default_vocab(), config=SCAN_CONFIG)
    record = TranscriptRecord(
        id=transcript_id,
        session_id=session_id,
        filename=filename,
        format=suffix,
        created_at=_now_iso(),
        flags=flags,
        corrections=[None] * len(flags),
        decisions=[ReviewDecision() for _ in flags],
    )
    storage.save_transcript(record)
    return record


def _flag_context(flag_id: str, flag: Flag) -> FlagContext:
    return FlagContext(
        id=flag_id,
        span=flag.span,
        sentence=flag.context or flag.span,
        candidates=list(flag.candidates),
        detector=flag.detector,
        reason=flag.reason,
        nearby=("", ""),
        related=[],
    )


def _same_text(a: str, b: str) -> bool:
    def normalize(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    return normalize(a) == normalize(b)


def run_correction(
    storage: Storage,
    record: TranscriptRecord,
    *,
    api_key: str,
    model: str | None = None,
    corrector: Corrector | None = None,
) -> TranscriptRecord:
    """Run the LLM `correct` pass over every Flag on ``record``.

    A no-op when ``record`` already has Corrections from a prior successful
    run: re-running `correct` on an already-corrected Transcript, and any
    diffing/versioning across multiple passes, is explicitly out of scope —
    only a *failed* run (no Corrections yet, ``correct_error`` set) is
    retriable.

    On failure, the record is persisted with ``correct_error`` set (a
    visible, retriable state) and the error is re-raised for the caller to
    report; on success, prior Corrections are replaced and any error is
    cleared. Review Decisions are left untouched either way.
    """
    if record.has_corrections:
        return record

    if not record.flags:
        record.corrections = []
        record.correct_error = None
        storage.save_transcript(record)
        return record

    model = model if model is not None else os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    try:
        active_corrector = corrector or OpenRouterCorrector(model, api_key=api_key)
        by_id: dict[str, Correction] = {}
        for start in range(0, len(record.flags), CHUNK_SIZE):
            batch = list(enumerate(record.flags))[start : start + CHUNK_SIZE]
            batch_ctx = [_flag_context(str(i), flag) for i, flag in batch]
            for result in active_corrector.correct(batch_ctx):
                by_id[result.id] = result
    except (MissingAPIKeyError, CorrectorError) as exc:
        record.correct_error = str(exc)
        storage.save_transcript(record)
        raise

    corrections: list[Correction | None] = []
    for i, flag in enumerate(record.flags):
        correction = by_id.get(str(i))
        if (
            correction is not None
            and correction.replacement is not None
            and _same_text(correction.replacement, flag.span)
        ):
            # Seen in practice: a model can propose a "correction" identical
            # to the original span. Nothing to act on, so don't treat it as
            # an actionable replacement.
            correction = Correction(
                id=correction.id,
                replacement=None,
                confidence=correction.confidence,
                rationale=correction.rationale,
            )
        corrections.append(correction)

    record.corrections = corrections
    record.correct_error = None
    storage.save_transcript(record)
    return record


def _default_replacement(flag: Flag, correction: Correction | None) -> str:
    """The text an Accept should default to, absent an explicit edit: the
    Flag's LLM Correction where one exists, else its own top local Candidate,
    else its unchanged span."""
    if correction is not None and correction.replacement:
        return correction.replacement
    if flag.candidates:
        return flag.candidates[0]
    return flag.span


def set_decision(record: TranscriptRecord, flag_id: int, *, action: str, text: str | None) -> None:
    """Record a reviewer's Accept/Reject on Flag ``flag_id``.

    Accepting always carries explicit replacement text: the reviewer's edit
    if given, else the Flag's LLM Correction, else its top local Candidate,
    else its unchanged span (accepting with no edit and no Correction or
    Candidate is a same-text no-op — legal, just inert on Export).
    """
    if not (0 <= flag_id < len(record.flags)):
        raise IndexError(f"No flag {flag_id} on transcript {record.id}")

    if action == "reject":
        record.decisions[flag_id] = ReviewDecision(status="rejected", text=None)
        return

    if action == "accept":
        if text is not None and text.strip():
            replacement = text
        else:
            correction = record.corrections[flag_id] if record.corrections else None
            replacement = _default_replacement(record.flags[flag_id], correction)
        record.decisions[flag_id] = ReviewDecision(status="accepted", text=replacement)
        return

    raise ValueError(f"Unknown review action {action!r} (expected 'accept' or 'reject')")


def export_transcript(storage: Storage, record: TranscriptRecord) -> str:
    """Apply every accepted Review Decision's text back into its Flag's Cue
    and serialize to the Transcript's original format. Flags left pending
    or rejected keep their Cue's original text."""
    cues = storage.load_cues(record.session_id, record.id)
    by_cue: dict[int, list[tuple[int, ReviewDecision]]] = {}
    for flag_id, (flag, decision) in enumerate(zip(record.flags, record.decisions)):
        if decision.status == "accepted" and decision.text is not None:
            by_cue.setdefault(flag.cue_index, []).append((flag_id, decision))

    for cue in cues:
        edits = by_cue.get(cue.index)
        if not edits:
            continue
        text = cue.text
        for flag_id, decision in edits:
            span = record.flags[flag_id].span
            text = text.replace(span, decision.text or "", 1)
        cue.text = text

    return serialize(cues, format=record.format)


@dataclass
class FlagRow:
    """One Flag shaped for the review page: its Correction (if any), the
    reviewer's current Review Decision, whether it's a not-an-error (or
    same-text no-op) verdict — never an actionable accept — and the text an
    Accept should default to."""

    id: int
    flag: Flag
    correction: Correction | None
    decision: ReviewDecision
    dismissed: bool
    default_text: str


def transcript_rows(record: TranscriptRecord) -> list[FlagRow]:
    rows = []
    for flag_id, flag in enumerate(record.flags):
        correction = record.corrections[flag_id] if record.corrections else None
        rows.append(
            FlagRow(
                id=flag_id,
                flag=flag,
                correction=correction,
                decision=record.decisions[flag_id],
                dismissed=correction is not None and correction.replacement is None,
                default_text=_default_replacement(flag, correction),
            )
        )
    return rows


@dataclass
class CorrectionSummary:
    confirmed: int
    dismissed: int
    total: int


def correction_summary(record: TranscriptRecord) -> CorrectionSummary | None:
    if not record.has_corrections:
        return None
    confirmed = sum(1 for c in record.corrections if c is not None and c.replacement is not None)
    dismissed = sum(1 for c in record.corrections if c is not None and c.replacement is None)
    total = sum(1 for c in record.corrections if c is not None)
    return CorrectionSummary(confirmed=confirmed, dismissed=dismissed, total=total)
