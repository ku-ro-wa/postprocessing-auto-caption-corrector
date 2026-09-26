"""Orchestration for the web layer: upload+scan, the LLM correct pass,
recording Review Decisions, and assembling an Export. Calls the same core
entry points the CLI uses (``parser.parse``/``serialize``, ``detect.detect``,
and ``readthrough.read_through`` over its ``Reader`` seam) — no new
core-pipeline logic lives here.

The web `correct` pass is the Read-through (ADR 0006). Unlike the CLI's own
``correct`` command (``correct.py``, with its interactive/threshold
``Reviewer`` for a single synchronous run), the web layer's job is to get
the Read-through's verdicts and then hold each Flag's Review Decision open
across many separate HTTP requests — so only ``read_through`` itself is
reused here, not ``correct.py``'s CLI-specific orchestration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from caption_checker.apply import apply_corrections, cues_spanned
from caption_checker.corrector import Correction, CorrectorError, MissingAPIKeyError
from caption_checker.detect import detect
from caption_checker.models import DEFAULT_MODEL, DetectConfig, Flag
from caption_checker.parser import serialize, tokenize
from caption_checker.readthrough import (
    DETECTOR_READ_THROUGH,
    OpenRouterReader,
    Reader,
    read_through,
    v4,
)
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


def _same_text(a: str, b: str) -> bool:
    def normalize(s: str) -> str:
        return "".join(ch.lower() for ch in s if ch.isalnum())

    return normalize(a) == normalize(b)


def _without_echo(flag: Flag, correction: Correction | None) -> Correction | None:
    """``correction``, downgraded to not-an-error when it only repeats the
    Flag's own span. Seen in practice: a model can propose a "correction"
    identical to the original span. Nothing to act on, so don't treat it as
    an actionable replacement."""
    if (
        correction is not None
        and correction.replacement is not None
        and _same_text(correction.replacement, flag.span)
    ):
        return replace(correction, replacement=None)
    return correction


def run_correction(
    storage: Storage,
    record: TranscriptRecord,
    *,
    api_key: str,
    model: str | None = None,
    reader: Reader | None = None,
    priming_terms: Sequence[str] = (),
) -> TranscriptRecord:
    """Run the Read-through over ``record``, its Flags as hints and
    ``priming_terms`` given to the model.

    Each hint's verdict becomes that Flag's Correction (a verdict that widens
    a hint replaces the Flag's span in place); each error the Read-through
    found by itself is appended as a new Flag with a pending Review Decision.
    A chunk that fails leaves its Flags unjudged and is counted in
    ``failed_chunks``.

    A no-op when ``record`` is already corrected: re-running `correct`, and
    any diffing/versioning across multiple passes, is explicitly out of
    scope — only a *failed* run (every chunk failed, or no API key;
    ``correct_error`` set) is retriable; the chunks of a partly failed run
    are not. On failure the record is persisted
    with ``correct_error`` set and the error re-raised for the caller to
    report.
    """
    if record.corrected:
        return record

    model = model if model is not None else os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    cues = storage.load_cues(record.session_id, record.id)
    try:
        active_reader = reader or OpenRouterReader(v4(model), api_key=api_key)
        result = read_through(cues, record.flags, active_reader, priming_terms=priming_terms)
        if result.chunk_count and result.failed_chunks == result.chunk_count:
            raise CorrectorError(
                f"The Read-through failed on all {result.chunk_count} chunk(s); "
                "nothing was judged. Try again."
            )
    except (MissingAPIKeyError, CorrectorError) as exc:
        record.correct_error = str(exc)
        storage.save_transcript(record)
        raise

    index_of = {id(flag): i for i, flag in enumerate(record.flags)}
    corrections: list[Correction | None] = [None] * len(record.flags)
    for item in result.items:
        correction = _without_echo(item.flag, item.correction)
        if item.hint is None:
            record.flags.append(item.flag)
            corrections.append(correction)
            record.decisions.append(ReviewDecision())
            continue
        i = index_of[id(item.hint)]
        if item.flag is not item.hint:
            # Widened: an accepted text was written for the narrower span.
            record.flags[i] = item.flag
            if record.decisions[i].status == "accepted":
                record.decisions[i] = ReviewDecision()
        corrections[i] = correction

    record.corrections = corrections
    record.correct_error = None
    record.corrected_at = _now_iso()
    record.chunk_count = result.chunk_count
    record.failed_chunks = result.failed_chunks
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
    """Splice every accepted Review Decision's text into the Flag's span
    (``apply_corrections``, ADR 0001 -- a span across Cues included) and
    serialize to the Transcript's original format. Flags left pending or
    rejected keep their original text."""
    cues = storage.load_cues(record.session_id, record.id)
    accepted = [
        (flag, decision.text)
        for flag, decision in zip(record.flags, record.decisions)
        if decision.status == "accepted" and decision.text is not None
    ]
    return serialize(apply_corrections(cues, accepted), format=record.format)


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
    #: "cue 7", or "cues 7–8" for a span across a Cue boundary.
    cue_label: str


def transcript_rows(storage: Storage, record: TranscriptRecord) -> list[FlagRow]:
    """One row per Flag, in transcript order — the Read-through's own finds
    are appended to ``record.flags`` but listed where they occur."""
    cues = storage.load_cues(record.session_id, record.id)
    words_by_gi = {w.global_index: w for w in tokenize(cues)}
    rows = []
    for flag_id, flag in enumerate(record.flags):
        correction = record.corrections[flag_id] if record.corrections else None
        spanned = cues_spanned(flag, cues, words_by_gi)
        rows.append(
            FlagRow(
                id=flag_id,
                flag=flag,
                correction=correction,
                decision=record.decisions[flag_id],
                dismissed=correction is not None and correction.replacement is None,
                default_text=_default_replacement(flag, correction),
                cue_label=(
                    f"cues {spanned[0].index}–{spanned[-1].index}"
                    if len(spanned) > 1
                    else f"cue {flag.cue_index}"
                ),
            )
        )
    rows.sort(key=lambda r: (r.flag.cue_index, min(r.flag.global_indices)))
    return rows


@dataclass
class CorrectionSummary:
    confirmed: int
    dismissed: int
    total: int
    found: int


def correction_summary(record: TranscriptRecord) -> CorrectionSummary | None:
    if not record.corrected:
        return None
    confirmed = sum(1 for c in record.corrections if c is not None and c.replacement is not None)
    dismissed = sum(1 for c in record.corrections if c is not None and c.replacement is None)
    total = sum(1 for c in record.corrections if c is not None)
    found = sum(1 for f in record.flags if f.detector == DETECTOR_READ_THROUGH)
    return CorrectionSummary(confirmed=confirmed, dismissed=dismissed, total=total, found=found)
