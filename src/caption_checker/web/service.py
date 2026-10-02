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
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

from caption_checker.apply import apply_corrections, cues_spanned, splice
from caption_checker.correct import prompt_price, read_through_estimate
from caption_checker.corrector import Correction, CorrectorError, MissingAPIKeyError
from caption_checker.detect import detect
from caption_checker.detectors.base import make_flag
from caption_checker.models import DETECTOR_REVIEWER, DetectConfig, Flag
from caption_checker.parser import serialize, tokenize
from caption_checker.readthrough import (
    DETECTOR_READ_THROUGH,
    ConfigError,
    OpenRouterReader,
    Reader,
    ReadThroughConfig,
    plan_chunks,
    read_through,
    select_config,
)
from caption_checker.vocab import Vocab, load_vocab
from caption_checker.web import cue_edit
from caption_checker.web.free_tier import FreeTier
from caption_checker.web.models import ReviewDecision, TranscriptRecord
from caption_checker.web.source_video import (
    InvalidVideoLinkError,
    MetadataLookup,
    lookup_metadata,
    parse_video_id,
)
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
    storage: Storage,
    session_id: str,
    filename: str,
    content: bytes,
    *,
    video_link: str = "",
    video_lookup: MetadataLookup = lookup_metadata,
) -> TranscriptRecord:
    """Validate, persist, and automatically scan an uploaded file, linking
    its Source video when ``video_link`` is given (its metadata from
    ``video_lookup``).

    Raises ``InvalidTranscriptError`` when the upload isn't a parseable
    SRT/VTT, or ``InvalidVideoLinkError`` when ``video_link`` isn't a
    YouTube link — either before any Transcript record is created.
    """
    video_id = parse_video_id(video_link) if video_link.strip() else None
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
    if video_id is not None:
        _link_video(record, video_id, video_lookup)
    storage.save_transcript(record)
    return record


def _link_video(record: TranscriptRecord, video_id: str, lookup: MetadataLookup) -> None:
    """Set the Source video and whatever ``lookup`` finds about it; a failed
    lookup leaves no metadata, never stale metadata from a previous video."""
    metadata = lookup(video_id)
    record.video_id = video_id
    record.video_title = metadata.title if metadata else None
    record.video_channel = metadata.channel if metadata else None


def set_source_video(
    storage: Storage,
    record: TranscriptRecord,
    video_link: str,
    *,
    video_lookup: MetadataLookup = lookup_metadata,
) -> None:
    """Link or change the Transcript's Source video, looking its metadata up
    again. Raises ``InvalidVideoLinkError`` without saving (or looking
    anything up) when the link doesn't parse."""
    _link_video(record, parse_video_id(video_link), video_lookup)
    storage.save_transcript(record)


def clear_source_video(storage: Storage, record: TranscriptRecord) -> None:
    record.video_id = None
    record.video_title = None
    record.video_channel = None
    storage.save_transcript(record)


def suggested_priming_terms(record: TranscriptRecord) -> str:
    """What the review page prefills an empty Priming terms field with: the
    Source video's ``<title>, <channel>``, for the reviewer to trim into
    terms. Empty when there's no metadata, or when the reviewer has already
    submitted terms of their own."""
    if record.priming_terms:
        return ""
    return ", ".join(t for t in (record.video_title, record.video_channel) if t)


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


def config_from_env(environ: Mapping[str, str] = os.environ) -> ReadThroughConfig:
    """The Read-through configuration the server runs: ``OPENROUTER_CONFIG``
    names a registered one, ``OPENROUTER_MODEL`` means prompt v4 with that
    model, neither means ``DEFAULT_CONFIG``. Both set, or an unknown name,
    raises :class:`ConfigError` rather than falling back."""
    name = environ.get("OPENROUTER_CONFIG") or None
    model = environ.get("OPENROUTER_MODEL") or None
    if name is not None and model is not None:
        raise ConfigError(
            "set OPENROUTER_CONFIG or OPENROUTER_MODEL, not both: a "
            "configuration names its own model"
        )
    try:
        return select_config(name, model)
    except ConfigError as exc:
        raise ConfigError(f"OPENROUTER_CONFIG: {exc}") from exc


def _hints(record: TranscriptRecord) -> list[Flag]:
    """The Flags the Read-through takes as hints: all but the reviewer's."""
    return [f for f in record.flags if f.detector != DETECTOR_REVIEWER]


def run_correction(
    storage: Storage,
    record: TranscriptRecord,
    *,
    api_key: str,
    config: ReadThroughConfig | None = None,
    reader: Reader | None = None,
    priming_terms: Sequence[str] = (),
) -> TranscriptRecord:
    """Run the Read-through over ``record``, its Flags as hints and
    ``priming_terms`` given to the model. ``reader`` (tests) or else
    ``config`` (default: :func:`config_from_env`) says what reads it.

    Each hint's verdict becomes that Flag's Correction (a verdict that widens
    a hint replaces the Flag's span in place); each error the Read-through
    found by itself is appended as a new Flag with a pending Review Decision.
    Flags the reviewer raised are not hints, so the model never judges or
    widens them, and a Flag it finds or widens onto one is dropped (ADR 0009).
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

    cues = storage.load_cues(record.session_id, record.id)
    try:
        active_reader = reader or OpenRouterReader(
            config or config_from_env(), api_key=api_key
        )
        result = read_through(cues, _hints(record), active_reader, priming_terms=priming_terms)
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
    reviewer_words = {
        gi
        for f in record.flags
        if f.detector == DETECTOR_REVIEWER
        for gi in f.global_indices
    }
    for item in result.items:
        if reviewer_words.intersection(item.flag.global_indices):
            if item.hint is None:
                continue
            # Widened onto it: the verdict was for the wider span, so the hint
            # keeps its own span and is left unjudged.
            item = replace(item, flag=item.hint, correction=None)
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


def transcript_word_count(storage: Storage, record: TranscriptRecord) -> int:
    """The Transcript's Words: what a Free tier run charges the Allowance,
    since the Read-through reads every one."""
    return len(tokenize(storage.load_cues(record.session_id, record.id)))


def free_tier_estimate_usd(
    storage: Storage,
    record: TranscriptRecord,
    config: ReadThroughConfig,
    priming_terms: Sequence[str] = (),
) -> float:
    """The pre-run cost estimate ``correct --estimate`` would print for a
    run of ``config`` over ``record``."""
    check_free_tier_config(config)
    cues = storage.load_cues(record.session_id, record.id)
    hints = _hints(record)
    chunks = plan_chunks(cues, hints, priming_terms=priming_terms)
    estimate = read_through_estimate(
        chunks, len(hints), model_id=config.model_id, messages=config.build_messages
    )
    assert estimate.approx_cost_usd is not None
    return estimate.approx_cost_usd


def check_free_tier_config(config: ReadThroughConfig) -> None:
    """Raise :class:`ConfigError` when ``config``'s model has no known price:
    the Free tier checks the Daily budget against a pre-run estimate, so it
    can't run a model it can't price."""
    if prompt_price(config.model_id) is None:
        raise ConfigError(
            f"the Free tier needs a cost estimate, and {config.model_id!r} has "
            "no known price: run a configuration whose model has one, or turn "
            "the limits off (serve --no-limits) for local use"
        )


def run_on_free_tier(
    storage: Storage,
    record: TranscriptRecord,
    free_tier: FreeTier,
    *,
    api_key: str,
    config: ReadThroughConfig,
    reader: Reader | None = None,
    priming_terms: Sequence[str] = (),
) -> TranscriptRecord:
    """:func:`run_correction` paid by the server's key (ADR 0008): checked
    against the Session's Allowance and the Daily budget before it starts --
    raising :class:`LimitReached` without calling the model -- then charged
    what OpenRouter reported (the estimate, if any reply carried no cost) and,
    if it produced a result, the Transcript's words. A started run always
    finishes."""
    if record.corrected:
        return record
    estimate = free_tier_estimate_usd(storage, record, config, priming_terms)
    reservation = free_tier.reserve(
        record.session_id, transcript_word_count(storage, record), estimate_usd=estimate
    )
    active_reader: Reader | None = None
    produced = False
    try:
        active_reader = reader or OpenRouterReader(config, api_key=api_key)
        run_correction(
            storage, record, api_key=api_key, reader=active_reader, priming_terms=priming_terms
        )
        produced = True
    finally:
        cost = active_reader.spend.cost_usd if active_reader is not None else 0.0
        free_tier.settle(
            reservation, produced=produced, cost_usd=estimate if cost is None else cost
        )
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
    return serialize(apply_corrections(cues, _accepted(record)), format=record.format)


def _accepted(record: TranscriptRecord) -> list[tuple[Flag, str]]:
    """What Export writes: each accepted Flag with its Review Decision's text."""
    return [
        (flag, decision.text)
        for flag, decision in zip(record.flags, record.decisions)
        if decision.status == "accepted" and decision.text is not None
    ]


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
    default_text: str
    #: "cue 7", or "cues 7–8" for a span across a Cue boundary.
    cue_label: str

    @property
    def by_reviewer(self) -> bool:
        return self.flag.detector == DETECTOR_REVIEWER

    @property
    def status(self) -> str:
        return _status(self.correction, self.decision)

    @property
    def dismissed(self) -> bool:
        return self.status == "dismissed"


def _status(correction: Correction | None, decision: ReviewDecision) -> str:
    """How the review page shows a Flag: its Review Decision's status, or
    "dismissed" for a not-an-error verdict (which has no Accept)."""
    if correction is not None and correction.replacement is None:
        return "dismissed"
    return decision.status


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
class CuePiece:
    """A stretch of a Cue's text as it would export: a Flag's span (its
    accepted text, if accepted) with that Flag's id and status, or the text
    between spans (``flag_id`` ``None``)."""

    text: str
    flag_id: int | None = None
    status: str | None = None


@dataclass
class CueRow:
    """One Cue shaped for the All Cues view, its text as Export would write
    it. ``merged_into`` is the Cue an accepted cross-Cue span moved all this
    Cue's text into -- Export removes such a Cue; the view keeps it."""

    index: int
    start: timedelta
    end: timedelta
    pieces: list[CuePiece]
    merged_into: int | None

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces)

    @property
    def spans(self) -> list[CuePiece]:
        return [p for p in self.pieces if p.flag_id is not None]


def cue_rows(
    storage: Storage, record: TranscriptRecord, cue_indices: Sequence[int] | None = None
) -> list[CueRow]:
    """Every Cue in order -- or those in ``cue_indices`` -- through the
    same splice as Export (``splice``), each Flag's span marked."""
    cues = storage.load_cues(record.session_id, record.id)
    accepted = _accepted(record)
    accepted_ids = {id(flag) for flag, _ in accepted}
    marked = [f for f in record.flags if id(f) not in accepted_ids]
    flag_ids = {id(flag): i for i, flag in enumerate(record.flags)}

    def piece(text: str, flag: Flag | None) -> CuePiece:
        if flag is None:
            return CuePiece(text)
        i = flag_ids[id(flag)]
        correction = record.corrections[i] if record.corrections else None
        return CuePiece(text, i, _status(correction, record.decisions[i]))

    return [
        CueRow(
            index=s.cue.index,
            start=s.cue.start,
            end=s.cue.end,
            pieces=[piece(text, flag) for text, flag in s.pieces],
            merged_into=s.merged_into,
        )
        for s in splice(cues, accepted, marked)
        if cue_indices is None or s.cue.index in cue_indices
    ]


def cues_affected(storage: Storage, record: TranscriptRecord, flag_id: int) -> list[int]:
    """The indices of the Cues a Review Decision on ``flag_id`` can change."""
    cues = storage.load_cues(record.session_id, record.id)
    words_by_gi = {w.global_index: w for w in tokenize(cues)}
    return [c.index for c in cues_spanned(record.flags[flag_id], cues, words_by_gi)]


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


class CueEditError(ValueError):
    """Raised when a Cue's edit can't be recorded at all."""


@dataclass
class CueEditResult:
    """What :func:`edit_cue` did: the Flags it recorded or updated (ids), and
    each change it could not save."""

    flag_ids: list[int]
    unsaved: list[str]

    @property
    def saved(self) -> bool:
        return bool(self.flag_ids)


def edit_cue(
    storage: Storage, record: TranscriptRecord, cue_index: int, new_text: str
) -> CueEditResult:
    """Record the reviewer's edit of Cue ``cue_index`` (ADR 0009).

    ``new_text`` is diffed at Word level against the Cue as Export would
    write it, and each changed stretch is written into ``record`` as a
    Flag's accepted Review Decision: an existing Flag's when the change sits
    inside its span, otherwise a new Flag raised by the reviewer (which
    supersedes -- rejects -- any Flag it partly overlaps). Nothing is changed
    unless the whole edit is recorded consistently: Export must then write
    ``new_text``'s Words in this Cue and every other Cue as before.
    """
    cues = storage.load_cues(record.session_id, record.id)
    accepted = _accepted(record)
    accepted_ids = {id(flag) for flag, _ in accepted}
    marked = [f for f in record.flags if id(f) not in accepted_ids]
    before = splice(cues, accepted, marked)
    spliced = next((s for s in before if s.cue.index == cue_index), None)
    if spliced is None:
        raise IndexError(f"No cue {cue_index} on transcript {record.id}")
    if spliced.merged_into is not None:
        raise CueEditError(
            f"Cue {cue_index} was merged into Cue {spliced.merged_into} by an accepted fix; "
            "edit that Cue, or reject the fix."
        )
    new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")
    if not new_text.split():
        raise CueEditError("A Cue can't be left empty.")

    flag_ids = {id(flag): i for i, flag in enumerate(record.flags)}
    corrections = list(record.corrections) + [None] * (len(record.flags) - len(record.corrections))
    words_by_gi = {w.global_index: w for w in tokenize(cues)}

    def live(flag: Flag) -> bool:
        i = flag_ids[id(flag)]
        return _status(corrections[i], record.decisions[i]) != "dismissed"

    def crosses_cues(flag: Flag) -> bool:
        return words_by_gi[flag.global_indices[-1]].cue_index != flag.cue_index

    plan = cue_edit.plan_edit(spliced, new_text, live=live, crosses_cues=crosses_cues)

    flags = list(record.flags)
    decisions = [replace(d) for d in record.decisions]
    touched: list[int] = []
    cues_by_index = {c.index: c for c in cues}
    cue_words = [w for w in words_by_gi.values() if w.cue_index == cue_index]
    unsaved = list(plan.unsaved)
    saved_regions = []
    for region in plan.regions:
        if region.inside is not None:
            i = flag_ids[id(region.inside)]
            decisions[i] = ReviewDecision(status="accepted", text=region.replacement)
            if region.inside.detector == DETECTOR_REVIEWER:
                flags[i] = replace(region.inside, candidates=[region.replacement])
            touched.append(i)
            saved_regions.append(region)
            continue
        start, end = region.origin
        span_words = sorted(
            (w for w in cue_words if w.char_offset < end and w.char_offset + len(w.text) > start),
            key=lambda w: w.global_index,
        )
        if not span_words:
            unsaved.append(f"Not saved: {region.replacement!r}: it covers no Word of the Cue.")
            continue
        for old in region.flags:
            decisions[flag_ids[id(old)]] = ReviewDecision(status="rejected", text=None)
        flag = make_flag(
            span_words,
            cues_by_index,
            detector=DETECTOR_REVIEWER,
            reason="Reviewer edit",
            confidence=1.0,
            candidates=[region.replacement],
        )
        flag.context = spliced.cue.text.replace("\n", " ")
        flags.append(flag)
        corrections.append(None)
        decisions.append(ReviewDecision(status="accepted", text=region.replacement))
        touched.append(len(flags) - 1)
        saved_regions.append(region)

    if not touched:
        return CueEditResult([], unsaved)

    tentative = replace(record, flags=flags, corrections=corrections, decisions=decisions)
    after = {s.cue.index: s.text for s in splice(cues, _accepted(tentative))}
    expected = cue_edit.apply_regions(spliced.text, saved_regions)
    unchanged_elsewhere = all(
        after[s.cue.index] == s.text for s in before if s.cue.index != cue_index
    )
    if after[cue_index].split() != expected.split() or not unchanged_elsewhere:
        raise CueEditError(
            "That edit couldn't be recorded without changing other text, so nothing was saved."
        )

    record.flags, record.corrections, record.decisions = flags, corrections, decisions
    return CueEditResult(touched, unsaved)
