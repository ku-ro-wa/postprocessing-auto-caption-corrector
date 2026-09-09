"""Orchestrator for the ``correct`` command.

:func:`run_correction` mirrors :func:`caption_checker.detect.detect`: it takes
parsed cues plus an injected reviewer and (except for an ``estimate_only`` dry
run) a :class:`~caption_checker.corrector.Corrector`, and returns the corrected
cues together with a per-flag outcome record -- the sidecar payload. ``cli.py``
stays a thin wrapper that builds the real collaborators and calls this.

Pipeline: detect -> internal-match bypass -> decision-cache lookup -> LLM pass
over the residue (chunked, with a one-shot retry) -> review -> character-offset
splice -> corrected file + sidecar (+ optional eval table).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TextIO

from caption_checker.apply import apply_corrections
from caption_checker.cache import CachedCorrection, DecisionCache
from caption_checker.corrector import (
    Correction,
    Corrector,
    CorrectorError,
    FlagContext,
    ensure_ids_match,
)
from caption_checker.detect import detect
from caption_checker.models import (
    DEFAULT_MODEL,
    DETECTOR_PHONETIC_INTERNAL,
    Cue,
    DetectConfig,
    Flag,
    flag_to_dict,
    format_timestamp,
)
from caption_checker.normalize import clean, is_wordlike
from caption_checker.parser import tokenize
from caption_checker.phonetics import codes, similar
from caption_checker.vocab import Vocab

CHUNK_SIZE = 25

# --- outcome vocabulary (CONTEXT.md / sidecar) ------------------------------
APPLIED = "applied"
REJECTED = "rejected"
NOT_AN_ERROR = "not-an-error"
BYPASSED = "bypassed"
CACHED = "cached"
SKIPPED_PARSE_FAILURE = "skipped-parse-failure"

# --- where a pending correction came from (its provenance) -----------------
SOURCE_BYPASS = "bypass"
SOURCE_CACHE = "cache"
SOURCE_LLM = "llm"
SOURCE_PARSE_FAILURE = "parse-failure"

# --- the reviewer's default for a pending correction ----------------------
PRESET_ACCEPT = "accept"
PRESET_SKIP = "skip"


def _preset_for(replacement: str | None) -> str:
    """A not-an-error / no-candidate verdict defaults to skip; anything with a
    concrete replacement defaults to accept."""
    return PRESET_SKIP if replacement is None else PRESET_ACCEPT


# Static, no-network prompt pricing (USD per token) for cost estimates. Unknown
# models fall back to a token count. Figures are order-of-magnitude only.
_MODEL_PROMPT_PRICE: dict[str, float] = {
    "google/gemini-2.0-flash-001": 1.0e-7,
    "google/gemini-flash-1.5": 7.5e-8,
    "anthropic/claude-3.5-haiku": 8.0e-7,
    "openai/gpt-4o-mini": 1.5e-7,
}


class MaxCallsExceededError(RuntimeError):
    def __init__(self, needed: int, limit: int) -> None:
        super().__init__(
            f"run needs {needed} LLM request(s), over the --max-calls limit "
            f"of {limit}"
        )
        self.needed = needed
        self.limit = limit


# --- data shapes -----------------------------------------------------------


@dataclass
class PendingCorrection:
    """A proposed correction awaiting review."""

    flag: Flag
    flag_id: str
    cue_text: str
    replacement: str | None
    source: str  # one of the SOURCE_* constants -- where it came from
    detector_confidence: float
    llm_confidence: float | None
    rationale: str
    preset: str  # PRESET_ACCEPT | PRESET_SKIP -- the reviewer's default

    @property
    def headline_confidence(self) -> float:
        """The number shown in review: the LLM's if it judged this one, else
        the detector's."""
        return (
            self.llm_confidence
            if self.llm_confidence is not None
            else self.detector_confidence
        )


@dataclass
class ReviewDecision:
    pending: PendingCorrection
    accepted: bool
    replacement: str | None


@dataclass
class OutcomeRecord:
    flag: Flag
    outcome: str
    correction: dict | None


@dataclass
class Estimate:
    flag_count: int
    residue_count: int
    chunk_count: int
    approx_tokens: int
    approx_cost_usd: float | None


@dataclass
class CorrectionResult:
    cues: list[Cue]
    outcomes: list[OutcomeRecord]
    corrector_calls: int
    chunk_count: int
    residue_count: int
    estimate: Estimate | None = None


@dataclass
class _Flagged:
    """A flag with its stable id and assembled LLM context."""

    id: str
    flag: Flag
    ctx: FlagContext = field(default=None)  # type: ignore[assignment]


# --- reviewers -----------------------------------------------------------


class Reviewer(Protocol):
    def review(
        self, pending: list[PendingCorrection]
    ) -> list[ReviewDecision]: ...


class ThresholdReviewer:
    """``--yes-above CONF``: accept every proposed correction at or above
    ``threshold``, skip the rest. Never interactive."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def review(
        self, pending: list[PendingCorrection]
    ) -> list[ReviewDecision]:
        out: list[ReviewDecision] = []
        for p in pending:
            accept = (
                p.replacement is not None
                and p.headline_confidence >= self.threshold
            )
            out.append(ReviewDecision(p, accept, p.replacement))
        return out


class InteractiveReviewer:
    """Walk the user through pending corrections one at a time.

    Keys: ``y`` accept, ``n`` skip, ``e`` edit the replacement then accept,
    ``a`` accept all remaining at or above this one's confidence, ``q`` stop
    and keep what's accepted so far. A pending correction ``preset`` to skip
    (a not-an-error verdict) defaults to ``n`` but can still be accepted --
    since there is no proposed replacement to accept, ``y`` (like ``e``) then
    asks for the correction text.
    """

    def __init__(
        self, stdin: TextIO | None = None, stderr: TextIO | None = None
    ) -> None:
        self._in = stdin if stdin is not None else sys.stdin
        self._err = stderr if stderr is not None else sys.stderr

    def review(
        self, pending: list[PendingCorrection]
    ) -> list[ReviewDecision]:
        decisions: list[ReviewDecision] = []
        auto_threshold: float | None = None
        for p in pending:
            if (
                auto_threshold is not None
                and p.replacement is not None
                and p.headline_confidence >= auto_threshold
            ):
                decisions.append(ReviewDecision(p, True, p.replacement))
                continue
            self._render(p)
            default = "y" if p.preset == PRESET_ACCEPT else "n"
            while True:
                line = self._in.readline()
                key = default if line == "" else (line.strip().lower() or default)
                if key == "y" and p.replacement is not None:
                    decisions.append(ReviewDecision(p, True, p.replacement))
                    break
                if key == "n":
                    decisions.append(ReviewDecision(p, False, p.replacement))
                    break
                if key in ("e", "y"):  # 'y' with nothing to accept -> ask
                    edited = self._prompt_replacement()
                    if edited:
                        decisions.append(ReviewDecision(p, True, edited))
                        break
                    continue
                if key == "a":
                    auto_threshold = p.headline_confidence
                    decisions.append(
                        ReviewDecision(
                            p, p.replacement is not None, p.replacement
                        )
                    )
                    break
                if key == "q":
                    return decisions
                self._err.write("  ? use y / n / e / a / q\n")
                self._err.flush()
        return decisions

    def _prompt_replacement(self) -> str:
        self._err.write("  replacement> ")
        self._err.flush()
        return self._in.readline().strip()

    def _render(self, p: PendingCorrection) -> None:
        f = p.flag
        marked = p.cue_text.replace(f.span, f"»{f.span}«", 1)
        lines = [
            f"[{format_timestamp(f.start)}] cue {f.cue_index}  ({p.source})",
            f"  {marked}",
        ]
        if p.replacement is None:
            lines.append("  suggestion: (not an error)")
        else:
            lines.append(
                f'  suggestion: "{p.replacement}"  '
                f"conf {p.headline_confidence:.2f}"
            )
        if p.rationale:
            lines.append(f"  {p.rationale}")
        keys = "[y/N/e/a/q]" if p.preset == PRESET_SKIP else "[Y/n/e/a/q]"
        lines.append(f"  {keys} ")
        self._err.write("\n".join(lines))
        self._err.flush()


# --- bypass --------------------------------------------------------------


def should_bypass(flag: Flag, config: DetectConfig) -> bool:
    """True when a flag is safe to apply with no LLM call: a *pure*
    ``phonetic_internal`` flag (a merged one still goes to the LLM) whose top
    candidate clearly wins -- it is the only one, or it beats the runner-up on
    Jaro-Winkler by at least ``config.bypass_jw_margin``."""
    if flag.detector != DETECTOR_PHONETIC_INTERNAL:
        return False
    if not flag.candidates:
        return False
    if len(flag.candidates) == 1:
        return True
    span = clean(flag.span)
    top = similar(span, clean(flag.candidates[0]))
    runner = similar(span, clean(flag.candidates[1]))
    return (top - runner) >= config.bypass_jw_margin


# --- flag context assembly --------------------------------------------------


def _build_flagged(
    flags: list[Flag], cues: list[Cue], words: list
) -> list[_Flagged]:
    ordered = sorted(
        flags, key=lambda f: (f.cue_index, min(f.global_indices))
    )
    cues_by_index = {c.index: c for c in cues}

    code_index: dict[str, set[str]] = {}
    for w in words:
        if not is_wordlike(w.text):
            continue
        cw = clean(w.text)
        if len(cw) < 3:
            continue
        for c in codes(w.text):
            code_index.setdefault(c, set()).add(cw)

    out: list[_Flagged] = []
    for i, flag in enumerate(ordered):
        fid = f"f{i}"
        own = {clean(flag.span)}
        related = sorted(
            {
                s
                for c in codes(flag.span)
                for s in code_index.get(c, set())
                if s not in own
            }
        )[:3]
        before = cues_by_index.get(flag.cue_index - 1)
        after = cues_by_index.get(flag.cue_index + 1)
        ctx = FlagContext(
            id=fid,
            span=flag.span,
            sentence=flag.context,
            candidates=list(flag.candidates),
            detector=flag.detector,
            reason=flag.reason,
            nearby=(
                before.text.replace("\n", " ") if before else "",
                after.text.replace("\n", " ") if after else "",
            ),
            related=related,
        )
        out.append(_Flagged(id=fid, flag=flag, ctx=ctx))
    return out


# --- orchestrator ---------------------------------------------------------


def run_correction(
    cues: list[Cue],
    *,
    reviewer: Reviewer,
    corrector: Corrector | None = None,
    vocab: Vocab | None = None,
    config: DetectConfig | None = None,
    model_id: str = DEFAULT_MODEL,
    cache: DecisionCache | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_calls: int | None = None,
    estimate_only: bool = False,
) -> CorrectionResult:
    config = config or DetectConfig()
    cache = cache or DecisionCache(None, enabled=False)
    words = tokenize(cues)
    cues_by_index = {c.index: c for c in cues}

    flagged = _build_flagged(
        detect(cues, vocab=vocab, config=config), cues, words
    )

    pending: list[PendingCorrection] = []
    residue: list[_Flagged] = []

    def _pending(
        item: _Flagged,
        *,
        replacement: str | None,
        source: str,
        llm_confidence: float | None,
        rationale: str,
        preset: str,
    ) -> PendingCorrection:
        return PendingCorrection(
            flag=item.flag,
            flag_id=item.id,
            cue_text=cues_by_index[item.flag.cue_index].text.replace(
                "\n", " "
            ),
            replacement=replacement,
            source=source,
            detector_confidence=item.flag.confidence,
            llm_confidence=llm_confidence,
            rationale=rationale,
            preset=preset,
        )

    for item in flagged:
        if should_bypass(item.flag, config):
            pending.append(
                _pending(
                    item,
                    replacement=item.flag.candidates[0],
                    source=SOURCE_BYPASS,
                    llm_confidence=None,
                    rationale=item.flag.reason,
                    preset=PRESET_ACCEPT,
                )
            )
            continue
        hit = cache.get(item.flag.span, model_id)
        if hit is not None:
            pending.append(
                _pending(
                    item,
                    replacement=hit.replacement,
                    source=SOURCE_CACHE,
                    llm_confidence=hit.confidence,
                    rationale=hit.rationale,
                    preset=_preset_for(hit.replacement),
                )
            )
            continue
        residue.append(item)

    chunks = [
        residue[i : i + chunk_size]
        for i in range(0, len(residue), chunk_size)
    ]

    if estimate_only:
        return CorrectionResult(
            cues=cues,
            outcomes=[],
            corrector_calls=0,
            chunk_count=len(chunks),
            residue_count=len(residue),
            estimate=_estimate(flagged, residue, chunks, model_id),
        )

    if max_calls is not None and len(chunks) > max_calls:
        raise MaxCallsExceededError(len(chunks), max_calls)
    if chunks and corrector is None:
        raise ValueError("run_correction needs a corrector for the residue")

    calls = 0
    for chunk in chunks:
        batch = [item.ctx for item in chunk]
        indexed: dict[str, Correction] | None = None
        # One retry on a malformed reply -- but never issue a request that
        # would push the run past --max-calls.
        for _attempt in (1, 2):
            if max_calls is not None and calls >= max_calls:
                break
            try:
                calls += 1
                assert corrector is not None
                indexed = _index_corrections(corrector.correct(batch), batch)
                break
            except CorrectorError:
                indexed = None
        if indexed is None:
            for item in chunk:
                pending.append(
                    _pending(
                        item,
                        replacement=None,
                        source=SOURCE_PARSE_FAILURE,
                        llm_confidence=None,
                        rationale="batch parse failure after one retry",
                        preset=PRESET_SKIP,
                    )
                )
            continue
        for item in chunk:
            c = indexed[item.id]
            pending.append(
                _pending(
                    item,
                    replacement=c.replacement,
                    source=SOURCE_LLM,
                    llm_confidence=c.confidence,
                    rationale=c.rationale,
                    preset=_preset_for(c.replacement),
                )
            )
            cache.set(
                item.flag.span,
                model_id,
                CachedCorrection(c.replacement, c.confidence, c.rationale),
            )
    cache.save()

    pending.sort(
        key=lambda p: (p.flag.cue_index, min(p.flag.global_indices))
    )
    decided = {
        id(d.pending): d for d in reviewer.review(pending)
    }

    accepted: list[tuple[Flag, str]] = []
    outcomes: list[OutcomeRecord] = []
    for p in pending:
        d = decided.get(id(p))
        is_accept = d is not None and d.accepted
        replacement = (
            d.replacement
            if (d is not None and d.replacement is not None)
            else p.replacement
        )
        applied = False
        if is_accept and replacement is not None:
            accepted.append((p.flag, replacement))
            applied = True
        outcomes.append(
            OutcomeRecord(
                flag=p.flag,
                outcome=_classify(p, applied),
                correction=_correction_dict(p, replacement, applied),
            )
        )

    corrected = apply_corrections(cues, accepted)
    return CorrectionResult(
        cues=corrected,
        outcomes=outcomes,
        corrector_calls=calls,
        chunk_count=len(chunks),
        residue_count=len(residue),
    )


def _index_corrections(
    corrections: list[Correction], batch: list[FlagContext]
) -> dict[str, Correction]:
    """Match the reply to the request one-to-one, or raise so the chunk is
    retried / skipped."""
    if len(corrections) != len(batch):
        raise CorrectorError(
            f"reply has {len(corrections)} corrections for {len(batch)} flags"
        )
    ensure_ids_match({c.id for c in corrections}, {fc.id for fc in batch})
    return {c.id: c for c in corrections}


def _classify(p: PendingCorrection, applied: bool) -> str:
    if p.source == SOURCE_PARSE_FAILURE:
        return SKIPPED_PARSE_FAILURE
    if applied:
        if p.source == SOURCE_BYPASS:
            return BYPASSED
        if p.source == SOURCE_CACHE:
            return CACHED
        return APPLIED
    if p.replacement is None and p.source in (SOURCE_LLM, SOURCE_CACHE):
        return NOT_AN_ERROR
    return REJECTED


def _correction_dict(
    p: PendingCorrection, replacement: str | None, applied: bool
) -> dict | None:
    if p.source == SOURCE_PARSE_FAILURE:
        return None
    return {
        "replacement": replacement if applied else p.replacement,
        "source": p.source,
        "detector_confidence": p.detector_confidence,
        "llm_confidence": p.llm_confidence,
        "rationale": p.rationale,
    }


# --- estimate -----------------------------------------------------------


def _estimate(
    flagged: list[_Flagged],
    residue: list[_Flagged],
    chunks: list,
    model_id: str,
) -> Estimate:
    tokens = 320  # rough system-prompt + framing overhead
    for item in residue:
        tokens += len(json.dumps(item.ctx.as_dict())) // 4
    price = _MODEL_PROMPT_PRICE.get(model_id)  # USD per token, or None
    cost = round(tokens * price, 6) if price is not None else None
    return Estimate(
        flag_count=len(flagged),
        residue_count=len(residue),
        chunk_count=len(chunks),
        approx_tokens=tokens,
        approx_cost_usd=cost,
    )


# --- output writers ----------------------------------------------------


def sidecar_path(output: Path) -> Path:
    return output.with_name(output.name + ".flags.json")


def write_sidecar(path: Path, outcomes: list[OutcomeRecord]) -> None:
    payload = []
    for o in outcomes:
        entry: dict = {"flag": flag_to_dict(o.flag), "outcome": o.outcome}
        if o.correction is not None:
            entry["correction"] = o.correction
        payload.append(entry)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_eval_table(path: Path, outcomes: list[OutcomeRecord]) -> None:
    rows = [
        "| timestamp | span | detector | suggestion | llm_conf | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        c = o.correction or {}
        suggestion = c.get("replacement") or ""
        conf = c.get("llm_confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else ""
        rows.append(
            f"| {format_timestamp(o.flag.start)} | {o.flag.span} | "
            f"{o.flag.detector} | {suggestion} | {conf_s} | |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
