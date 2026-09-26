"""Shared helpers for detectors."""

from __future__ import annotations

from caption_checker.models import Cue, Flag, Word
from caption_checker.normalize import _SPAN_EDGE


def index_cues(cues: list[Cue]) -> dict[int, Cue]:
    return {cue.index: cue for cue in cues}


def span_text(span_words: list[Word]) -> str:
    """Surface text of a span with only the outer punctuation trimmed --
    interior punctuation ("fast. Hang") stays, as the splice keeps it."""
    raw = " ".join(w.text for w in span_words)
    return raw.strip(_SPAN_EDGE) or raw


def make_flag(
    span_words: list[Word],
    cues_by_index: dict[int, Cue],
    *,
    detector: str,
    reason: str,
    confidence: float,
    candidates: list[str] | None = None,
) -> Flag:
    """Build a Flag covering one or more consecutive Words. Every Word comes
    from a real cue, so ``cues_by_index`` is expected to hold both endpoints."""
    first, last = span_words[0], span_words[-1]
    start_cue = cues_by_index[first.cue_index]
    end_cue = cues_by_index[last.cue_index]
    return Flag(
        span=span_text(span_words),
        global_indices=[w.global_index for w in span_words],
        cue_index=first.cue_index,
        start=start_cue.start,
        end=end_cue.end,
        detector=detector,
        reason=reason,
        candidates=list(candidates or []),
        confidence=round(confidence, 3),
    )
