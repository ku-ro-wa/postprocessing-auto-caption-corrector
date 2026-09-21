"""Regression-gate scoring: matches the local detector pipeline's output
against a hand-curated Scored corpus of should-flag / should-not-flag cases.
See CONTEXT.md's Evaluation section (Regression gate, Cold flag, Scored
corpus) for the vocabulary this module implements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from caption_checker.models import DETECTOR_OOV, Flag

Verdict = Literal["should-flag", "should-not-flag"]


@dataclass(frozen=True)
class ScoredCase:
    """One Scored corpus entry: a span expected to be flagged (optionally
    with a candidate correction) or expected to be left alone."""

    source: str
    span: str
    verdict: Verdict
    candidate: str | None = None


@dataclass(frozen=True)
class ScoreReport:
    recall: float
    precision: float
    cold_flag_rate: float
    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int
    total_flags: int
    cold_flags: int


def is_cold_flag(flag: Flag) -> bool:
    """True when a merged flag includes ``oov`` and carries no candidates at
    all -- the system has zero vocabulary-matched context for the span.
    ``context_embedding`` also emits empty ``candidates``, but it isn't a
    vocabulary check, so its presence alone doesn't make a flag cold."""
    return DETECTOR_OOV in flag.detector.split("+") and not flag.candidates


def load_corpus(path: str | Path) -> list[ScoredCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ScoredCase(
            source=entry["source"],
            span=entry["span"],
            verdict=entry["verdict"],
            candidate=entry.get("candidate"),
        )
        for entry in raw
    ]


def _matches(flags: list[Flag], span: str) -> list[Flag]:
    span = span.lower()
    return [f for f in flags if span in f.span.lower()]


def score(
    cases: list[ScoredCase], flags_by_source: dict[str, list[Flag]]
) -> ScoreReport:
    """Score Scored corpus ``cases`` against ``detect()`` output already
    grouped by source file. Recall (should-flag cases caught) and precision
    (should-not-flag cases left alone) are each computed over their own case
    class, never blended into one score. Cold-flag rate is computed over
    every flag ``detect()`` produced across the referenced sources."""
    true_positives = false_negatives = 0
    true_negatives = false_positives = 0

    for case in cases:
        hits = _matches(flags_by_source.get(case.source, []), case.span)
        if case.verdict == "should-flag":
            hit = any(
                case.candidate is None
                or any(case.candidate.lower() in c.lower() for c in f.candidates)
                for f in hits
            )
            if hit:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if hits:
                false_positives += 1
            else:
                true_negatives += 1

    all_flags = [f for flags in flags_by_source.values() for f in flags]
    cold_flags = sum(1 for f in all_flags if is_cold_flag(f))

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 1.0
    )
    precision = (
        true_negatives / (true_negatives + false_positives)
        if (true_negatives + false_positives)
        else 1.0
    )
    cold_flag_rate = cold_flags / len(all_flags) if all_flags else 0.0

    return ScoreReport(
        recall=recall,
        precision=precision,
        cold_flag_rate=cold_flag_rate,
        true_positives=true_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        total_flags=len(all_flags),
        cold_flags=cold_flags,
    )
