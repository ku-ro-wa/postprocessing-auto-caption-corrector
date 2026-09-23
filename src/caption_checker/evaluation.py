"""Regression-gate scoring: matches the local detector pipeline's output
against a hand-curated Scored corpus of should-flag / should-not-flag cases.
See CONTEXT.md's Evaluation section (Regression gate, Cold flag, Scored
corpus) for the vocabulary this module implements."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from caption_checker.models import DETECTOR_OOV, Flag, Word
from caption_checker.normalize import clean

Verdict = Literal["should-flag", "should-not-flag"]


class CorpusError(ValueError):
    """A Scored corpus entry doesn't locate any Words in its transcript --
    almost always a typo in ``span`` or ``context``."""


@dataclass(frozen=True)
class ScoredCase:
    """One Scored corpus entry: a span expected to be flagged (optionally
    with a candidate correction) or expected to be left alone.

    ``span`` is matched as whole Words, not a substring, so "monitor" never
    matches "monitoring". Without ``context`` every occurrence counts; with
    it, only the occurrence of ``span`` inside that surrounding text does --
    needed when a real word is wrong in one place and right in others.
    ``kind`` is a free tag (``non-word``, ``real-word``, ``format``) that
    recall is broken down by."""

    source: str
    span: str
    verdict: Verdict
    candidate: str | None = None
    context: str | None = None
    kind: str | None = None


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
    #: kind -> (should-flag cases caught, should-flag cases of that kind)
    recall_by_kind: dict[str, tuple[int, int]] = field(default_factory=dict)


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
            context=entry.get("context"),
            kind=entry.get("kind"),
        )
        for entry in raw
    ]


def _find(tokens: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(tokens) - n + 1) if tokens[i : i + n] == needle]


def _locate(case: ScoredCase, words: list[Word]) -> list[set[int]]:
    """Global indices of each occurrence of ``case.span`` (restricted to the
    one inside ``case.context`` when given)."""
    tokens = [clean(w.text) for w in words]
    span = [clean(t) for t in case.span.split()]
    if case.context is None:
        starts = _find(tokens, span)
    else:
        context = [clean(t) for t in case.context.split()]
        inner = _find(context, span)
        starts = [c + inner[0] for c in _find(tokens, context)] if inner else []
    if not starts:
        where = f" within {case.context!r}" if case.context else ""
        raise CorpusError(f"{case.source}: {case.span!r}{where} not found")
    return [{words[i + k].global_index for k in range(len(span))} for i in starts]


def _overlapping(flags: list[Flag], occurrences: list[set[int]]) -> list[Flag]:
    return [
        f for f in flags if any(occ.intersection(f.global_indices) for occ in occurrences)
    ]


def score(
    cases: list[ScoredCase],
    flags_by_source: dict[str, list[Flag]],
    words_by_source: dict[str, list[Word]],
) -> ScoreReport:
    """Score Scored corpus ``cases`` against ``detect()`` output already
    grouped by source file, locating each case among that source's Words. A
    flag counts against a case when it covers any of the case's Words.
    Recall (should-flag cases caught) and precision (should-not-flag cases
    left alone) are each computed over their own case class, never blended
    into one score; ``recall_by_kind`` breaks recall down by ``kind`` tag. Cold-flag rate is computed over
    every flag ``detect()`` produced across the referenced sources."""
    true_positives = false_negatives = 0
    true_negatives = false_positives = 0
    by_kind: dict[str, list[int]] = {}

    for case in cases:
        occurrences = _locate(case, words_by_source.get(case.source, []))
        hits = _overlapping(flags_by_source.get(case.source, []), occurrences)
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
            if case.kind is not None:
                tally = by_kind.setdefault(case.kind, [0, 0])
                tally[0] += hit
                tally[1] += 1
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
        recall_by_kind={k: (v[0], v[1]) for k, v in sorted(by_kind.items())},
    )
