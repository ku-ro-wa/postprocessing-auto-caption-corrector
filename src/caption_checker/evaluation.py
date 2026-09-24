"""Regression-gate and eval scoring: matches a system's Flags against a
hand-curated Scored corpus of should-flag / should-not-flag cases. See
CONTEXT.md's Evaluation section (Regression gate, Cold flag, Scored corpus,
Flag-level precision) for the vocabulary this module implements."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Collection, Literal

from caption_checker.detect import detect
from caption_checker.models import DETECTOR_OOV, Cue, Flag, Word
from caption_checker.normalize import clean
from caption_checker.parser import parse, tokenize

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
    #: Flag-level precision over the exhaustive sources only; None when they
    #: produced no flags (hand-made fixtures list some errors, not all of them).
    flag_precision: float | None = None
    flags_touching_errors: int = 0
    exhaustive_flags: int = 0


def is_cold_flag(flag: Flag) -> bool:
    """True when a merged flag includes ``oov`` and carries no candidates at
    all -- the system has zero vocabulary-matched context for the span."""
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
    exhaustive_sources: Collection[str] = (),
) -> ScoreReport:
    """Score Scored corpus ``cases`` against ``detect()`` output already
    grouped by source file, locating each case among that source's Words. A
    flag counts against a case when it covers any of the case's Words.
    Recall (should-flag cases caught) and precision (should-not-flag cases
    left alone) are each computed over their own case class, never blended
    into one score; ``recall_by_kind`` breaks recall down by ``kind`` tag. Cold-flag rate is computed over
    every flag ``detect()`` produced across the referenced sources.

    Flag-level precision -- flags touching any should-flag case, of any
    ``kind`` and whatever their candidates, over all flags emitted -- is
    computed only over ``exhaustive_sources``: transcripts whose errors are
    listed exhaustively (Audited transcripts, Auto-labelled corpora). Anywhere
    else an untouched flag may just be an unlisted error."""
    true_positives = false_negatives = 0
    true_negatives = false_positives = 0
    by_kind: dict[str, list[int]] = {}
    error_indices: dict[str, set[int]] = {}

    for case in cases:
        occurrences = _locate(case, words_by_source.get(case.source, []))
        hits = _overlapping(flags_by_source.get(case.source, []), occurrences)
        if case.verdict == "should-flag":
            error_indices.setdefault(case.source, set()).update(*occurrences)
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

    exhaustive_flags = flags_touching_errors = 0
    for source in exhaustive_sources:
        errors = error_indices.get(source, set())
        for flag in flags_by_source.get(source, []):
            exhaustive_flags += 1
            flags_touching_errors += bool(errors.intersection(flag.global_indices))
    # None rather than a vacuous 1.0 when nothing was flagged: a system that
    # emits no Flags hasn't earned perfect precision.
    flag_precision = (
        flags_touching_errors / exhaustive_flags if exhaustive_flags else None
    )

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
        flag_precision=flag_precision,
        flags_touching_errors=flags_touching_errors,
        exhaustive_flags=exhaustive_flags,
    )


#: A system under test: whatever turns one transcript's Cues into Flags --
#: the local detector pipeline today, the Read-through later (ADR 0006).
System = Callable[[list[Cue]], list[Flag]]


@dataclass(frozen=True)
class NamedCorpus:
    """A corpus the eval command scores: its cases, the directory
    their ``source`` files live in, and which of those sources list their
    errors exhaustively (the only ones Flag-level precision is computed on).
    An exhaustive source with no cases is still run and its flags counted."""

    cases_path: Path
    data_dir: Path
    exhaustive_sources: tuple[str, ...] = ()


_TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

CORPORA: dict[str, NamedCorpus] = {
    # Cases from the 5 Audited transcripts plus the hand-made fixtures' planted
    # cases, which count toward recall and case precision but not Flag-level
    # precision. Every number here is a Dev set number.
    "scored": NamedCorpus(
        cases_path=_TEST_DATA / "scored_corpus.json",
        data_dir=_TEST_DATA,
        exhaustive_sources=(
            "agi-are-we-there-yet.auto.vtt",
            "ai-researchers-pace-demand.auto.srt",
            "andrew-ng-ai-opportunities.auto.vtt",
            "prediction-markets-ads.auto.srt",
            "social-media-addictive.auto.srt",
        ),
    ),
}


def _local_pipeline() -> System:
    return detect


#: Systems the eval command can score, by name. Factories, so a system that
#: needs an API key or a heavy import only pays for it when chosen.
SYSTEMS: dict[str, Callable[[], System]] = {
    "local": _local_pipeline,
}


def run_eval(corpus: NamedCorpus, system: System) -> ScoreReport:
    """Run ``system`` over every source ``corpus`` names -- in its cases or
    among its exhaustive sources -- and score the Flags it returns."""
    cases = load_corpus(corpus.cases_path)
    sources = dict.fromkeys([*(c.source for c in cases), *corpus.exhaustive_sources])
    flags_by_source: dict[str, list[Flag]] = {}
    words_by_source: dict[str, list[Word]] = {}
    for source in sources:
        cues = parse(corpus.data_dir / source)
        flags_by_source[source] = system(cues)
        words_by_source[source] = tokenize(cues)
    return score(
        cases,
        flags_by_source,
        words_by_source,
        exhaustive_sources=corpus.exhaustive_sources,
    )
