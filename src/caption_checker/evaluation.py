"""Regression-gate and eval scoring: matches a system's Flags against a
hand-curated Scored corpus of should-flag / should-not-flag cases. See
CONTEXT.md's Evaluation section (Regression gate, Cold flag, Scored corpus,
Flag-level precision) for the vocabulary this module implements."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Collection, Literal

from caption_checker.detect import detect
from caption_checker.earnings21 import CAVEAT, DEFAULT_CACHE_DIR
from caption_checker.models import DETECTOR_OOV, Cue, DetectConfig, Flag, Word
from caption_checker.normalize import clean
from caption_checker.parser import parse, tokenize
from caption_checker.vocab import load_vocab

if TYPE_CHECKING:
    from caption_checker.corrector import Spend
    from caption_checker.readthrough import Reader

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
    recall is broken down by; ``entity`` marks a case on a named entity,
    tallied separately whatever its kind."""

    source: str
    span: str
    verdict: Verdict
    candidate: str | None = None
    context: str | None = None
    kind: str | None = None
    entity: bool = False


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
    #: (entity cases caught, entity cases); None when no case is an entity.
    entity_recall: tuple[int, int] | None = None
    #: kind -> (should-flag cases caught by a flag proposing the case's
    #: candidate, should-flag cases of that kind): correction quality, kept
    #: beside detection recall rather than blended into it.
    with_candidate_by_kind: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: Audio covered by the scored sources (last cue end, summed), for cost
    #: per audio hour; set by :func:`run_eval`.
    audio_seconds: float = 0.0


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
            entity=entry.get("entity", False),
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
    match_candidates: bool = True,
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
    else an untouched flag may just be an unlisted error.

    With ``match_candidates`` off, a case is caught by any flag touching it,
    whatever its candidates -- detection recall, which the eval command
    always scores (ADR 0006); the Regression gate keeps the stricter rule.
    ``with_candidate_by_kind`` tallies the stricter rule either way."""
    true_positives = false_negatives = 0
    true_negatives = false_positives = 0
    by_kind: dict[str, list[int]] = {}
    with_candidate: dict[str, list[int]] = {}
    entity = [0, 0]
    error_indices: dict[str, set[int]] = {}

    for case in cases:
        occurrences = _locate(case, words_by_source.get(case.source, []))
        hits = _overlapping(flags_by_source.get(case.source, []), occurrences)
        if case.verdict == "should-flag":
            error_indices.setdefault(case.source, set()).update(*occurrences)
            proposed = any(
                case.candidate is None
                or any(case.candidate.lower() in c.lower() for c in f.candidates)
                for f in hits
            )
            hit = proposed if match_candidates else bool(hits)
            if hit:
                true_positives += 1
            else:
                false_negatives += 1
            if case.kind is not None:
                tally = by_kind.setdefault(case.kind, [0, 0])
                tally[0] += hit
                tally[1] += 1
                strict = with_candidate.setdefault(case.kind, [0, 0])
                strict[0] += proposed
                strict[1] += 1
            if case.entity:
                entity[0] += hit
                entity[1] += 1
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
        entity_recall=(entity[0], entity[1]) if entity[1] else None,
        with_candidate_by_kind={
            k: (v[0], v[1]) for k, v in sorted(with_candidate.items())
        },
    )


#: A system under test: whatever turns one transcript's Cues, plus its
#: Priming terms (empty when the eval runs without them), into Flags -- the
#: local detector pipeline or the Read-through (ADR 0006).
System = Callable[[list[Cue], list[str]], list[Flag]]


@dataclass(frozen=True)
class NamedCorpus:
    """A corpus the eval command scores: its cases, the directory
    their ``source`` files live in, and which of those sources list their
    errors exhaustively (the only ones Flag-level precision is computed on).
    An exhaustive source with no cases is still run and its flags counted.

    An Auto-labelled corpus built into the cache also has a ``manifest_path``
    listing every source -- all exhaustive -- with its Priming terms;
    ``headline_kinds`` are the kinds its headline recall is computed over,
    and ``caveat`` is printed with its numbers."""

    cases_path: Path
    data_dir: Path
    exhaustive_sources: tuple[str, ...] = ()
    manifest_path: Path | None = None
    headline_kinds: tuple[str, ...] = ()
    caveat: str | None = None
    #: A Held-out set: scored only for a final comparison, never while
    #: tuning (the eval command demands ``--final``).
    held_out: bool = False


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


# 5 more Audited transcripts, audited after the local pipeline was frozen
# (issue #22): more varied subjects and speakers than the Scored corpus, and
# errors only -- should-not-flag cases would mean picking spans after seeing
# a system's Flags. A Held-out set for ADR 0006's verdict; its misses then
# motivated issue #27, which moved it to the Dev set.
_AUDITED_DEV = (
    "fixing-vocal-positions-kpop-groups.auto.srt",
    "makeup-brands-hate-project-pan.auto.srt",
    "overpriced-coffee-personality-trait.auto.srt",
    "why-ceos-make-so-much-money-now.auto.srt",
    "why-every-resto-has-the-same-desserts.auto.srt",
)
CORPORA["audited-dev"] = NamedCorpus(
    cases_path=_TEST_DATA / "audited_dev_corpus.json",
    data_dir=_TEST_DATA,
    exhaustive_sources=_AUDITED_DEV,
)


def _earnings21(split: str) -> NamedCorpus:
    return NamedCorpus(
        cases_path=DEFAULT_CACHE_DIR / split / "cases.json",
        data_dir=DEFAULT_CACHE_DIR / split,
        manifest_path=DEFAULT_CACHE_DIR / split / "manifest.json",
        headline_kinds=("non-word", "real-word"),
        caveat=CAVEAT,
        held_out=split == "heldout",
    )


# Auto-labelled corpora, built by `caption-checker build-earnings21`. The
# held-out split is scored only for the final comparison (ADR 0006): looking
# at it to motivate a change makes it a Dev set.
CORPORA["earnings21-dev"] = _earnings21("dev")
CORPORA["earnings21-heldout"] = _earnings21("heldout")


def _local_detect(cues: list[Cue], priming_terms: list[str]) -> list[Flag]:
    if not priming_terms:
        return detect(cues)
    config = DetectConfig()
    vocab = load_vocab(terms=priming_terms, algo=config.phonetic_algo)
    return detect(cues, vocab=vocab, config=config)


def _local_pipeline(model: str) -> System:
    return _local_detect


class ReadThroughSystem:
    """The Read-through as a system under test: the local pipeline's Flags
    go in as hints, and the Flags it claims are errors -- a verdict with a
    replacement -- come out. A not-an-error verdict is a dismissal, not a
    Flag; a chunk that failed twice contributes nothing and is counted in
    ``failed_chunks``. Spend accumulates across every transcript run."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader
        self.failed_chunks = 0

    @property
    def spend(self) -> Spend:
        return self.reader.spend

    def __call__(self, cues: list[Cue], priming_terms: list[str]) -> list[Flag]:
        from caption_checker.readthrough import read_through

        result = read_through(
            cues, _local_detect(cues, priming_terms), self.reader,
            priming_terms=priming_terms,
        )
        self.failed_chunks += result.failed_chunks
        return [
            item.flag
            for item in result.items
            if item.correction is not None and item.correction.replacement is not None
        ]


def _read_through(model: str) -> System:
    from caption_checker import readthrough

    return ReadThroughSystem(readthrough.build_reader(model))


#: Systems the eval command can score, by name, each built for a model slug
#: (which the local pipeline ignores). Factories, so a system that needs an
#: API key or a heavy import only pays for it when chosen.
SYSTEMS: dict[str, Callable[[str], System]] = {
    "local": _local_pipeline,
    "read-through": _read_through,
}


def run_eval(
    corpus: NamedCorpus, system: System, *, priming: bool = False
) -> ScoreReport:
    """Run ``system`` over every source ``corpus`` names -- in its cases,
    among its exhaustive sources, or in its manifest -- and score the Flags it
    returns. With ``priming`` each source's Priming terms are handed to the
    system; without, it gets none."""
    if not corpus.cases_path.exists():
        raise CorpusError(
            f"{corpus.cases_path} not found; for an Earnings-21 corpus run "
            "`caption-checker build-earnings21` first"
        )
    manifest: dict[str, dict] = {}
    if corpus.manifest_path is not None:
        manifest = json.loads(corpus.manifest_path.read_text(encoding="utf-8"))["sources"]
    if priming and not manifest:
        raise CorpusError("this corpus has no Priming terms to run with")
    cases = load_corpus(corpus.cases_path)
    exhaustive = (*corpus.exhaustive_sources, *manifest)
    sources = dict.fromkeys([*(c.source for c in cases), *exhaustive])
    flags_by_source: dict[str, list[Flag]] = {}
    words_by_source: dict[str, list[Word]] = {}
    audio_seconds = 0.0
    for source in sources:
        cues = parse(corpus.data_dir / source)
        terms = manifest.get(source, {}).get("priming_terms", []) if priming else []
        flags_by_source[source] = system(cues, list(terms))
        words_by_source[source] = tokenize(cues)
        audio_seconds += max((c.end.total_seconds() for c in cues), default=0.0)
    report = score(
        cases,
        flags_by_source,
        words_by_source,
        exhaustive_sources=exhaustive,
        # Detection, on every corpus (ADR 0006): a system that flags an error
        # but leaves the fix to a later pass still found it.
        match_candidates=False,
    )
    return replace(report, audio_seconds=audio_seconds)
