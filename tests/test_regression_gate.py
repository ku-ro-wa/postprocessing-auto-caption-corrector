"""Regression gate: runs the local detector pipeline against the Scored
corpus and reports recall, precision, and cold-flag rate as three separate
numbers (CONTEXT.md's Regression gate). Floors are hand-set constants,
updated deliberately when a tradeoff is knowingly accepted -- never
auto-derived from a historical baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

from caption_checker.detect import detect
from caption_checker.evaluation import ScoreReport, load_corpus, score
from caption_checker.models import DetectConfig
from caption_checker.parser import parse, tokenize

DATA_DIR = Path(__file__).parent / "data"
CORPUS_PATH = DATA_DIR / "scored_corpus.json"

# Set 2026-09-23 from the full manual audio pass over the 5 real videos.
# Recall is low on purpose: the corpus now holds every error heard, including
# the real-word errors (right spelling, wrong word) no detector catches yet.
# Precision is measured over should-not-flag spans that were mostly picked
# *because* they got flagged, so it's a regression floor, not a rate.
#
# Lowered recall / raised cold ceiling 2026-09-23, knowingly: phonetic_internal
# no longer offers a common word for a capitalized mid-sentence token. That
# dropped 6 false positives on real names (Chollet -> "should") but also 5
# misheard names (Navia, Nome, Kimmy, Quen, Viti) that were only "caught" via
# equally wrong candidates ("now", "name", "come") the bypass would have
# applied. Several oov merges lost those junk candidates and went cold.
MIN_RECALL = 0.44
MIN_PRECISION = 0.66
MAX_COLD_FLAG_RATE = 0.6

# The context-embedding tier (the `check`/`correct` default when installed)
# currently adds two false positives (e.g. "Kalshi") and no catches.
MIN_PRECISION_WITH_EMBEDDINGS = 0.61


def _run(config: DetectConfig) -> ScoreReport:
    cases = load_corpus(CORPUS_PATH)
    cues = {case.source: parse(DATA_DIR / case.source) for case in cases}
    flags_by_source = {
        source: detect(source_cues, config=config) for source, source_cues in cues.items()
    }
    words_by_source = {source: tokenize(source_cues) for source, source_cues in cues.items()}
    return score(cases, flags_by_source, words_by_source)


@pytest.mark.parametrize("with_embeddings", [False, True], ids=["lexical", "embeddings"])
def test_regression_gate_meets_floors(with_embeddings: bool) -> None:
    if with_embeddings:
        pytest.importorskip("sentence_transformers")
    report = _run(DetectConfig(enable_embeddings=with_embeddings))
    min_precision = MIN_PRECISION_WITH_EMBEDDINGS if with_embeddings else MIN_PRECISION
    assert report.recall >= MIN_RECALL, (
        f"recall {report.recall:.2f} below floor {MIN_RECALL} "
        f"({report.false_negatives} missed real error(s))"
    )
    assert report.precision >= min_precision, (
        f"precision {report.precision:.2f} below floor {min_precision} "
        f"({report.false_positives} known-good span(s) flagged)"
    )
    assert report.cold_flag_rate <= MAX_COLD_FLAG_RATE, (
        f"cold-flag rate {report.cold_flag_rate:.2f} above ceiling "
        f"{MAX_COLD_FLAG_RATE} ({report.cold_flags}/{report.total_flags} flags)"
    )
