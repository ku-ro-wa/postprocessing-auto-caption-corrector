"""Regression gate: runs the local detector pipeline against the Scored
corpus and reports recall, precision, and cold-flag rate as three separate
numbers (CONTEXT.md's Regression gate). Floors are hand-set constants,
updated deliberately when a tradeoff is knowingly accepted -- never
auto-derived from a historical baseline."""

from __future__ import annotations

from pathlib import Path

from caption_checker.detect import detect
from caption_checker.evaluation import ScoreReport, load_corpus, score
from caption_checker.models import DetectConfig
from caption_checker.parser import parse

DATA_DIR = Path(__file__).parent / "data"
CORPUS_PATH = DATA_DIR / "scored_corpus.json"
CONFIG = DetectConfig(enable_embeddings=False)

MIN_RECALL = 1.0
MIN_PRECISION = 1.0
MAX_COLD_FLAG_RATE = 0.5


def _run() -> ScoreReport:
    cases = load_corpus(CORPUS_PATH)
    sources = {case.source for case in cases}
    flags_by_source = {
        source: detect(parse(DATA_DIR / source), config=CONFIG) for source in sources
    }
    return score(cases, flags_by_source)


def test_regression_gate_meets_floors() -> None:
    report = _run()
    assert report.recall >= MIN_RECALL, (
        f"recall {report.recall:.2f} below floor {MIN_RECALL} "
        f"({report.false_negatives} missed real error(s))"
    )
    assert report.precision >= MIN_PRECISION, (
        f"precision {report.precision:.2f} below floor {MIN_PRECISION} "
        f"({report.false_positives} known-good span(s) flagged)"
    )
    assert report.cold_flag_rate <= MAX_COLD_FLAG_RATE, (
        f"cold-flag rate {report.cold_flag_rate:.2f} above ceiling "
        f"{MAX_COLD_FLAG_RATE} ({report.cold_flags}/{report.total_flags} flags)"
    )
