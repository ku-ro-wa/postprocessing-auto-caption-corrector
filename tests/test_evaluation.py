from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from caption_checker.evaluation import (
    CorpusError,
    ScoredCase,
    is_cold_flag,
    load_corpus,
    score,
)
from caption_checker.models import Flag, Word

# "a.srt" as a token stream: the planted error sits at indices 1-2.
TEXT = "the cough ka raft consensus"
WORDS = {
    "a.srt": [
        Word(text=t, cue_index=1, char_offset=0, global_index=i)
        for i, t in enumerate(TEXT.split())
    ]
}


def _flag(
    detector: str,
    candidates: list[str] | None = None,
    span: str = "cubernetes",
    indices: list[int] | None = None,
) -> Flag:
    return Flag(
        span=span,
        global_indices=indices or [0],
        cue_index=1,
        start=timedelta(seconds=0),
        end=timedelta(seconds=1),
        detector=detector,
        reason="test",
        candidates=candidates or [],
    )


# --- is_cold_flag -------------------------------------------------------------


def test_bare_oov_flag_with_no_candidates_is_cold() -> None:
    assert is_cold_flag(_flag("oov"))


def test_oov_merged_with_phonetic_vocab_is_not_cold() -> None:
    assert not is_cold_flag(_flag("oov+phonetic_vocab", ["Kubernetes"]))


# --- load_corpus ----------------------------------------------------------


def test_load_corpus_parses_should_flag_and_should_not_flag_entries(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "source": "sample.srt",
                    "span": "cough ka",
                    "verdict": "should-flag",
                    "candidate": "Kafka",
                },
                {
                    "source": "sample.srt",
                    "span": "raft",
                    "verdict": "should-not-flag",
                },
            ]
        ),
        encoding="utf-8",
    )
    cases = load_corpus(corpus_path)
    assert cases == [
        ScoredCase("sample.srt", "cough ka", "should-flag", "Kafka"),
        ScoredCase("sample.srt", "raft", "should-not-flag", None),
    ]


# --- score ------------------------------------------------------------------


def test_score_counts_true_positive_when_span_and_candidate_match() -> None:
    cases = [ScoredCase("a.srt", "cough ka", "should-flag", "Kafka")]
    flags = {"a.srt": [_flag("split_word", ["Kafka"], span="cough ka", indices=[1, 2])]}
    report = score(cases, flags, WORDS)
    assert report.true_positives == 1
    assert report.false_negatives == 0
    assert report.recall == 1.0


def test_score_counts_false_negative_when_expected_span_missing() -> None:
    cases = [ScoredCase("a.srt", "cough ka", "should-flag", "Kafka")]
    report = score(cases, {"a.srt": []}, WORDS)
    assert report.false_negatives == 1
    assert report.recall == 0.0


def test_score_counts_false_negative_when_candidate_wrong() -> None:
    cases = [ScoredCase("a.srt", "cough ka", "should-flag", "Kafka")]
    flags = {"a.srt": [_flag("split_word", ["Kubernetes"], span="cough ka", indices=[1, 2])]}
    report = score(cases, flags, WORDS)
    assert report.false_negatives == 1
    assert report.true_positives == 0


def test_score_counts_true_negative_when_known_good_span_untouched() -> None:
    cases = [ScoredCase("a.srt", "raft", "should-not-flag")]
    report = score(cases, {"a.srt": []}, WORDS)
    assert report.true_negatives == 1
    assert report.precision == 1.0


def test_score_counts_false_positive_when_known_good_span_flagged() -> None:
    cases = [ScoredCase("a.srt", "raft", "should-not-flag")]
    flags = {"a.srt": [_flag("oov", span="raft", indices=[3])]}
    report = score(cases, flags, WORDS)
    assert report.false_positives == 1
    assert report.precision == 0.0


def test_score_reports_cold_flag_rate_over_all_produced_flags() -> None:
    cases = [ScoredCase("a.srt", "cough ka", "should-flag", "Kafka")]
    flags = {
        "a.srt": [
            _flag("split_word", ["Kafka"], span="cough ka", indices=[1, 2]),
            _flag("oov", span="mystery", indices=[0]),
        ]
    }
    report = score(cases, flags, WORDS)
    assert report.total_flags == 2
    assert report.cold_flags == 1
    assert report.cold_flag_rate == 0.5


def test_score_with_no_cases_of_a_class_defaults_that_metric_to_perfect() -> None:
    report = score([], {}, {})
    assert report.recall == 1.0
    assert report.precision == 1.0
    assert report.cold_flag_rate == 0.0


def test_score_matches_whole_words_not_substrings() -> None:
    """A case for "raft" must not be satisfied by a flag on a longer word
    that merely contains it."""
    words = {"b.srt": [Word("rafters", 1, 0, 0), Word("raft", 1, 8, 1)]}
    cases = [ScoredCase("b.srt", "raft", "should-flag")]
    report = score(cases, {"b.srt": [_flag("oov", span="rafters", indices=[0])]}, words)
    assert report.false_negatives == 1


def test_score_context_pins_one_occurrence() -> None:
    words = {
        "b.srt": [
            Word(t, 1, 0, i) for i, t in enumerate("monitor this and monitor that".split())
        ]
    }
    cases = [ScoredCase("b.srt", "monitor", "should-flag", context="monitor that")]
    wrong = score(cases, {"b.srt": [_flag("oov", span="monitor", indices=[0])]}, words)
    right = score(cases, {"b.srt": [_flag("oov", span="monitor", indices=[3])]}, words)
    assert wrong.false_negatives == 1
    assert right.true_positives == 1


def test_score_flag_covering_part_of_a_span_counts() -> None:
    cases = [ScoredCase("a.srt", "cough ka", "should-flag")]
    report = score(cases, {"a.srt": [_flag("oov", span="ka", indices=[2])]}, WORDS)
    assert report.true_positives == 1


def test_score_breaks_recall_down_by_kind() -> None:
    cases = [
        ScoredCase("a.srt", "cough ka", "should-flag", kind="real-word"),
        ScoredCase("a.srt", "raft", "should-flag", kind="non-word"),
    ]
    flags = {"a.srt": [_flag("oov", span="raft", indices=[3])]}
    report = score(cases, flags, WORDS)
    assert report.recall_by_kind == {"non-word": (1, 1), "real-word": (0, 1)}


def test_score_raises_when_a_case_locates_nothing() -> None:
    cases = [ScoredCase("a.srt", "kubernetes", "should-flag")]
    with pytest.raises(CorpusError):
        score(cases, {"a.srt": []}, WORDS)
