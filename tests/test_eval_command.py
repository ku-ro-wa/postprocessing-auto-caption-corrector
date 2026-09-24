"""The ``eval`` command and the ``run_eval`` seam behind it: a named corpus
scored against a pluggable system under test, entirely offline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from click.testing import CliRunner

from caption_checker.cli import main
from caption_checker.evaluation import NamedCorpus, System, run_eval
from caption_checker.models import Cue, Flag

SRT = """1
00:00:00,000 --> 00:00:02,000
the cough ka raft consensus
"""


def _corpus(tmp_path: Path) -> NamedCorpus:
    (tmp_path / "audited.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "fixture.srt").write_text(SRT, encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"source": "audited.srt", "span": "cough ka", "verdict": "should-flag",
                 "kind": "real-word"},
                {"source": "audited.srt", "span": "raft", "verdict": "should-not-flag"},
                {"source": "fixture.srt", "span": "the", "verdict": "should-flag"},
            ]
        ),
        encoding="utf-8",
    )
    return NamedCorpus(cases_path=cases, data_dir=tmp_path, exhaustive_sources=("audited.srt",))


def _flag_words(*indices: int) -> System:
    """A fake system under test that flags the given global indices."""

    def system(cues: list[Cue]) -> list[Flag]:
        return [
            Flag(
                span=str(i),
                global_indices=[i],
                cue_index=1,
                start=timedelta(0),
                end=timedelta(seconds=1),
                detector="fake",
                reason="test",
            )
            for i in indices
        ]

    return system


def test_run_eval_scores_the_system_on_every_source_in_the_corpus(tmp_path: Path) -> None:
    report = run_eval(_corpus(tmp_path), _flag_words(2, 4))
    assert report.recall == 0.5  # caught audited's "cough ka", missed fixture's "the"
    assert report.recall_by_kind == {"real-word": (1, 1)}
    assert report.total_flags == 4  # two flags on each of the two sources
    assert report.flag_precision == 0.5  # only audited.srt's two flags count


def test_run_eval_scores_exhaustive_sources_that_have_no_cases(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    (tmp_path / "clean.srt").write_text(SRT, encoding="utf-8")
    corpus = replace(corpus, exhaustive_sources=(*corpus.exhaustive_sources, "clean.srt"))
    report = run_eval(corpus, _flag_words(2))
    assert report.exhaustive_flags == 2
    assert report.flag_precision == 0.5


def test_eval_command_prints_every_number_for_the_scored_corpus() -> None:
    result = CliRunner().invoke(main, ["eval"])
    assert result.exit_code == 0, result.output
    for label in (
        "corpus: scored",
        "system: local",
        "recall:",
        "real-word",
        "case precision:",
        "flag-level precision:",
        "cold-flag rate:",
    ):
        assert label in result.output


def test_eval_command_rejects_an_unknown_system() -> None:
    result = CliRunner().invoke(main, ["eval", "--system", "nope"])
    assert result.exit_code != 0
