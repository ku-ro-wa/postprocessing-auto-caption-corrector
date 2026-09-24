"""The ``eval`` command and the ``run_eval`` seam behind it: a named corpus
scored against a pluggable system under test, entirely offline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from caption_checker.cli import main
from caption_checker.evaluation import (
    CORPORA,
    SYSTEMS,
    CorpusError,
    NamedCorpus,
    System,
    run_eval,
)
from caption_checker.models import Cue, Flag
from caption_checker.parser import parse

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

    def system(cues: list[Cue], priming_terms: list[str]) -> list[Flag]:
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


def _earnings_like(tmp_path: Path) -> NamedCorpus:
    """An Auto-labelled corpus as ``build-earnings21`` writes it: every source
    in the manifest is exhaustive and carries Priming terms."""
    for source in ("a.srt", "b.srt"):
        (tmp_path / source).write_text(SRT, encoding="utf-8")
    (tmp_path / "cases.json").write_text(
        json.dumps(
            [
                {"source": "a.srt", "span": "cough ka", "verdict": "should-flag",
                 "candidate": "Kafka", "kind": "real-word", "entity": True},
                {"source": "a.srt", "span": "the", "verdict": "should-flag",
                 "kind": "function-word", "entity": False},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"sources": {"a.srt": {"priming_terms": ["Kafka Inc"]},
                         "b.srt": {"priming_terms": ["Bee Corp"]}}}
        ),
        encoding="utf-8",
    )
    return NamedCorpus(
        cases_path=tmp_path / "cases.json",
        data_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        headline_kinds=("non-word", "real-word"),
        caveat="labels are noisy",
    )


def test_run_eval_treats_every_manifest_source_as_exhaustive(tmp_path: Path) -> None:
    report = run_eval(_earnings_like(tmp_path), _flag_words(2))
    assert report.exhaustive_flags == 2  # b.srt has no cases but is still scored
    assert report.flags_touching_errors == 1
    assert report.entity_recall == (1, 1)  # a detection, whatever its candidates


def test_run_eval_scores_detection_and_tallies_candidate_matches_aside(
    tmp_path: Path,
) -> None:
    # ADR 0006: one rule on every corpus -- a Flag touching the error is a
    # catch; proposing the case's candidate is a secondary number.
    report = run_eval(_earnings_like(tmp_path), _flag_words(2))
    assert report.recall_by_kind["real-word"] == (1, 1)
    assert report.with_candidate_by_kind["real-word"] == (0, 1)  # no Kafka


def test_eval_command_prints_candidate_matches_beside_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(CORPORA, "earnings21-dev", _earnings_like(tmp_path))
    monkeypatch.setitem(SYSTEMS, "local", lambda model: _flag_words(2))
    result = CliRunner().invoke(main, ["eval", "--corpus", "earnings21-dev"])
    assert "real-word: 1.000 (1/1; 0/1 with candidate)" in result.output


def test_run_eval_passes_priming_terms_only_when_asked(tmp_path: Path) -> None:
    corpus = _earnings_like(tmp_path)
    calls: list[list[str]] = []

    def system(cues: list[Cue], priming_terms: list[str]) -> list[Flag]:
        calls.append(priming_terms)
        return []

    run_eval(corpus, system)
    run_eval(corpus, system, priming=True)
    assert calls == [[], [], ["Kafka Inc"], ["Bee Corp"]]


def test_run_eval_refuses_priming_on_a_corpus_without_terms(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="Priming terms"):
        run_eval(_corpus(tmp_path), _flag_words(), priming=True)


def test_run_eval_explains_how_to_build_a_missing_corpus(tmp_path: Path) -> None:
    corpus = replace(_earnings_like(tmp_path), cases_path=tmp_path / "nope.json")
    with pytest.raises(CorpusError, match="build-earnings21"):
        run_eval(corpus, _flag_words())


def test_local_system_adds_priming_terms_to_the_domain_vocabulary(tmp_path: Path) -> None:
    (tmp_path / "t.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nwe deploy on kubernetis today\n",
        encoding="utf-8",
    )
    cues = parse(tmp_path / "t.srt")
    local = SYSTEMS["local"]("unused")
    assert any(f.span == "kubernetis" for f in local(cues, []))
    assert not any(f.span == "kubernetis" for f in local(cues, ["Kubernetis"]))


def test_eval_command_reports_headline_recall_entity_and_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(CORPORA, "earnings21-dev", _earnings_like(tmp_path))
    result = CliRunner().invoke(main, ["eval", "--corpus", "earnings21-dev", "--priming"])
    assert result.exit_code == 0, result.output
    for label in (
        "priming: on",
        "all-kinds recall:",
        "headline recall (non-word, real-word):",
        "entity:",
        "function-word",
        "labels are noisy",
    ):
        assert label in result.output


def test_eval_command_fails_cleanly_on_an_unbuilt_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = replace(_earnings_like(tmp_path), cases_path=tmp_path / "nope.json")
    monkeypatch.setitem(CORPORA, "earnings21-dev", missing)
    result = CliRunner().invoke(main, ["eval", "--corpus", "earnings21-dev"])
    assert result.exit_code != 0
    assert "build-earnings21" in result.output


def test_eval_command_scores_the_read_through_with_its_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from caption_checker.corrector import Spend
    from caption_checker.readthrough import StubReader

    def factory(model: str) -> StubReader:
        stub = StubReader(extra={"cough ka": "Kafka"})
        stub.spend = Spend(requests=2, cost_usd=0.01)
        return stub

    monkeypatch.setattr("caption_checker.readthrough.build_reader", factory)
    monkeypatch.setitem(CORPORA, "earnings21-dev", _earnings_like(tmp_path))
    result = CliRunner().invoke(
        main,
        ["eval", "--corpus", "earnings21-dev", "--system", "read-through",
         "--model", "some/model"],
    )
    assert result.exit_code == 0, result.output
    assert "system: read-through (some/model)" in result.output
    assert "real-word: 1.000 (1/1; 1/1 with candidate)" in result.output
    # two 2-second transcripts, one spend object per system
    assert "cost: $0.0100 for 0.001 audio hours ($9.00 per audio hour)" in result.output


def test_read_through_system_scores_only_claimed_errors(tmp_path: Path) -> None:
    from caption_checker.evaluation import ReadThroughSystem
    from caption_checker.readthrough import StubReader

    (tmp_path / "t.srt").write_text(SRT, encoding="utf-8")
    cues = parse(tmp_path / "t.srt")
    system = ReadThroughSystem(StubReader(extra={"cough ka": "Kafka", "raft": None}))  # type: ignore[dict-item]
    assert [f.span for f in system(cues, [])] == ["cough ka"]


def test_run_eval_measures_audio_duration(tmp_path: Path) -> None:
    report = run_eval(_corpus(tmp_path), _flag_words())
    assert report.audio_seconds == 4.0  # two 2-second sources
