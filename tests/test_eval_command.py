"""The ``eval`` command and the ``run_eval`` seam behind it: a named corpus
scored against a pluggable system under test, entirely offline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from caption_checker.cli import main
from caption_checker.evaluation import (
    CORPORA,
    SYSTEMS,
    CorpusError,
    NamedCorpus,
    System,
    load_corpus,
    run_eval,
    score,
)
from caption_checker.models import Cue, Flag
from caption_checker.parser import parse, tokenize
from caption_checker.readthrough import (
    CONFIGS,
    ReadThroughConfig,
    StubReader,
    build_messages,
    parse_reply,
)

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


def test_run_eval_checks_every_case_locates_before_running_the_system(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    cases = json.loads(corpus.cases_path.read_text())
    cases.append({"source": "fixture.srt", "span": "absent", "verdict": "should-flag"})
    corpus.cases_path.write_text(json.dumps(cases), encoding="utf-8")
    ran: list[int] = []

    def system(cues: list[Cue], priming_terms: list[str]) -> list[Flag]:
        ran.append(1)
        return []

    with pytest.raises(CorpusError, match="absent"):
        run_eval(corpus, system)
    assert ran == []  # a broken corpus costs no Read-through calls


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
    local = SYSTEMS["local"](CONFIGS["flash-v4"])
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

    configs: list[ReadThroughConfig] = []

    def factory(config: ReadThroughConfig) -> StubReader:
        configs.append(config)
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
    # --model alone means prompt v4 with that model
    [config] = configs
    assert config.model_id == "some/model"
    assert config.build_messages is build_messages
    assert config.parse_reply is parse_reply
    assert "system: read-through (v4: some/model)" in result.output
    assert "real-word: 1.000 (1/1; 1/1 with candidate)" in result.output
    # two 2-second transcripts, one spend object per system
    assert "cost: $0.0100 for 0.001 audio hours ($9.00 per audio hour)" in result.output


def _eval_read_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str
) -> tuple[Result, list[tuple[ReadThroughConfig, StubReader]]]:
    """Run ``eval --system read-through`` on a small Dev-like corpus with a
    stand-in Reader; return the result and each (configuration, Reader) the
    factory was asked for."""
    made: list[tuple[ReadThroughConfig, StubReader]] = []

    def factory(config: ReadThroughConfig) -> StubReader:
        made.append((config, StubReader(extra={"cough ka": "Kafka"})))
        return made[-1][1]

    monkeypatch.setattr("caption_checker.readthrough.build_reader", factory)
    monkeypatch.setitem(CORPORA, "earnings21-dev", _earnings_like(tmp_path))
    result = CliRunner().invoke(
        main,
        ["eval", "--corpus", "earnings21-dev", "--system", "read-through", *args],
    )
    return result, made


def test_eval_command_runs_a_registered_read_through_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, made = _eval_read_through(tmp_path, monkeypatch, "--config", "flash-v4")
    assert result.exit_code == 0, result.output
    [(config, reader)] = made
    assert config is CONFIGS["flash-v4"]
    assert "system: read-through (flash-v4: google/gemini-2.5-flash)" in result.output
    assert "real-word: 1.000 (1/1; 1/1 with candidate)" in result.output
    # flash-v4 is today's Read-through: the same messages for the same
    # chunk, and the same reading of the same reply
    assert reader.requests
    for request in reader.requests:
        assert config.build_messages(request) == build_messages(request)
    request = reader.requests[0]
    gi, text = request.words[0]
    verdicts = [[h.start, h.end, h.span, None, "misheard", 0.5, h.id, "w"] for h in request.hints]
    verdicts.append([gi, gi, text, "X", "misheard", 0.95, None, "w"])
    reply = json.dumps({"verdicts": verdicts})
    assert parse_reply(reply, request)
    assert config.parse_reply(reply, request) == parse_reply(reply, request)


def test_qwen_p2_is_the_default_read_through_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ADR 0007: what eval scores with no flags is what correct ships
    result, made = _eval_read_through(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert [config for config, _ in made] == [CONFIGS["qwen3.6-plus-p2"]]
    assert "system: read-through (qwen3.6-plus-p2: qwen/qwen3.6-plus)" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--config", "flash-v4", "--model", "some/model"), "not both"),
        (("--config", "no-such-config"), "no-such-config"),
    ],
)
def test_eval_command_refuses_a_bad_read_through_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], message: str
) -> None:
    result, made = _eval_read_through(tmp_path, monkeypatch, *args)
    assert result.exit_code != 0
    assert message in result.output
    assert "Traceback" not in result.output
    assert made == []  # failed before building a Reader


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("gemini-3.8-flash-v4", "google/gemini-3.8-flash"),
        ("deepseek-v4-pro-v4", "deepseek/deepseek-v4-pro"),
        ("qwen3.6-plus-v4", "qwen/qwen3.6-plus"),
        ("gpt-5.6-luna-v4", "openai/gpt-5.6-luna"),
    ],
)
def test_eval_command_runs_a_reasoning_model_with_prompt_v4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, model: str
) -> None:
    # #28's shortlist runs with reasoning off, which these models only do
    # when asked; the prompt and reply format stay v4's.
    result, made = _eval_read_through(tmp_path, monkeypatch, "--config", name)
    assert result.exit_code == 0, result.output
    [(config, _)] = made
    assert config.model_id == model
    assert config.build_messages is build_messages
    assert config.parse_reply is parse_reply
    assert f"system: read-through ({name}: {model})" in result.output


@pytest.mark.parametrize(
    ("name", "model", "keyed"),
    [
        ("gemini-3.5-flash-lite-p2", "google/gemini-3.5-flash-lite", True),
        ("gpt-5.6-luna-p2", "openai/gpt-5.6-luna", False),
        ("qwen3.6-plus-p2", "qwen/qwen3.6-plus", False),
        ("deepseek-v4-pro-p2", "deepseek/deepseek-v4-pro", True),
        ("gemini-3.8-flash-p2", "google/gemini-3.8-flash", False),
    ],
)
def test_eval_command_runs_a_frozen_challenger_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, model: str, keyed: bool
) -> None:
    # #33 froze one configuration per challenger for #34's ranking: v4's task
    # with its own error rules, and for some its own reply format -- which
    # it then reads.
    result, made = _eval_read_through(tmp_path, monkeypatch, "--config", name)
    assert result.exit_code == 0, result.output
    [(config, reader)] = made
    assert config.model_id == model
    assert f"system: read-through ({name}: {model})" in result.output
    request = reader.requests[0]
    system, _ = config.build_messages(request)
    assert "List a change only when both of these hold" in system["content"]
    verdict = [0, 0, request.words[0][1], "X", "misheard", 0.95, "why"]
    if keyed:
        assert '{"hints": {...}, "errors": [...]}' in system["content"]
        reply = {"hints": {h.id: [h.start, h.end, h.span, None, "misheard", 0.9, "ok"]
                           for h in request.hints}, "errors": [verdict]}
    else:
        assert '{"verdicts": [...]}' in system["content"]
        reply = {"verdicts": [
            *([h.start, h.end, h.span, None, "misheard", 0.9, h.id, "ok"]
              for h in request.hints),
            [*verdict[:6], None, verdict[6]],
        ]}
    parsed = config.parse_reply(json.dumps(reply), request)
    assert [(v.start, v.replacement) for v in parsed if v.hint is None] == [(0, "X")]


def test_eval_command_reports_request_errors_apart_from_failed_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(config: ReadThroughConfig) -> StubReader:  # noqa: ARG001
        return StubReader(request_error="HTTP 402 Payment Required: no credit")

    monkeypatch.setattr("caption_checker.readthrough.build_reader", factory)
    monkeypatch.setitem(CORPORA, "earnings21-dev", _earnings_like(tmp_path))
    result = CliRunner().invoke(
        main, ["eval", "--corpus", "earnings21-dev", "--system", "read-through"]
    )
    assert result.exit_code == 0, result.output
    # one chunk per transcript, both lost to the request, not the reply
    assert (
        "read-through: 2 failed chunks (2 on request errors, last: "
        "HTTP 402 Payment Required: no credit)" in result.output
    )


def test_eval_command_reports_failed_chunks_without_request_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _ = _eval_read_through(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert "read-through: 0 failed chunks (0 on request errors)\n" in result.output


def test_read_through_system_scores_only_claimed_errors(tmp_path: Path) -> None:
    from caption_checker.evaluation import ReadThroughSystem
    (tmp_path / "t.srt").write_text(SRT, encoding="utf-8")
    cues = parse(tmp_path / "t.srt")
    system = ReadThroughSystem(StubReader(extra={"cough ka": "Kafka", "raft": None}))  # type: ignore[dict-item]
    assert [f.span for f in system(cues, [])] == ["cough ka"]


def test_run_eval_measures_audio_duration(tmp_path: Path) -> None:
    report = run_eval(_corpus(tmp_path), _flag_words())
    assert report.audio_seconds == 4.0  # two 2-second sources


@pytest.mark.parametrize(
    "name", ["earnings21-heldout", "earnings21-heldout-2", "audited-heldout-2"]
)
def test_eval_command_refuses_a_held_out_corpus_without_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    held_out = replace(_earnings_like(tmp_path), held_out=True)
    monkeypatch.setitem(CORPORA, name, held_out)
    refused = CliRunner().invoke(main, ["eval", "--corpus", name])
    assert refused.exit_code != 0
    assert "--final" in refused.output
    final = CliRunner().invoke(main, ["eval", "--corpus", name, "--final"])
    assert final.exit_code == 0, final.output


def test_only_the_final_comparison_sets_are_held_out() -> None:
    # The first Audited Held-out set moved to the Dev set once its misses
    # motivated a change (issue #27); earnings21-heldout-2 and
    # audited-heldout-2 are fresh for #28's final scoring.
    assert {name for name, c in CORPORA.items() if c.held_out} == {
        "earnings21-heldout",
        "earnings21-heldout-2",
        "audited-heldout-2",
    }


@pytest.mark.parametrize("name, videos", [("audited-dev", 5), ("audited-heldout-2", 6)])
def test_audited_corpus_locates_every_case_in_its_transcripts(name: str, videos: int) -> None:
    # Scored with no Flags at all: checks the corpus itself (every span
    # locates, every transcript is exhaustive and has errors listed) without
    # running any system over a Held-out set.
    corpus = CORPORA[name]
    cases = load_corpus(corpus.cases_path)
    assert len(corpus.exhaustive_sources) == videos
    assert {c.source for c in cases} == set(corpus.exhaustive_sources)
    assert all(c.verdict == "should-flag" and c.kind for c in cases)
    words = {s: tokenize(parse(corpus.data_dir / s)) for s in corpus.exhaustive_sources}
    report = score(cases, {}, words, exhaustive_sources=corpus.exhaustive_sources)
    assert report.false_negatives == len(cases)


def test_audited_heldout_2_shares_no_transcript_with_another_corpus() -> None:
    fresh = set(CORPORA["audited-heldout-2"].exhaustive_sources)
    committed = {
        name: c
        for name, c in CORPORA.items()
        if c.manifest_path is None and name != "audited-heldout-2"
    }
    for name, corpus in committed.items():
        used = {c.source for c in load_corpus(corpus.cases_path)}
        assert not fresh & (used | set(corpus.exhaustive_sources)), name
