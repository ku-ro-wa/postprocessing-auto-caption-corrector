"""The ``correct`` command over the ``CliRunner`` seam. ``build_reader`` (the
default Read-through) and ``build_corrector`` (``--per-flag``) are
monkeypatched to stubs so nothing touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from caption_checker.cli import main
from caption_checker.corrector import StubCorrector

DATA_DIR = Path(__file__).parent / "data"
SRT = str(DATA_DIR / "sample_lecture.srt")
VTT = str(DATA_DIR / "sample_lecture.vtt")


@pytest.fixture
def stub(monkeypatch):
    made: list[StubCorrector] = []

    def factory(model):
        s = StubCorrector()
        made.append(s)
        return s

    monkeypatch.setattr("caption_checker.corrector.build_corrector", factory)
    return made


def _run(*args):
    return CliRunner().invoke(main, ["correct", *args])


# --- required output / guards --------------------------------------------


def test_missing_output_is_a_usage_error(stub) -> None:
    result = _run(SRT)
    assert result.exit_code != 0
    assert "-o PATH" in result.output


def test_non_tty_without_yes_above_names_a_flag(stub) -> None:
    result = _run(SRT, "-o", "out.srt", "--no-cache")
    assert result.exit_code != 0
    assert "not a TTY" in result.output
    assert "con sensus" in result.output  # the first flag, named
    assert stub == []  # never reached the corrector


def test_non_tty_with_yes_above_completes(tmp_path, stub) -> None:
    out = tmp_path / "out.srt"
    result = _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache",
    )
    assert result.exit_code == 0
    assert "consensus algorithms" in out.read_text()
    assert (tmp_path / "out.srt.flags.json").exists()


def test_vtt_is_supported(tmp_path, stub) -> None:
    out = tmp_path / "out.vtt"
    result = _run(
        "--per-flag", VTT, "-o", str(out), "--yes-above", "0.5", "--no-cache",
    )
    assert result.exit_code == 0
    assert "consensus algorithms" in out.read_text()


# --- sidecar / exit code ----------------------------------------------


def test_sidecar_has_one_valid_entry_per_flag(tmp_path, stub) -> None:
    out = tmp_path / "out.srt"
    _run("--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache")

    sidecar = json.loads((tmp_path / "out.srt.flags.json").read_text())
    valid = {
        "applied", "rejected", "not-an-error", "bypassed", "cached",
        "skipped-parse-failure",
    }
    assert len(sidecar) == 4
    for entry in sidecar:
        assert entry["outcome"] in valid
        assert "span" in entry["flag"]


def test_exit_zero_even_when_nothing_meets_the_threshold(tmp_path, stub) -> None:
    out = tmp_path / "out.srt"
    result = _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.999", "--no-cache",
    )
    assert result.exit_code == 0
    # stub confidence is 0.9 < 0.999 -> everything left uncorrected
    assert "con sensus algorithms" in out.read_text()
    sidecar = json.loads((tmp_path / "out.srt.flags.json").read_text())
    assert {e["outcome"] for e in sidecar} == {"rejected"}


# --- estimate (#7) ---------------------------------------------------


def test_estimate_prints_counts_and_makes_no_call(tmp_path) -> None:
    out = tmp_path / "out.srt"
    result = _run(
        SRT, "-o", str(out), "--estimate", "--no-cache"
    )
    assert result.exit_code == 0
    assert "flags: 4" in result.output
    assert "batches: 1" in result.output
    assert "approx cost:" in result.output
    assert not out.exists()
    assert not (tmp_path / "out.srt.flags.json").exists()


def test_estimate_counts_reflect_the_bypass(tmp_path, stub) -> None:
    fixture = tmp_path / "b.srt"
    fixture.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\n"
        "The weather today is calm and clear.\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\n"
        "We watched a wether cross the field.\n",
        encoding="utf-8",
    )
    result = _run(
        "--per-flag", str(fixture), "-o", str(tmp_path / "o.srt"), "--estimate",
        "--no-cache",
    )
    assert "flags: 1" in result.output
    assert "residue (after bypass + cache): 0" in result.output
    assert "batches: 0" in result.output


def test_estimate_shows_a_dollar_cost_for_a_priced_model(tmp_path) -> None:
    # the default model has a static price on file -> a concrete $ figure
    result = _run(
        SRT, "-o", str(tmp_path / "o.srt"), "--estimate", "--no-cache",
    )
    assert "approx cost: $0." in result.output


def test_estimate_reports_no_price_for_an_unknown_model(tmp_path) -> None:
    result = _run(
        SRT, "-o", str(tmp_path / "o.srt"), "--model", "made/up-model",
        "--estimate", "--no-cache",
    )
    assert "no price on file" in result.output


# --- max-calls (#7) ------------------------------------------------


def test_max_calls_below_required_aborts(tmp_path, stub) -> None:
    result = _run(
        "--per-flag", SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5",
        "--max-calls", "0", "--no-cache",
    )
    assert result.exit_code != 0
    assert "max-calls" in result.output
    assert stub[0].calls == []
    assert not (tmp_path / "o.srt").exists()


def test_max_calls_at_the_limit_proceeds(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    result = _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--max-calls", "1",
        "--no-cache",
    )
    assert result.exit_code == 0
    assert out.exists()


# --- detection pass-through --------------------------------------------


def test_oov_zipf_passes_through_to_detection(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    # "driven" (in "event driven") is a real, moderately common word: only
    # flagged once the OOV ceiling is raised well above its Zipf.
    _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.0", "--oov-zipf", "5.5",
        "--no-cache",
    )
    spans = {
        e["flag"]["span"]
        for e in json.loads((tmp_path / "o.srt.flags.json").read_text())
    }
    assert any("driven" in s for s in spans)


def test_vocab_file_passes_through_to_detection(tmp_path, stub) -> None:
    vocab = tmp_path / "extra.txt"
    vocab.write_text("cubernetes\n", encoding="utf-8")  # now a known term
    out = tmp_path / "o.srt"
    _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--vocab", str(vocab),
        "--no-cache",
    )
    spans = {
        e["flag"]["span"]
        for e in json.loads((tmp_path / "o.srt.flags.json").read_text())
    }
    assert "cubernetes" not in spans


# --- eval table (#8) ---------------------------------------------------


def test_eval_out_writes_a_six_column_table(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    table = tmp_path / "eval.md"
    _run(
        "--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--eval-out", str(table),
        "--no-cache",
    )
    lines = table.read_text().splitlines()
    assert lines[0].split("|")[1:-1] == [
        " timestamp ", " span ", " detector ", " suggestion ",
        " llm_conf ", " verdict ",
    ]
    assert lines[1] == "|---|---|---|---|---|---|"
    # one row per flag, verdict column blank
    assert len(lines) == 2 + 4
    for row in lines[2:]:
        assert row.rstrip().endswith("| |")


def test_no_eval_table_without_the_flag(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    _run("--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache")
    assert not (tmp_path / "eval.md").exists()


# --- decision cache over the CLI (#5) --------------------------------


def test_second_cli_run_uses_the_cache(tmp_path, monkeypatch) -> None:
    made: list[StubCorrector] = []
    monkeypatch.setattr(
        "caption_checker.corrector.build_corrector",
        lambda model: made.append(StubCorrector()) or made[-1],
    )
    cache = tmp_path / "cache.json"
    common = ("-o", str(tmp_path / "o.srt"), "--yes-above", "0.5",
              "--cache-file", str(cache))

    _run("--per-flag", SRT, *common)
    assert len(made[0].calls) == 1

    _run("--per-flag", SRT, *common)
    assert made[1].calls == []


def test_cache_file_isolation(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _run("--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--cache-file", str(a))
    _run("--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--cache-file", str(b))
    # each run had its own fresh stub; the b-run could not have reused a's cache
    assert len(stub[0].calls) == 1
    assert len(stub[1].calls) == 1


# --- Read-through and Priming terms (ADR 0006) ------------------------------


@pytest.fixture
def reader(monkeypatch):
    from caption_checker.readthrough import StubReader

    made: list[StubReader] = []

    def factory(model):
        made.append(StubReader(extra={"leader election": "leader elections"}))
        return made[-1]

    monkeypatch.setattr("caption_checker.readthrough.build_reader", factory)
    return made


def test_read_through_is_the_default_pass(tmp_path, stub, reader) -> None:
    out = tmp_path / "o.srt"
    result = _run(
        SRT, "-o", str(out), "--yes-above", "0.5", "--priming-term", "Kafka",
    )
    assert result.exit_code == 0, result.output
    assert stub == []  # never built the per-flag corrector
    assert "leader elections using" in out.read_text()
    assert reader[0].requests[0].priming_terms == ["Kafka"]
    sidecar = json.loads((tmp_path / "o.srt.flags.json").read_text())
    assert any(e["flag"]["detector"] == "read_through" for e in sidecar)


def test_per_flag_restores_the_per_flag_pass(tmp_path, stub, reader) -> None:
    out = tmp_path / "o.srt"
    result = _run(SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache", "--per-flag")
    assert result.exit_code == 0, result.output
    assert reader == []
    assert len(stub[0].calls) == 1
    assert "leader election using" in out.read_text()  # no Read-through find


def test_read_through_flag_is_gone() -> None:
    result = _run(SRT, "-o", "o.srt", "--read-through")
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_estimate_prices_the_read_through_by_default(tmp_path, reader) -> None:
    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate")
    assert result.exit_code == 0, result.output
    assert "pass: read-through" in result.output
    assert "batches: 1" in result.output
    assert "residue" not in result.output  # no bypass or cache in this pass
    assert reader == []


def test_estimate_prices_the_per_flag_pass_with_per_flag(tmp_path, stub) -> None:
    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate", "--per-flag", "--no-cache")
    assert result.exit_code == 0, result.output
    assert "pass: per-flag" in result.output
    assert "residue (after bypass + cache): 4" in result.output
    assert stub == []


def test_priming_terms_join_the_local_vocabulary(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    _run("--per-flag", SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache",
         "--priming-term", "cubernetes")
    assert "look at cubernetes" in out.read_text()  # now a known term


def test_check_accepts_priming_terms() -> None:
    result = CliRunner().invoke(main, ["check", SRT, "--priming-term", "cubernetes"])
    assert "con sensus" in result.output  # still checked
    assert "cubernetes" not in result.output


# --- Read-through configuration (#40, ADR 0007) ----------------------------


@pytest.fixture
def configs(monkeypatch):
    """Each Read-through configuration ``correct`` built a reader from."""
    from caption_checker.readthrough import StubReader

    made = []

    def factory(config):
        made.append(config)
        return StubReader()

    monkeypatch.setattr("caption_checker.readthrough.build_reader", factory)
    return made


def test_read_through_defaults_to_the_qwen_configuration(tmp_path, configs) -> None:
    from caption_checker.readthrough import (
        _ERRORS_TWO_TESTS,
        CONFIGS,
        ChunkRequest,
        _prompt,
    )

    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5")
    assert result.exit_code == 0, result.output
    [config] = configs
    assert config is CONFIGS["qwen3.6-plus-p2"]
    assert config.model_id == "qwen/qwen3.6-plus"
    assert config.request_options == {"reasoning": {"enabled": False}}
    # the two-tests prompt, not v4's
    request = ChunkRequest(words=[(0, "hello")])
    assert config.build_messages(request) == _prompt(errors=_ERRORS_TWO_TESTS)(request)


def test_config_picks_a_registered_read_through_configuration(tmp_path, configs) -> None:
    from caption_checker.readthrough import CONFIGS

    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5",
                  "--config", "flash-v4")
    assert result.exit_code == 0, result.output
    assert configs == [CONFIGS["flash-v4"]]


def test_model_runs_prompt_v4_with_that_model(tmp_path, configs) -> None:
    from caption_checker.readthrough import v4

    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5",
                  "--model", "some/slug")
    assert result.exit_code == 0, result.output
    assert configs == [v4("some/slug")]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--config", "flash-v4", "--model", "some/slug"), "not both"),
        (("--config", "no-such-name"), "flash-v4"),  # lists the registered names
        (("--per-flag", "--config", "flash-v4"), "--per-flag"),
    ],
)
def test_bad_configuration_flags_are_usage_errors(
    tmp_path, configs, stub, args, message
) -> None:
    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5", *args)
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert configs == [] and stub == []


def test_per_flag_defaults_to_gemini_flash(tmp_path, monkeypatch) -> None:
    from caption_checker.corrector import StubCorrector

    models = []

    def factory(model):
        models.append(model)
        return StubCorrector()

    monkeypatch.setattr("caption_checker.corrector.build_corrector", factory)
    result = _run("--per-flag", SRT, "-o", str(tmp_path / "o.srt"),
                  "--yes-above", "0.5", "--no-cache")
    assert result.exit_code == 0, result.output
    assert models == ["google/gemini-2.5-flash"]


def test_estimate_prices_the_default_configuration_s_model(tmp_path, configs) -> None:
    from caption_checker.correct import _MODEL_PROMPT_PRICE

    assert "qwen/qwen3.6-plus" in _MODEL_PROMPT_PRICE
    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate")
    assert result.exit_code == 0, result.output
    assert "configuration: qwen3.6-plus-p2 (qwen/qwen3.6-plus)" in result.output
    assert "approx cost: $0." in result.output
    assert configs == []


def test_estimate_names_the_model_with_model(tmp_path, configs) -> None:
    result = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate",
                  "--model", "google/gemini-2.5-flash")
    assert "configuration: v4 (google/gemini-2.5-flash)" in result.output
    assert "approx cost: $0." in result.output


def test_estimate_counts_the_configuration_s_own_prompt(tmp_path, configs) -> None:
    # same model, different prompts: the p2 prompt is longer than v4's
    default = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate")
    v4 = _run(SRT, "-o", str(tmp_path / "o.srt"), "--estimate",
              "--model", "qwen/qwen3.6-plus")
    cost = lambda r: float(r.output.split("approx cost: $")[1].split()[0])  # noqa: E731
    assert cost(default) > cost(v4)
