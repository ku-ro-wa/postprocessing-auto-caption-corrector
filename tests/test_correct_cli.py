"""The ``correct`` command over the ``CliRunner`` seam. ``build_corrector`` is
monkeypatched to a ``StubCorrector`` so nothing touches the network."""

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
    result = _run(SRT, "--no-embeddings")
    assert result.exit_code != 0
    assert "-o PATH" in result.output


def test_non_tty_without_yes_above_names_a_flag(stub) -> None:
    result = _run(SRT, "-o", "out.srt", "--no-cache", "--no-embeddings")
    assert result.exit_code != 0
    assert "not a TTY" in result.output
    assert "con sensus" in result.output  # the first flag, named
    assert stub == []  # never reached the corrector


def test_non_tty_with_yes_above_completes(tmp_path, stub) -> None:
    out = tmp_path / "out.srt"
    result = _run(
        SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache",
        "--no-embeddings",
    )
    assert result.exit_code == 0
    assert "consensus algorithms" in out.read_text()
    assert (tmp_path / "out.srt.flags.json").exists()


def test_vtt_is_supported(tmp_path, stub) -> None:
    out = tmp_path / "out.vtt"
    result = _run(
        VTT, "-o", str(out), "--yes-above", "0.5", "--no-cache",
        "--no-embeddings",
    )
    assert result.exit_code == 0
    assert "consensus algorithms" in out.read_text()


# --- sidecar / exit code ----------------------------------------------


def test_sidecar_has_one_valid_entry_per_flag(tmp_path, stub) -> None:
    out = tmp_path / "out.srt"
    _run(SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache", "--no-embeddings")

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
        SRT, "-o", str(out), "--yes-above", "0.999", "--no-cache",
        "--no-embeddings",
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
        SRT, "-o", str(out), "--estimate", "--no-cache", "--no-embeddings"
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
        str(fixture), "-o", str(tmp_path / "o.srt"), "--estimate",
        "--no-cache", "--no-embeddings",
    )
    assert "flags: 1" in result.output
    assert "residue (after bypass + cache): 0" in result.output
    assert "batches: 0" in result.output


def test_estimate_shows_a_dollar_cost_for_a_priced_model(tmp_path) -> None:
    # the default model has a static price on file -> a concrete $ figure
    result = _run(
        SRT, "-o", str(tmp_path / "o.srt"), "--estimate", "--no-cache",
        "--no-embeddings",
    )
    assert "approx cost: $0." in result.output


def test_estimate_reports_no_price_for_an_unknown_model(tmp_path) -> None:
    result = _run(
        SRT, "-o", str(tmp_path / "o.srt"), "--model", "made/up-model",
        "--estimate", "--no-cache", "--no-embeddings",
    )
    assert "no price on file" in result.output


# --- max-calls (#7) ------------------------------------------------


def test_max_calls_below_required_aborts(tmp_path, stub) -> None:
    result = _run(
        SRT, "-o", str(tmp_path / "o.srt"), "--yes-above", "0.5",
        "--max-calls", "0", "--no-cache", "--no-embeddings",
    )
    assert result.exit_code != 0
    assert "max-calls" in result.output
    assert stub[0].calls == []
    assert not (tmp_path / "o.srt").exists()


def test_max_calls_at_the_limit_proceeds(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    result = _run(
        SRT, "-o", str(out), "--yes-above", "0.5", "--max-calls", "1",
        "--no-cache", "--no-embeddings",
    )
    assert result.exit_code == 0
    assert out.exists()


# --- detection pass-through --------------------------------------------


def test_oov_zipf_passes_through_to_detection(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    # "driven" (in "event driven") is a real, moderately common word: only
    # flagged once the OOV ceiling is raised well above its Zipf.
    _run(
        SRT, "-o", str(out), "--yes-above", "0.0", "--oov-zipf", "5.5",
        "--no-cache", "--no-embeddings",
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
        SRT, "-o", str(out), "--yes-above", "0.5", "--vocab", str(vocab),
        "--no-cache", "--no-embeddings",
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
        SRT, "-o", str(out), "--yes-above", "0.5", "--eval-out", str(table),
        "--no-cache", "--no-embeddings",
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
    _run(SRT, "-o", str(out), "--yes-above", "0.5", "--no-cache", "--no-embeddings")
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
              "--cache-file", str(cache), "--no-embeddings")

    _run(SRT, *common)
    assert len(made[0].calls) == 1

    _run(SRT, *common)
    assert made[1].calls == []


def test_cache_file_isolation(tmp_path, stub) -> None:
    out = tmp_path / "o.srt"
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _run(SRT, "-o", str(out), "--yes-above", "0.5", "--cache-file", str(a), "--no-embeddings")
    _run(SRT, "-o", str(out), "--yes-above", "0.5", "--cache-file", str(b), "--no-embeddings")
    # each run had its own fresh stub; the b-run could not have reused a's cache
    assert len(stub[0].calls) == 1
    assert len(stub[1].calls) == 1
