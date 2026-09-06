from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from caption_checker.cli import main
from caption_checker.detect import detect
from caption_checker.models import DetectConfig
from caption_checker.parser import parse, tokenize

DATA_DIR = Path(__file__).parent / "data"
NO_EMBED = DetectConfig(enable_embeddings=False)


def _flags(filename: str = "sample_lecture.srt", config: DetectConfig = NO_EMBED):
    return detect(parse(DATA_DIR / filename), config=config)


def _covering(flags, *substrings):
    """Flags whose span contains any of the given substrings (case-insensitive)."""
    out = []
    for flag in flags:
        span = flag.span.lower()
        if any(s.lower() in span for s in substrings):
            out.append(flag)
    return out


def test_cubernetes_flagged_with_candidate() -> None:
    hits = _covering(_flags(), "cubernetes")
    assert hits, "expected a flag on 'cubernetes'"
    flag = hits[0]
    assert "Kubernetes" in flag.candidates
    assert "phonetic_vocab" in flag.detector


def test_consensus_split_flagged() -> None:
    hits = _covering(_flags(), "con sensus")
    assert hits, "expected a flag spanning 'con sensus'"
    flag = hits[0]
    assert "consensus" in [c.lower() for c in flag.candidates]
    assert "split_word" in flag.detector
    assert len(flag.global_indices) == 2


def test_kafka_split_flagged() -> None:
    hits = _covering(_flags(), "cough ka")
    assert hits, "expected a flag spanning 'cough ka'"
    assert all("Kafka" in f.candidates for f in hits)
    assert all("split_word" in f.detector for f in hits)
    # appears in cue 5 and cue 6
    assert {f.cue_index for f in hits} == {5, 6}


def test_clean_terms_not_flagged() -> None:
    flags = _flags()
    flagged_spans = " ".join(f.span.lower() for f in flags)
    for term in (
        "welcome",
        "lecture",
        "distributed",
        "systems",
        "raft",
        "paxos",
        "protocol",
        "leader",
        "election",
        "randomized",
    ):
        assert term not in flagged_spans, f"{term!r} should not be flagged"


def test_flag_rate_under_15pct() -> None:
    cues = parse(DATA_DIR / "sample_lecture.srt")
    flags = detect(cues, config=NO_EMBED)
    word_count = len(tokenize(cues))
    assert len(flags) / word_count < 0.15


def test_vtt_path_finds_same_errors() -> None:
    flags = _flags("sample_lecture.vtt")
    assert _covering(flags, "cubernetes")
    assert _covering(flags, "con sensus")
    assert _covering(flags, "cough ka")


def test_context_attached() -> None:
    flag = _covering(_flags(), "cubernetes")[0]
    assert "cubernetes" in flag.context
    assert flag.context.endswith(".")


def test_merge_combines_overlapping_detectors() -> None:
    flag = _covering(_flags(), "cubernetes")[0]
    # oov and phonetic_vocab both fire on this token and are merged
    assert flag.detector == "oov+phonetic_vocab"
    assert flag.confidence > 0.9


def test_no_false_positive_on_clean_transcript() -> None:
    clean_srt = (
        "1\n00:00:00,000 --> 00:00:02,000\n"
        "Welcome to the lecture on distributed systems.\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n"
        "Today we discuss consensus and the Raft protocol.\n"
    )
    path = DATA_DIR / "_tmp_clean.srt"
    path.write_text(clean_srt, encoding="utf-8")
    try:
        flags = detect(parse(path), config=NO_EMBED)
        assert flags == []
    finally:
        path.unlink()


# --- CLI ---------------------------------------------------------------------


def test_cli_json_output_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["check", str(DATA_DIR / "sample_lecture.srt"), "--no-embeddings",
         "--format", "json"],
    )
    assert result.exit_code == 1  # flags present
    payload = json.loads(result.output)
    assert isinstance(payload, list) and payload
    required = {
        "span", "global_indices", "cue_index", "start", "end",
        "detector", "reason", "candidates", "confidence", "context",
    }
    for item in payload:
        assert required <= item.keys()
        assert isinstance(item["start"], (int, float))
        assert isinstance(item["end"], (int, float))
        assert isinstance(item["global_indices"], list)


def test_cli_text_output_and_exit_code() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["check", str(DATA_DIR / "sample_lecture.srt"), "--no-embeddings"],
    )
    assert result.exit_code == 1
    assert "»cubernetes«" in result.output
    assert 'Kubernetes' in result.output


def test_cli_clean_file_exit_zero(tmp_path: Path) -> None:
    clean = tmp_path / "clean.srt"
    clean.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nWelcome to the lecture.\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["check", str(clean), "--no-embeddings"])
    assert result.exit_code == 0
    assert "no likely caption errors" in result.output


# --- embedding tier (optional dependency) -----------------------------------


def test_embeddings_optional_still_flags_planted_errors() -> None:
    """With embeddings enabled but the dep possibly absent, the three planted
    errors must still come through from the lexical detectors."""
    flags = _flags(config=DetectConfig(enable_embeddings=True))
    assert _covering(flags, "cubernetes")
    assert _covering(flags, "con sensus")
    assert _covering(flags, "cough ka")


def test_embedding_detector_runs_when_available() -> None:
    pytest.importorskip("sentence_transformers")
    from caption_checker.detectors import context_embedding

    cues = parse(DATA_DIR / "sample_lecture.srt")
    words = tokenize(cues)
    from caption_checker.vocab import load_vocab

    flags = context_embedding.find(
        words, cues, load_vocab(), DetectConfig(enable_embeddings=True),
        existing=[],
    )
    assert isinstance(flags, list)
    assert all(f.detector == "context_embedding" for f in flags)
