"""Character-offset splice: an accepted correction changes exactly its span
and nothing else (ADR-0001)."""

from __future__ import annotations

from pathlib import Path

import pytest

from caption_checker.apply import apply_corrections
from caption_checker.detect import detect
from caption_checker.models import DetectConfig
from caption_checker.parser import parse, serialize

DATA_DIR = Path(__file__).parent / "data"
NO_EMBED = DetectConfig(enable_embeddings=False)


def _flag(cues, substring):
    for flag in detect(cues, config=NO_EMBED):
        if substring.lower() in flag.span.lower():
            return flag
    raise AssertionError(f"no flag covering {substring!r}")


@pytest.mark.parametrize(
    "filename,fmt", [("sample_lecture.srt", "srt"), ("sample_lecture.vtt", "vtt")]
)
def test_single_token_splice_preserves_every_other_byte(filename, fmt) -> None:
    cues = parse(DATA_DIR / filename)
    flag = _flag(cues, "cubernetes")

    corrected = apply_corrections(cues, [(flag, "Kubernetes")])

    baseline = serialize(cues, format=fmt)
    got = serialize(corrected, format=fmt)
    assert got == baseline.replace("cubernetes", "Kubernetes")
    assert got.count("Kubernetes") == 1


@pytest.mark.parametrize(
    "filename,fmt", [("sample_lecture.srt", "srt"), ("sample_lecture.vtt", "vtt")]
)
def test_multi_token_span_collapses_to_one_word(filename, fmt) -> None:
    cues = parse(DATA_DIR / filename)
    flag = _flag(cues, "con sensus")
    assert len(flag.global_indices) == 2

    corrected = apply_corrections(cues, [(flag, "consensus")])
    got = serialize(corrected, format=fmt)

    baseline = serialize(cues, format=fmt)
    assert got == baseline.replace("con sensus", "consensus")
    # the surrounding words are untouched
    assert "talk about consensus algorithms" in got


def test_trailing_punctuation_outside_span_is_kept() -> None:
    cues = parse(DATA_DIR / "sample_lecture.srt")
    flag = _flag(cues, "cough ka")  # cue 5 text: "... cough ka, which ..."

    corrected = apply_corrections(cues, [(flag, "Kafka")])
    text = next(c.text for c in corrected if c.index == flag.cue_index)
    assert "talk about Kafka, which is" in text


def test_multiple_corrections_in_one_run() -> None:
    cues = parse(DATA_DIR / "sample_lecture.srt")
    pairs = [
        (_flag(cues, "con sensus"), "consensus"),
        (_flag(cues, "cubernetes"), "Kubernetes"),
    ]
    got = serialize(apply_corrections(cues, pairs), format="srt")
    baseline = serialize(cues, format="srt")
    assert got == baseline.replace("con sensus", "consensus").replace(
        "cubernetes", "Kubernetes"
    )


def test_no_corrections_returns_input_unchanged() -> None:
    cues = parse(DATA_DIR / "sample_lecture.srt")
    assert apply_corrections(cues, []) == cues
