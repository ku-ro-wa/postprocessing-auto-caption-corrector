"""Character-offset splice: an accepted correction changes exactly its span
and nothing else (ADR-0001)."""

from __future__ import annotations

from pathlib import Path

import pytest

from caption_checker.apply import apply_corrections
from caption_checker.detect import detect
from caption_checker.detectors.base import index_cues, make_flag
from caption_checker.models import DetectConfig, Flag
from caption_checker.parser import parse, serialize, tokenize

DATA_DIR = Path(__file__).parent / "data"


def _flag(cues, substring):
    for flag in detect(cues, config=DetectConfig()):
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


# --- spans across Cues (ADR-0001, amended) -----------------------------------

CROSS_SRT = """1
00:00:00,000 --> 00:00:02,000
we reached con

2
00:00:02,000 --> 00:00:04,000
sensus. Then the

3
00:00:04,000 --> 00:00:06,000
cough

4
00:00:06,000 --> 00:00:08,000
ka broker ran on cubernetes.
"""


@pytest.fixture
def cross_cues(tmp_path: Path):
    path = tmp_path / "cross.srt"
    path.write_text(CROSS_SRT, encoding="utf-8")
    return parse(path)


def _span_flag(cues, text: str) -> Flag:
    """A Flag over the consecutive Words spelling ``text``, wherever they sit."""
    words = tokenize(cues)
    target = text.split()
    for i in range(len(words) - len(target) + 1):
        if [w.text for w in words[i : i + len(target)]] == target:
            return make_flag(
                words[i : i + len(target)],
                index_cues(cues),
                detector="test",
                reason="test",
                confidence=0.9,
            )
    raise AssertionError(f"no Words spelling {text!r}")


def test_a_two_cue_span_goes_into_the_first_cue(cross_cues) -> None:
    flag = _span_flag(cross_cues, "con sensus.")
    corrected = apply_corrections(cross_cues, [(flag, "consensus")])
    texts = [c.text for c in corrected]
    assert texts[0] == "we reached consensus"
    # the span's words leave the second Cue; its punctuation and the rest stay
    assert texts[1] == ". Then the"
    assert [(c.start, c.end) for c in corrected] == [
        (c.start, c.end) for c in cross_cues
    ]


def test_a_span_that_empties_a_cue_removes_it(cross_cues) -> None:
    flag = _span_flag(cross_cues, "the cough ka")
    corrected = apply_corrections(cross_cues, [(flag, "the Kafka")])
    assert [c.index for c in corrected] == [1, 2, 4]
    assert corrected[1].text == "sensus. Then the Kafka"
    assert corrected[2].text == "broker ran on cubernetes."


def test_leading_whitespace_left_by_removed_words_is_tidied(tmp_path: Path) -> None:
    path = tmp_path / "ws.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nit was con\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nsensus  then\nmore\n",
        encoding="utf-8",
    )
    cues = parse(path)
    corrected = apply_corrections(cues, [(_span_flag(cues, "con sensus"), "consensus")])
    assert corrected[1].text == "then\nmore"


def test_corrections_in_one_cue_beside_a_cross_cue_one(cross_cues) -> None:
    pairs = [
        (_span_flag(cross_cues, "cough ka"), "Kafka"),
        (_span_flag(cross_cues, "broker"), "brokers"),
        (_span_flag(cross_cues, "cubernetes."), "Kubernetes"),
        (_span_flag(cross_cues, "reached"), "reach"),
    ]
    corrected = apply_corrections(cross_cues, pairs)
    assert [c.text for c in corrected] == [
        "we reach con",
        "sensus. Then the",
        "Kafka",
        "brokers ran on Kubernetes.",
    ]


def test_srt_round_trip_renumbers_after_a_cue_is_removed(cross_cues, tmp_path: Path) -> None:
    flag = _span_flag(cross_cues, "the cough ka")
    corrected = apply_corrections(cross_cues, [(flag, "the Kafka")])
    out = tmp_path / "out.srt"
    out.write_text(serialize(corrected, format="srt"), encoding="utf-8")

    reparsed = parse(out)
    assert [c.index for c in reparsed] == [1, 2, 3]
    assert [c.text for c in reparsed] == [c.text for c in corrected]
    assert [(c.start, c.end) for c in reparsed] == [(c.start, c.end) for c in corrected]


def test_vtt_round_trip_after_a_cue_is_removed(cross_cues, tmp_path: Path) -> None:
    flag = _span_flag(cross_cues, "the cough ka")
    corrected = apply_corrections(cross_cues, [(flag, "the Kafka")])
    out = tmp_path / "out.vtt"
    out.write_text(serialize(corrected, format="vtt"), encoding="utf-8")
    assert [c.text for c in parse(out)] == [c.text for c in corrected]


def test_an_empty_replacement_in_one_cue_is_not_tidied(cross_cues) -> None:
    corrected = apply_corrections(cross_cues, [(_span_flag(cross_cues, "cough"), "")])
    assert [c.index for c in corrected] == [1, 2, 3, 4]  # only a cross-Cue cut removes a Cue
    assert corrected[2].text == ""


def _cues_from(tmp_path: Path, *texts: str):
    path = tmp_path / "p.srt"
    path.write_text(
        "".join(
            f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\n{t}\n\n"
            for i, t in enumerate(texts, start=1)
        ),
        encoding="utf-8",
    )
    return parse(path)


def _flag_over(cues, first: int, last: int) -> Flag:
    words = tokenize(cues)[first : last + 1]
    return make_flag(words, index_cues(cues), detector="t", reason="t", confidence=1.0)


def test_an_all_punctuation_edge_word_stays_outside_the_span(tmp_path: Path) -> None:
    cues = _cues_from(tmp_path, "say — foo now")
    flag = _flag_over(cues, 1, 2)  # [—, foo]
    assert flag.span == "foo"
    assert apply_corrections(cues, [(flag, "bar")])[0].text == "say — bar now"


def test_punctuation_alone_in_the_last_cue_is_not_cut(tmp_path: Path) -> None:
    cues = _cues_from(tmp_path, "say foo", "… now")
    flag = _flag_over(cues, 1, 2)  # [foo, …]
    assert flag.span == "foo"
    assert [c.text for c in apply_corrections(cues, [(flag, "bar")])] == ["say bar", "… now"]
