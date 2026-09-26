"""The Read-through seam: ``read_through()`` driven with parsed cues, local
Flags as hints, and a ``StubReader``; plus the reply parser. Never the
network."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from caption_checker.corrector import CorrectorError
from caption_checker.detect import detect
from caption_checker.parser import parse
from caption_checker.readthrough import (
    DETECTOR_READ_THROUGH,
    ChunkRequest,
    StubReader,
    build_messages,
    parse_reply,
    plan_chunks,
    read_through,
)

SRT = """1
00:00:00,000 --> 00:00:03,000
Today we deploy on cubernetes.

2
00:00:03,000 --> 00:00:06,000
The chad GPT model is fast.

3
00:00:06,000 --> 00:00:09,000
Hang the CEO spoke at length.
"""


@pytest.fixture
def cues(tmp_path: Path):
    path = tmp_path / "t.srt"
    path.write_text(SRT, encoding="utf-8")
    return parse(path)


def _request(**kw) -> ChunkRequest:
    words = [(0, "Today"), (1, "we"), (2, "deploy"), (3, "on"), (4, "cubernetes.")]
    return ChunkRequest(words=words, hints=kw.pop("hints", []), **kw)


# --- chunking ------------------------------------------------------------


def test_chunks_cover_every_word_once_and_break_after_sentences(cues) -> None:
    chunks = plan_chunks(cues, [], chunk_words=4)
    indices = [gi for c in chunks for gi, _ in c.words]
    assert indices == list(range(17))
    # the first break waits for the sentence end at "cubernetes." (word 4)
    assert chunks[0].words[-1] == (4, "cubernetes.")
    assert chunks[1].before.endswith("cubernetes.")


def test_hints_ride_with_the_chunk_their_first_word_is_in(cues) -> None:
    flags = detect(cues)
    chunks = plan_chunks(cues, flags, chunk_words=4)
    assert [h.span for h in chunks[0].hints] == ["cubernetes"]
    assert chunks[1].hints == [] and chunks[2].hints == []


def test_priming_terms_reach_the_prompt(cues) -> None:
    [chunk] = plan_chunks(cues, [], priming_terms=["Jensen Huang"])
    system, user = build_messages(chunk)
    assert "Jensen Huang" in user["content"]
    assert "[4]cubernetes." in user["content"]


# --- reply parsing ---------------------------------------------------------


def test_parse_reply_reads_verdicts_with_word_ranges() -> None:
    reply = json.dumps(
        {"verdicts": [{"start": 4, "end": 4, "span": "cubernetes.",
                       "replacement": "Kubernetes", "confidence": 0.9,
                       "rationale": "platform"}]}
    )
    [v] = parse_reply(reply, _request())
    assert (v.start, v.end, v.replacement, v.hint) == (4, 4, "Kubernetes", None)


def test_parse_reply_reads_compact_positional_verdicts(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {"verdicts": [
            [4, 4, "cubernetes", "Kubernetes", "misheard", 0.95, "h0", "platform"],
            [6, 7, "chad GPT", "ChatGPT", "misheard", 0.9, None, "product"],
        ]}
    )
    hint, found = parse_reply(reply, chunk)
    assert (hint.hint, hint.replacement, hint.confidence) == ("h0", "Kubernetes", 0.95)
    assert (found.start, found.end, found.rationale) == (6, 7, "product")


def test_parse_reply_accepts_a_bare_array_in_a_code_fence() -> None:
    reply = '```json\n[{"start": 2, "end": 2, "replacement": "employ"}]\n```'
    assert parse_reply(reply, _request())[0].replacement == "employ"


def test_parse_reply_relocates_a_range_that_disagrees_with_its_span() -> None:
    reply = json.dumps([{"start": 1, "end": 1, "span": "cubernetes", "replacement": "K"}])
    [v] = parse_reply(reply, _request())
    assert (v.start, v.end) == (4, 4)


def test_parse_reply_drops_an_item_it_cannot_place() -> None:
    reply = json.dumps([{"start": 90, "end": 91, "span": "nowhere", "replacement": "x"}])
    assert parse_reply(reply, _request()) == []


def test_parse_reply_treats_a_formatting_only_change_as_no_error(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        [
            {"start": 4, "end": 4, "span": "cubernetes", "replacement": "Cubernetes",
             "hint": "h0"},
            {"start": 6, "end": 7, "span": "chad GPT", "replacement": "Chad-GPT."},
        ]
    )
    [v] = parse_reply(reply, chunk)  # the new find is dropped as a no-op
    assert v.hint == "h0" and v.replacement is None


def test_parse_reply_keeps_only_misheard_causes(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        [
            {"start": 4, "end": 4, "span": "cubernetes", "replacement": "Kubernetes",
             "cause": "style", "hint": "h0"},
            {"start": 12, "end": 12, "span": "the", "replacement": "a", "cause": "grammar"},
            {"start": 11, "end": 11, "span": "Hang", "replacement": "Huang",
             "cause": "misheard"},
        ]
    )
    hint, found = parse_reply(reply, chunk)
    assert hint.replacement is None  # the speaker's own slip is not an ASR error
    assert (found.start, found.replacement) == (11, "Huang")


def test_parse_reply_rejects_non_json() -> None:
    with pytest.raises(CorrectorError):
        parse_reply("sorry, no", _request())


def test_parse_reply_requires_a_verdict_for_every_hint(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    with pytest.raises(CorrectorError, match="hint"):
        parse_reply("[]", chunk)


# --- the orchestrator ---------------------------------------------------------


def test_hints_get_verdicts_and_new_errors_become_flags(cues) -> None:
    reader = StubReader(extra={"chad GPT": "ChatGPT", "Hang": "Huang"})
    result = read_through(cues, detect(cues), reader)
    by_span = {i.flag.span: i for i in result.items}
    assert by_span["cubernetes"].correction.replacement == "Kubernetes"
    assert by_span["cubernetes"].flag.detector != DETECTOR_READ_THROUGH  # the hint
    chad = by_span["chad GPT"]
    assert chad.flag.detector == DETECTOR_READ_THROUGH
    assert chad.flag.global_indices == [6, 7]
    assert chad.flag.cue_index == 2
    assert chad.flag.context == "The chad GPT model is fast."
    assert chad.correction.replacement == "ChatGPT"
    assert result.calls == 1


def test_a_hint_widened_by_the_model_keeps_its_detector(cues) -> None:
    flags = [f for f in detect(cues) if f.span == "cubernetes"]
    reader = StubReader(
        widen={"cubernetes": "on cubernetes"},
        replacement_for={"on cubernetes": "on Kubernetes"},
    )
    [item] = read_through(cues, flags, reader).items
    assert item.flag.span == "on cubernetes"
    assert item.flag.detector == f"{flags[0].detector}+{DETECTOR_READ_THROUGH}"


def test_a_not_an_error_verdict_is_kept_on_its_hint(cues) -> None:
    reader = StubReader(null_spans={"cubernetes"})
    [item] = read_through(cues, detect(cues), reader).items
    assert item.correction.replacement is None


def test_a_new_find_across_cues_is_kept_on_its_first_cue(cues) -> None:
    reader = StubReader(extra={"fast. Hang": "fast. Huang"})
    [item] = read_through(cues, [], reader).items
    assert item.flag.span == "fast. Hang"
    assert item.flag.global_indices == [10, 11]
    assert item.flag.cue_index == 2
    assert (item.flag.start, item.flag.end) == (cues[1].start, cues[2].end)
    # the review context runs across the boundary
    assert item.flag.context == (
        "The chad GPT model is fast. Hang the CEO spoke at length."
    )
    assert item.correction.replacement == "fast. Huang"


def test_overlapping_verdicts_keep_the_more_confident_one(cues) -> None:
    reader = StubReader(
        extra={"chad GPT": "ChatGPT", "GPT model": "GPT-4 model"},
        confidence_for={"GPT model": 0.4},
    )
    [item] = read_through(cues, [], reader).items
    assert item.flag.span == "chad GPT"


def test_a_hint_outranks_a_more_confident_new_find_over_it(cues) -> None:
    reader = StubReader(extra={"on cubernetes": "on Kubernetes"}, confidence_for={
        "on cubernetes": 0.99, "cubernetes": 0.9})
    [item] = read_through(cues, detect(cues), reader).items
    assert item.flag.span == "cubernetes"  # the hint keeps its verdict


def test_a_hint_widened_across_cues_keeps_the_wider_span(cues) -> None:
    flags = [f for f in detect(cues) if f.span == "cubernetes"]
    reader = StubReader(
        widen={"cubernetes": "cubernetes. The"},
        replacement_for={"cubernetes. The": "Kubernetes. The"},
    )
    [item] = read_through(cues, flags, reader).items
    assert item.hint is flags[0]
    assert item.flag.global_indices == [4, 5]
    assert item.flag.cue_index == 1
    assert item.flag.detector == f"{flags[0].detector}+{DETECTOR_READ_THROUGH}"
    assert "cubernetes. The" in item.flag.context
    assert item.correction.replacement == "Kubernetes. The"


def test_a_hint_cannot_widen_over_another_hint(cues) -> None:
    [kube] = detect(cues)
    deploy = replace(kube, span="deploy", global_indices=[2], candidates=["employ"])
    reader = StubReader(widen={"deploy": "deploy on cubernetes"})
    result = read_through(cues, [deploy, kube], reader)
    assert [i.flag for i in result.items] == [deploy, kube]  # each on its own span


def test_new_finds_below_the_confidence_floor_are_dropped_but_hints_kept(cues) -> None:
    reader = StubReader(
        extra={"chad GPT": "ChatGPT", "Hang": "Huang"},
        confidence_for={"Hang": 0.5, "cubernetes": 0.5},
    )
    result = read_through(cues, detect(cues), reader, min_confidence=0.9)
    assert [i.flag.span for i in result.items] == ["cubernetes", "chad GPT"]


def test_a_chunk_is_retried_once_then_its_hints_fail(cues) -> None:
    reader = StubReader(garbage=True)
    result = read_through(cues, detect(cues), reader)
    assert reader.calls == 2
    assert result.failed_chunks == 1
    [item] = result.items
    assert item.flag.span == "cubernetes" and item.correction is None


def test_max_calls_caps_requests_including_retries(cues) -> None:
    reader = StubReader(garbage=True)
    read_through(cues, [], reader, chunk_words=4, max_calls=3)
    assert reader.calls == 3
