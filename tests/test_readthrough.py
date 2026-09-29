"""The Read-through seam: ``read_through()`` driven with parsed cues, local
Flags as hints, and a ``StubReader``; plus the reply parser. Never the
network."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from caption_checker.corrector import CorrectorError, RequestError, Spend
from caption_checker.detect import detect
from caption_checker.parser import parse
from caption_checker.readthrough import (
    CONFIGS,
    DETECTOR_READ_THROUGH,
    ChunkRequest,
    ChunkVerdict,
    Hint,
    OpenRouterReader,
    ReadThroughConfig,
    StubReader,
    build_messages,
    parse_keyed_reply,
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


@pytest.mark.parametrize(
    ("span", "replacement"),
    [
        ("con sensus", "consensus"),  # a word split in two
        ("AIdriven", "AI-driven"),  # two words run together
        ("anthropics", "Anthropic's"),
    ],
)
def test_parse_reply_keeps_a_hint_correction_that_moves_word_boundaries(
    span, replacement
) -> None:
    # Same letters, but not the same words: a reader sees the difference, so
    # this is a Correction, not a formatting-only echo of the span.
    tokens = ["The", *span.split(), "layer."]
    end = len(tokens) - 2
    hint = Hint(
        id="h0", start=1, end=end, span=span, candidates=[], reason="not a known word"
    )
    request = ChunkRequest(words=list(enumerate(tokens)), hints=[hint])
    reply = json.dumps([[1, end, span, replacement, "misheard", 0.9, "h0", "x"]])
    [v] = parse_reply(reply, request)
    assert v.replacement == replacement


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


# --- keyed replies: hint verdicts by hint id, new finds apart ----------------


def test_parse_keyed_reply_reads_hint_verdicts_and_new_finds(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {
            "hints": {"h0": [4, 4, "cubernetes", "Kubernetes", "misheard", 0.95, "platform"]},
            "errors": [[6, 7, "chad GPT", "ChatGPT", "misheard", 0.9, "product"]],
        }
    )
    hint, found = parse_keyed_reply(reply, chunk)
    assert (hint.hint, hint.start, hint.replacement, hint.confidence) == (
        "h0", 4, "Kubernetes", 0.95,
    )
    assert (found.hint, found.start, found.end, found.replacement, found.rationale) == (
        None, 6, 7, "ChatGPT", "product",
    )


def test_parse_keyed_reply_answers_a_hint_not_an_error(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {"hints": {"h0": [4, 4, "cubernetes", None, "misheard", 0.8, "a name"]},
         "errors": []}
    )
    [v] = parse_keyed_reply(reply, chunk)
    assert (v.hint, v.replacement) == ("h0", None)


@pytest.mark.parametrize(
    "reply",
    [
        "sorry, no",  # not JSON
        "[]",  # not the keyed object
        '{"hints": [], "errors": []}',  # hints not keyed by id
        '{"hints": {}, "errors": {}}',  # errors not a list
        '{"hints": {}, "errors": []}',  # h0 has no verdict
        '{"hints": {"h0": "Kubernetes"}, "errors": []}',  # h0's verdict unreadable
    ],
)
def test_parse_keyed_reply_fails_the_chunk_on_a_malformed_reply(cues, reply) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    with pytest.raises(CorrectorError):
        parse_keyed_reply(reply, chunk)


@pytest.mark.parametrize(
    ("span", "replacement"),
    [("con sensus", "consensus"), ("AIdriven", "AI-driven")],
)
def test_parse_keyed_reply_keeps_a_hint_correction_that_moves_word_boundaries(
    span, replacement
) -> None:
    tokens = ["The", *span.split(), "layer."]
    end = len(tokens) - 2
    hint = Hint(
        id="h0", start=1, end=end, span=span, candidates=[], reason="not a known word"
    )
    request = ChunkRequest(words=list(enumerate(tokens)), hints=[hint])
    reply = json.dumps(
        {"hints": {"h0": [1, end, span, replacement, "misheard", 0.9, "x"]}, "errors": []}
    )
    [v] = parse_keyed_reply(reply, request)
    assert v.replacement == replacement


def test_parse_keyed_reply_fails_the_chunk_on_a_hint_verdict_of_the_wrong_shape(
    cues,
) -> None:
    # An extra field would shift every field after it -- the hint's verdict
    # must not quietly become "not an error".
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {"hints": {"h0": [4, 4, "cubernetes", "Kubernetes", None, "misheard", 0.9, "x"]},
         "errors": []}
    )
    with pytest.raises(CorrectorError, match="h0"):
        parse_keyed_reply(reply, chunk)


def test_parse_keyed_reply_drops_a_new_find_of_the_wrong_shape(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {"hints": {"h0": [4, 4, "cubernetes", "Kubernetes", "misheard", 0.9, "x"]},
         "errors": [[11, 11, "Hang", "Huang", None, "misheard", 0.9, "x"]]}
    )
    assert [v.hint for v in parse_keyed_reply(reply, chunk)] == ["h0"]


class _KeyedReader:
    """A Reader whose model answers in the keyed format, read by the keyed
    parser -- the Read-through as a keyed configuration runs it."""

    spend = Spend()

    def __init__(self, reply: dict) -> None:
        self.reply = json.dumps(reply)

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
        return parse_keyed_reply(self.reply, request)


def test_a_keyed_reply_places_corrections_across_a_cue_boundary(cues) -> None:
    # #26: a hint widened over a Cue boundary, and a new find across one,
    # keep their spans.
    flags = [f for f in detect(cues) if f.span == "cubernetes"]
    reader = _KeyedReader(
        {"hints": {"h0": [4, 5, "cubernetes. The", "Kubernetes. The", "misheard", 0.9, "x"]},
         "errors": [[10, 11, "fast. Hang", "fast. Huang", "misheard", 0.95, "a name"]]}
    )
    hint, found = read_through(cues, flags, reader).items
    assert hint.hint is flags[0]
    assert hint.flag.global_indices == [4, 5]
    assert hint.correction.replacement == "Kubernetes. The"
    assert found.flag.global_indices == [10, 11]
    assert (found.flag.start, found.flag.end) == (cues[1].start, cues[2].end)
    assert found.correction.replacement == "fast. Huang"


def test_parse_keyed_reply_ignores_a_verdict_for_an_unknown_hint(cues) -> None:
    [chunk] = plan_chunks(cues, detect(cues))
    reply = json.dumps(
        {"hints": {"h0": [4, 4, "cubernetes", "Kubernetes", "misheard", 0.9, "x"],
                   "h9": [11, 11, "Hang", "Huang", "misheard", 0.9, "x"]},
         "errors": []}
    )
    assert [v.hint for v in parse_keyed_reply(reply, chunk)] == ["h0"]


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


def test_a_chunk_read_on_its_retry_is_recovered_not_failed(cues) -> None:
    class BadFirstReply(StubReader):
        def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
            if self.calls == 0:
                self.calls += 1
                raise CorrectorError("reply is not JSON")
            return super().read(request)

    result = read_through(cues, detect(cues), BadFirstReply())
    assert result.failed_chunks == 0
    assert result.recovered_chunks == 1
    [item] = result.items
    assert item.correction is not None


def test_a_chunk_that_fails_twice_is_not_recovered(cues) -> None:
    result = read_through(cues, detect(cues), StubReader(garbage=True))
    assert result.recovered_chunks == 0


def test_a_chunk_whose_requests_fail_is_counted_as_a_request_failure(cues) -> None:
    # A request error (no credit, network) says nothing about the model's
    # reply format, so eval must be able to tell the two apart.
    reader = StubReader(request_error="HTTP 402 Payment Required: no credit")
    result = read_through(cues, detect(cues), reader)
    assert result.failed_chunks == 1
    assert result.request_failed_chunks == 1
    assert result.last_request_error == "HTTP 402 Payment Required: no credit"


def test_an_unreadable_reply_is_not_a_request_failure(cues) -> None:
    result = read_through(cues, detect(cues), StubReader(garbage=True))
    assert result.failed_chunks == 1
    assert result.request_failed_chunks == 0
    assert result.last_request_error is None


def test_a_chunk_with_any_unreadable_reply_is_not_a_request_failure(cues) -> None:
    # One unreadable reply is enough to show the model's format is at
    # fault, whichever attempt it came on.
    class OneOfEach(StubReader):
        def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
            self.calls += 1
            if self.calls == 1:
                raise RequestError("HTTP 402 Payment Required")
            raise CorrectorError("reply is not JSON")

    result = read_through(cues, detect(cues), OneOfEach())
    assert result.failed_chunks == 1
    assert result.request_failed_chunks == 0


def test_max_calls_caps_requests_including_retries(cues) -> None:
    reader = StubReader(garbage=True)
    read_through(cues, [], reader, chunk_words=4, max_calls=3)
    assert reader.calls == 3


def test_openrouter_reader_builds_and_reads_as_its_configuration_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    verdict = ChunkVerdict(start=0, end=0, replacement="X", confidence=1.0)
    config = ReadThroughConfig(
        name="test",
        model_id="some/model",
        build_messages=lambda request: [{"role": "user", "content": "custom"}],
        parse_reply=lambda content, request: [verdict] if content == "reply" else [],
    )
    reader = OpenRouterReader(config)
    sent: list[list[dict]] = []

    def chat(messages: list[dict], **options: object) -> str:
        sent.append(messages)
        return "reply"

    monkeypatch.setattr(reader.client, "chat", chat)
    assert reader.client.model_id == "some/model"
    assert reader.read(ChunkRequest(words=[(0, "a")])) == [verdict]
    assert sent == [[{"role": "user", "content": "custom"}]]


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("flash-v4", {}),  # today's request, unchanged
        ("gemini-3.8-flash-v4", {"reasoning": {"effort": "minimal"}}),
        ("deepseek-v4-pro-v4", {"reasoning": {"enabled": False}}),
        ("qwen3.6-plus-v4", {"reasoning": {"enabled": False}}),
        ("gpt-5.6-luna-v4", {"reasoning": {"enabled": False}}),
    ],
)
def test_openrouter_reader_sends_its_configurations_request_options(
    monkeypatch: pytest.MonkeyPatch, name: str, extra: dict
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    reader = OpenRouterReader(CONFIGS[name])
    sent: list[dict] = []

    def chat(messages: list[dict], **options: object) -> str:
        sent.append(options)
        return "[]"

    monkeypatch.setattr(reader.client, "chat", chat)
    reader.read(ChunkRequest(words=[(0, "a")]))
    assert sent == [{"timeout": 180, "response_format": {"type": "json_object"}, **extra}]


def test_flash_v4_sends_the_messages_prompt_v4_was_scored_with() -> None:
    # A snapshot of build_messages at f333419, before configurations existed:
    # flash-v4 is frozen, so a prompt change belongs in a new configuration.
    request = ChunkRequest(
        words=[(3, "we"), (4, "run"), (5, "cubernetes.")],
        hints=[Hint("h0", 5, 5, "cubernetes.", ["Kubernetes"], "not in vocabulary")],
        before="So today",
        after="and then",
        priming_terms=["Kafka", "Raft"],
    )
    messages = CONFIGS["flash-v4"].build_messages(request)
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
    assert digest == "54ca44927fe372376de4924ceb14445a67efd7bc2d3051642eb09c799f5f5ed5"


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_configuration_keeps_v4s_task_and_hint_format(name: str) -> None:
    # #28: a configuration may change only what counts as an error, the reply
    # format and brevity -- so the task, and the chunk as the model sees it,
    # are v4's.
    request = ChunkRequest(
        words=[(3, "we"), (4, "run"), (5, "cubernetes.")],
        hints=[Hint("h0", 5, 5, "cubernetes.", ["Kubernetes"], "not in vocabulary")],
        priming_terms=["Kafka"],
    )
    system, user = CONFIGS[name].build_messages(request)
    v4_system, v4_user = build_messages(request)
    assert user == v4_user
    task = v4_system["content"].split("Find every error")[0]
    assert system["content"].startswith(task)
