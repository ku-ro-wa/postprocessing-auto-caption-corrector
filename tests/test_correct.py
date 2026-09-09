"""The ``run_correction()`` seam. Drive it directly with parsed cues, a
``StubCorrector``, and a scripted reviewer; assert on the corrected cues, the
outcome records, and the number of corrector calls. Never the network."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from caption_checker.cache import CachedCorrection, DecisionCache
from caption_checker.correct import (
    InteractiveReviewer,
    PendingCorrection,
    ReviewDecision,
    ThresholdReviewer,
    run_correction,
    should_bypass,
)
from caption_checker.corrector import StubCorrector
from caption_checker.models import DETECTOR_PHONETIC_INTERNAL, DetectConfig, Flag
from caption_checker.parser import parse

DATA_DIR = Path(__file__).parent / "data"
NO_EMBED = DetectConfig(enable_embeddings=False)

SAMPLE = parse(DATA_DIR / "sample_lecture.srt")


def _cues(text: str, tmp_path: Path, name: str = "t.srt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return parse(path)


class FuncReviewer:
    """Scripted reviewer. ``decide(pending) -> ('accept'|'skip'|'quit', repl)``.
    'quit' stops the loop, leaving the rest undecided (== rejected)."""

    def __init__(self, decide):
        self.decide = decide
        self.seen: list[PendingCorrection] = []

    def review(self, pending):
        out = []
        for p in pending:
            self.seen.append(p)
            action, repl = self.decide(p)
            if action == "quit":
                break
            out.append(ReviewDecision(p, action == "accept", repl))
        return out


ACCEPT_ALL = FuncReviewer(lambda p: ("accept", p.replacement))


def _run(cues, corrector=None, reviewer=None, **kw):
    return run_correction(
        cues,
        corrector=corrector or StubCorrector(),
        reviewer=reviewer or ACCEPT_ALL,
        config=NO_EMBED,
        **kw,
    )


# --- basic pipeline ------------------------------------------------------


def test_accepts_all_corrects_the_planted_errors() -> None:
    result = _run(SAMPLE)
    joined = "\n".join(c.text for c in result.cues)
    assert "consensus algorithms" in joined
    assert "use Kafka for event" in joined
    assert "look at Kubernetes" in joined
    assert {o.outcome for o in result.outcomes} == {"applied"}


def test_outcome_record_keeps_both_confidences() -> None:
    stub = StubCorrector(confidence_for={"cubernetes": 0.42})
    result = _run(SAMPLE, corrector=stub)
    rec = next(o for o in result.outcomes if o.flag.span == "cubernetes")
    assert rec.correction["detector_confidence"] == pytest.approx(0.977)
    assert rec.correction["llm_confidence"] == pytest.approx(0.42)


def test_skip_leaves_span_untouched_and_records_rejected() -> None:
    reviewer = FuncReviewer(
        lambda p: ("skip", None) if p.flag.span == "cubernetes" else ("accept", p.replacement)
    )
    result = _run(SAMPLE, reviewer=reviewer)
    joined = "\n".join(c.text for c in result.cues)
    assert "look at cubernetes" in joined
    assert next(
        o.outcome for o in result.outcomes if o.flag.span == "cubernetes"
    ) == "rejected"


def test_edit_applies_the_edited_text() -> None:
    reviewer = FuncReviewer(
        lambda p: ("accept", "K8s") if p.flag.span == "cubernetes" else ("accept", p.replacement)
    )
    result = _run(SAMPLE, reviewer=reviewer)
    assert "look at K8s and container" in "\n".join(c.text for c in result.cues)


def test_quit_keeps_earlier_accepts_and_rejects_the_rest() -> None:
    # pending is reviewed in transcript order: con sensus, cough ka x2, cubernetes
    def decide(p):
        if p.flag.span == "cough ka":
            return ("quit", None)
        return ("accept", p.replacement)

    result = _run(SAMPLE, reviewer=FuncReviewer(decide))
    joined = "\n".join(c.text for c in result.cues)
    assert "consensus algorithms" in joined  # accepted before quit
    assert "cough ka" in joined  # quit before applying
    outcomes = {o.flag.span: o.outcome for o in result.outcomes}
    assert outcomes["con sensus"] == "applied"
    assert outcomes["cubernetes"] == "rejected"


# --- bypass (#6) -------------------------------------------------------


BYPASS_SRT = (
    "1\n00:00:00,000 --> 00:00:03,000\n"
    "The weather today is calm and clear.\n\n"
    "2\n00:00:03,000 --> 00:00:06,000\n"
    "We watched a wether cross the field.\n"
)


def test_pure_phonetic_internal_single_candidate_bypasses_the_llm(tmp_path) -> None:
    stub = StubCorrector()
    result = _run(_cues(BYPASS_SRT, tmp_path), corrector=stub)

    assert stub.calls == []  # no LLM call
    rec = next(o for o in result.outcomes if o.flag.span == "wether")
    assert rec.outcome == "bypassed"
    assert "watched a weather cross" in "\n".join(c.text for c in result.cues)


def test_bypassed_correction_can_still_be_rejected(tmp_path) -> None:
    reviewer = FuncReviewer(lambda p: ("skip", None))
    result = _run(_cues(BYPASS_SRT, tmp_path), reviewer=reviewer)
    rec = next(o for o in result.outcomes if o.flag.span == "wether")
    assert rec.outcome == "rejected"
    assert "watched a wether cross" in "\n".join(c.text for c in result.cues)


CLOSE_SRT = (
    "1\n00:00:00,000 --> 00:00:03,000\n"
    "The ceiling was high and the sealing was tight.\n\n"
    "2\n00:00:03,000 --> 00:00:06,000\n"
    "He painted the cieling white.\n"
)


def test_two_close_candidates_go_to_the_llm(tmp_path) -> None:
    stub = StubCorrector()
    result = _run(_cues(CLOSE_SRT, tmp_path), corrector=stub)
    assert [fc.span for call in stub.calls for fc in call] == ["cieling"]
    assert next(
        o.outcome for o in result.outcomes if o.flag.span == "cieling"
    ) == "applied"


def test_should_bypass_rules() -> None:
    def flag(detector, candidates):
        return Flag(
            span="cieling",
            global_indices=[0],
            cue_index=1,
            start=SAMPLE[0].start,
            end=SAMPLE[0].end,
            detector=detector,
            reason="r",
            candidates=candidates,
            confidence=0.7,
        )

    one = flag(DETECTOR_PHONETIC_INTERNAL, ["ceiling"])
    close = flag(DETECTOR_PHONETIC_INTERNAL, ["ceiling", "sealing"])
    merged = flag("oov+phonetic_internal", ["ceiling"])

    assert should_bypass(one, NO_EMBED)
    assert not should_bypass(merged, NO_EMBED)
    assert not should_bypass(close, DetectConfig(bypass_jw_margin=0.15))
    assert should_bypass(close, DetectConfig(bypass_jw_margin=0.10))


# --- cache (#5) -------------------------------------------------------


def test_second_run_hits_cache_and_skips_the_llm(tmp_path) -> None:
    cache_path = tmp_path / "c.json"

    first_stub = StubCorrector()
    _run(SAMPLE, corrector=first_stub, cache=DecisionCache.load(cache_path))
    assert len(first_stub.calls) == 1

    second_stub = StubCorrector()
    result = _run(
        SAMPLE, corrector=second_stub, cache=DecisionCache.load(cache_path)
    )
    assert second_stub.calls == []
    assert {o.outcome for o in result.outcomes} == {"cached"}
    assert "look at Kubernetes" in "\n".join(c.text for c in result.cues)


def test_no_cache_forces_the_llm_again_and_does_not_write(tmp_path) -> None:
    cache_path = tmp_path / "c.json"
    _run(SAMPLE, corrector=StubCorrector(), cache=DecisionCache.load(cache_path))

    disabled = DecisionCache.load(cache_path, enabled=False)
    stub = StubCorrector()
    _run(SAMPLE, corrector=stub, cache=disabled)
    assert len(stub.calls) == 1


def test_different_model_misses_the_cache(tmp_path) -> None:
    cache_path = tmp_path / "c.json"
    _run(
        SAMPLE,
        corrector=StubCorrector(),
        cache=DecisionCache.load(cache_path),
        model_id="model-a",
    )
    stub = StubCorrector()
    _run(
        SAMPLE,
        corrector=stub,
        cache=DecisionCache.load(cache_path),
        model_id="model-b",
    )
    assert len(stub.calls) == 1


def test_cached_correction_still_passes_through_review(tmp_path) -> None:
    cache = DecisionCache.load(tmp_path / "c.json")
    cache.set(
        "cubernetes", "test-model", CachedCorrection("Kubernetes", 0.9, "r")
    )
    reviewer = FuncReviewer(lambda p: ("skip", None))
    result = _run(
        SAMPLE, reviewer=reviewer, cache=cache, model_id="test-model"
    )
    assert "look at cubernetes" in "\n".join(c.text for c in result.cues)
    assert next(
        o.outcome for o in result.outcomes if o.flag.span == "cubernetes"
    ) == "rejected"


# --- batching + parse failure (#4) -----------------------------------


def test_flag_set_larger_than_a_chunk_makes_multiple_calls() -> None:
    stub = StubCorrector()
    _run(SAMPLE, corrector=stub, chunk_size=2)
    assert len(stub.calls) == 2


def test_garbage_chunk_is_retried_once_then_skipped() -> None:
    stub = StubCorrector(garbage_spans={"cubernetes"})
    result = _run(SAMPLE, corrector=stub, chunk_size=1)

    # 3 good chunks + 2 attempts at the bad one
    assert len(stub.calls) == 5
    outcomes = {o.flag.span: o.outcome for o in result.outcomes}
    assert outcomes["cubernetes"] == "skipped-parse-failure"
    assert outcomes["con sensus"] == "applied"
    joined = "\n".join(c.text for c in result.cues)
    assert "consensus algorithms" in joined
    assert "look at cubernetes" in joined  # the skipped one is untouched


def test_skipped_flag_has_no_correction_in_the_record() -> None:
    stub = StubCorrector(garbage_spans={"cubernetes"})
    result = _run(SAMPLE, corrector=stub, chunk_size=1)
    rec = next(o for o in result.outcomes if o.flag.span == "cubernetes")
    assert rec.correction is None


def test_max_calls_is_a_hard_ceiling_including_retries() -> None:
    # 4 one-flag chunks; the first fails and burns its retry. --max-calls 4
    # must still never issue a 5th request.
    stub = StubCorrector(garbage_spans={"con sensus"})
    result = _run(SAMPLE, corrector=stub, chunk_size=1, max_calls=4)
    assert len(stub.calls) <= 4
    assert result.corrector_calls <= 4


def test_max_calls_over_chunk_count_completes_normally() -> None:
    stub = StubCorrector()
    result = _run(SAMPLE, corrector=stub, chunk_size=2, max_calls=5)
    assert len(stub.calls) == 2
    assert {o.outcome for o in result.outcomes} == {"applied"}


# --- not-an-error (#4) ----------------------------------------------


def test_null_replacement_is_a_not_an_error_outcome() -> None:
    stub = StubCorrector(null_spans={"cubernetes"})
    result = _run(SAMPLE, corrector=stub, reviewer=ThresholdReviewer(0.5))
    rec = next(o for o in result.outcomes if o.flag.span == "cubernetes")
    assert rec.outcome == "not-an-error"
    assert "look at cubernetes" in "\n".join(c.text for c in result.cues)


def test_not_an_error_defaults_to_skip_but_can_be_overridden() -> None:
    stub = StubCorrector(null_spans={"cubernetes"})

    seen: list[PendingCorrection] = []

    def decide(p):
        seen.append(p)
        if p.flag.span == "cubernetes":
            assert p.preset == "skip"
            return ("accept", "Kubernetes")  # override
        return ("accept", p.replacement)

    result = _run(SAMPLE, corrector=stub, reviewer=FuncReviewer(decide))
    assert "look at Kubernetes" in "\n".join(c.text for c in result.cues)
    assert next(
        o.outcome for o in result.outcomes if o.flag.span == "cubernetes"
    ) == "applied"


# --- interactive reviewer keys (#3) --------------------------------


def _pending(span, repl, conf, preset="accept"):
    flag = Flag(
        span=span,
        global_indices=[0],
        cue_index=1,
        start=SAMPLE[0].start,
        end=SAMPLE[0].end,
        detector="oov",
        reason="reason",
        candidates=[repl] if repl else [],
        confidence=conf,
    )
    return PendingCorrection(
        flag=flag,
        flag_id="f0",
        cue_text=f"line with {span} here",
        replacement=repl,
        source="llm",
        detector_confidence=conf,
        llm_confidence=conf,
        rationale="because",
        preset=preset,
    )


def _review(keys: str, pending):
    reviewer = InteractiveReviewer(
        stdin=io.StringIO(keys), stderr=io.StringIO()
    )
    return reviewer.review(pending)


def test_interactive_yes_no() -> None:
    pending = [_pending("aaa", "AAA", 0.9), _pending("bbb", "BBB", 0.9)]
    decisions = _review("y\nn\n", pending)
    assert [d.accepted for d in decisions] == [True, False]


def test_interactive_edit() -> None:
    decisions = _review("e\nEDITED\n", [_pending("aaa", "AAA", 0.9)])
    assert decisions[0].accepted
    assert decisions[0].replacement == "EDITED"


def test_interactive_accept_all_above_confidence() -> None:
    pending = [
        _pending("hi", "HI", 0.9),
        _pending("mid", "MID", 0.95),
        _pending("lo", "LO", 0.6),
    ]
    decisions = _review("a\nn\n", pending)
    assert [d.accepted for d in decisions] == [True, True, False]


def test_interactive_quit_writes_what_was_accepted() -> None:
    pending = [_pending("a", "A", 0.9), _pending("b", "B", 0.9), _pending("c", "C", 0.9)]
    decisions = _review("y\nq\n", pending)
    assert len(decisions) == 1 and decisions[0].accepted


def test_interactive_preset_skip_defaults_to_n() -> None:
    pending = [_pending("x", None, 0.8, preset="skip")]
    decisions = _review("\n", pending)  # bare enter -> default
    assert decisions[0].accepted is False


def test_interactive_y_on_not_an_error_asks_for_the_correction() -> None:
    # no replacement to "accept", so y (like e) prompts for the text
    pending = [_pending("x", None, 0.8, preset="skip")]
    decisions = _review("y\nKubernetes\n", pending)
    assert decisions[0].accepted is True
    assert decisions[0].replacement == "Kubernetes"


def test_interactive_n_on_not_an_error_leaves_it_alone() -> None:
    pending = [_pending("x", None, 0.8, preset="skip")]
    decisions = _review("n\n", pending)
    assert decisions[0].accepted is False
