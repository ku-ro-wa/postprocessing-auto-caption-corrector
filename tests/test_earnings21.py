"""The Earnings-21 Auto-labelled corpus importer: Google's ASR output turned
into SRT, aligned against Rev's verbatim reference, and each disagreement
classified into a Scored corpus case. Offline -- the fetcher is faked."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from click.testing import CliRunner

from caption_checker import earnings21
from caption_checker.cli import main
from caption_checker.earnings21 import (
    build,
    build_call,
    parse_nlp,
    to_cues,
)
from caption_checker.evaluation import ScoredCase, _locate
from caption_checker.models import Cue
from caption_checker.parser import parse, serialize, tokenize

def parse_srt_text(text: str) -> list[Cue]:
    """Read serialised SRT back the way the eval command will."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.srt"
        path.write_text(text, encoding="utf-8")
        return parse(path)


HYP_HEADER = "token|speaker|ts|endTs|punctuation|case|tags"
REF_HEADER = "token|speaker|ts|endTs|punctuation|case|tags|wer_tags"


def _hyp(*rows: str) -> str:
    """Google rows as ``token speaker ts end [punct]``."""
    lines = [HYP_HEADER]
    for row in rows:
        token, speaker, ts, end, *punct = row.split()
        lines.append(f"{token}|{speaker}|{ts}|{end}|{''.join(punct)}||")
    return "\n".join(lines) + "\n"


def _hyp_words(text: str, speaker: str = "1") -> str:
    """A single-speaker hypothesis, one token per 0.2s."""
    rows = [
        f"{w} {speaker} {i * 0.2:.1f} {i * 0.2 + 0.2:.1f}"
        for i, w in enumerate(text.split())
    ]
    return _hyp(*rows)


def _ref_words(text: str) -> str:
    """Reference rows; ``word@3`` carries wer_tag id 3."""
    lines = [REF_HEADER]
    for w in text.split():
        token, _, tag = w.partition("@")
        tags = f"['{tag}']" if tag else "[]"
        lines.append(f"{token}|0||||LC|[]|{tags}")
    return "\n".join(lines) + "\n"


# --- parsing ---------------------------------------------------------------


def test_parse_nlp_reads_google_tokens_with_timings_and_punctuation() -> None:
    tokens = parse_nlp(_hyp("Good 1 3.2 3.4", "morning 1 3.4 3.7 ,"))
    assert [t.text for t in tokens] == ["Good", "morning,"]
    assert tokens[1].word == "morning"
    assert tokens[1].start == 3.4 and tokens[1].end == 3.7
    assert tokens[0].speaker == "1"


def test_parse_nlp_resolves_reference_wer_tags_to_entity_types() -> None:
    tokens = parse_nlp(_ref_words("the Monro@5 Inc@5"), {"5": {"entity_type": "ORG"}})
    assert [t.entity_types for t in tokens] == [frozenset(), {"ORG"}, {"ORG"}]


def test_parse_nlp_drops_reference_non_speech_markers() -> None:
    tokens = parse_nlp(_ref_words("so <crosstalk> * yes"))
    assert [t.word for t in tokens] == ["so", "yes"]


def test_parse_nlp_tolerates_a_misplaced_column() -> None:
    # Seen in call 4346923's reference: punctuation shifted into endTs.
    text = REF_HEADER + "\nplants.|3||.||LC|[]|[]\n"
    [token] = parse_nlp(text)
    assert (token.word, token.end) == ("plants.", None)


# --- hypothesis -> cues ----------------------------------------------------


def test_to_cues_breaks_on_sentence_end_and_speaker_change() -> None:
    tokens = parse_nlp(
        _hyp(
            "Hello 1 0.0 0.5",
            "there 1 0.5 1.0 .",
            "Next 1 1.0 1.5",
            "one 1 1.5 2.0",
            "Hi 2 2.0 2.5",
        )
    )
    cues = to_cues(tokens)
    assert [c.text for c in cues] == ["Hello there.", "Next one", "Hi"]
    assert cues[0].start == timedelta(seconds=0) and cues[0].end == timedelta(seconds=1)
    assert [c.index for c in cues] == [1, 2, 3]


def test_to_cues_breaks_long_runs_by_duration_and_length() -> None:
    slow = parse_nlp(_hyp(*(f"w{i} 1 {i * 2}.0 {i * 2 + 1}.0" for i in range(6))))
    assert all(c.end - c.start <= timedelta(seconds=7) for c in to_cues(slow))

    wordy = parse_nlp(_hyp_words(" ".join(["abcdefghij"] * 10)))
    assert all(len(c.text) <= 42 for c in to_cues(wordy))


def test_to_cues_never_emits_a_zero_length_cue() -> None:
    # A zero-length cue is dropped by the SRT round trip, shifting every
    # later Word's global index out from under the cases.
    tokens = parse_nlp(_hyp("are 1 5.0 5.0 .", "And 1 5.0 5.2"))
    cues = to_cues(tokens)
    assert all(c.end > c.start for c in cues)
    assert [w.text for w in tokenize(parse_srt_text(serialize(cues, "srt")))] == [
        "are.",
        "And",
    ]


def test_to_cues_keeps_word_order_when_timings_go_backwards() -> None:
    # Call 4344866's Google output repeats its closing sentence with the
    # original timings; SRT serialisation sorts cues by start time.
    tokens = parse_nlp(
        _hyp("You 1 10.0 10.5", "go. 1 10.5 11.0", "Bye. 2 12.0 13.0", "You 1 10.0 10.5", "go. 1 10.5 11.0")
    )
    cues = to_cues(tokens)
    assert all(a.start < b.start for a, b in zip(cues, cues[1:]))
    words = tokenize(parse_srt_text(serialize(cues, "srt")))
    assert [w.text for w in words] == ["You", "go.", "Bye.", "You", "go."]


# --- alignment + classification --------------------------------------------

TAGS = {
    "1": {"entity_type": "ORG"},
    "2": {"entity_type": "CARDINAL"},
    "3": {"entity_type": "PERSON"},
}


def _build(hyp: str, ref: str, tags: dict | None = None):
    return build_call(
        "call.srt", parse_nlp(_hyp_words(hyp)), parse_nlp(_ref_words(ref), tags or TAGS)
    )


def _cases(hyp: str, ref: str, tags: dict | None = None) -> list[dict]:
    return _build(hyp, ref, tags).cases


def test_real_word_substitution_on_an_entity_is_flagged_as_entity() -> None:
    [case] = _cases(
        "welcome to the Monroe inks earnings call",
        "welcome to the Monro@1 Inc@1 earnings call",
    )
    assert case["span"] == "Monroe inks"
    assert case["candidate"] == "Monro Inc"
    assert case["verdict"] == "should-flag"
    assert case["kind"] == "real-word"
    assert case["entity"] is True
    assert case["source"] == "call.srt"


def test_plain_real_word_substitution_is_not_an_entity() -> None:
    [case] = _cases("our cops were up", "our comps were up")
    assert (case["span"], case["kind"], case["entity"]) == ("cops", "real-word", False)


def test_a_zipf_zero_hypothesis_word_is_a_non_word() -> None:
    [case] = _cases("adjusted ebidda rose", "adjusted EBITDA rose")
    assert (case["span"], case["candidate"], case["kind"]) == ("ebidda", "EBITDA", "non-word")


def test_stopword_only_regions_are_function_words() -> None:
    [case] = _cases("we saw a gain", "we saw the gain")
    assert case["kind"] == "function-word"


def test_format_regions_by_tag_digits_or_spacing() -> None:
    assert _cases("sales of four units", "sales of 4@2 units")[0]["kind"] == "format"
    assert _cases("sales of for units", "sales of four@2 units")[0]["kind"] == "format"
    assert _cases("up by 5 points", "up by five points")[0]["kind"] == "format"
    assert _cases("our e-commerce site", "our ecommerce site")[0]["kind"] == "format"
    assert _cases("our ecommerce site", "our e commerce site")[0]["kind"] == "format"


def test_hyphen_and_case_and_punctuation_differences_are_not_errors() -> None:
    assert _cases("a Long-term, view", "a long term view") == []


def test_deletions_fillers_and_drift_are_dropped_and_counted() -> None:
    call = _build(
        "so we grew and margins rose okay",
        "so uh we grew very fast and margins rose okay",
    )
    assert call.cases == []
    assert call.stats["deletion"] == 2  # "uh" and "very fast" -- nothing on screen
    drift = _build(
        "so we are about to begin now",
        "so Morning Bret Morning Bret Hey uh thanks now",
    )
    assert drift.cases == []
    assert drift.stats["drift"] == 1


def test_filler_only_insertions_are_dropped() -> None:
    call = _build("so um we grew", "so we grew")
    assert call.cases == []
    assert call.stats["filler"] == 1


def test_insertions_keep_the_hypothesis_span_without_a_candidate() -> None:
    [case] = _cases("revenue grew strongly nicely this quarter", "revenue grew strongly this quarter")
    assert case["span"] == "nicely"
    assert case.get("candidate") is None
    assert case["kind"] == "real-word"


def test_context_is_the_shortest_window_unique_in_the_file() -> None:
    call = _build(
        "our cops rose and our costs rose and our cops fell",
        "our comps rose and our costs rose and our comps fell",
    )
    first, second = call.cases
    assert first["context"] == "cops rose"
    assert second["context"] == "cops fell"
    words = tokenize(call.cues)
    for case in call.cases:
        [occurrence] = _locate(ScoredCase(**case), words)
        assert len(occurrence) == 1


def test_every_case_locates_in_the_generated_cues() -> None:
    call = _build(
        "Good morning. Welcome to the Monroe inks call. The cops rose.",
        "Good morning. Welcome to the Monro@1 Inc@1 call. The comps rose.",
    )
    assert [c["span"] for c in call.cases] == ["Monroe inks", "cops"]
    words = tokenize(call.cues)
    for case in call.cases:
        assert _locate(ScoredCase(**case), words)


# --- building the corpora --------------------------------------------------

CALL_IDS = ("100", "200", "300", "400", "500", "600", "700", "800", "900")
_COLUMNS = "file_id,audio_length,sample_rate,company_name,financial_quarter,sector"
META = _COLUMNS + ",speaker_switches,unique_speakers,curator_id\n" + "".join(
    f"{call_id},60,24000,Co {call_id},1,Tech,1,1,1\n" for call_id in CALL_IDS
)
EVAL10 = _COLUMNS + ",utterances,unique_speakers,curator_id\n100,60,24000,Co 100,1,Tech,1,1,1\n"


def _fake_remote() -> dict[str, bytes]:
    files = {
        "earnings21-file-metadata.csv": META,
        "eval10-file-metadata.csv": EVAL10,
    }
    for call_id in CALL_IDS:
        files[f"output/google/{call_id}.nlp"] = _hyp_words("our cops were up")
        files[f"transcripts/nlp_references/{call_id}.nlp"] = _ref_words("our comps were up")
        files[f"transcripts/wer_tags/{call_id}.wer_tag.json"] = "{}"
    return {k: v.encode() for k, v in files.items()}


def _sources(cache_dir: Path, split: str) -> list[str]:
    return list(json.loads((cache_dir / split / "manifest.json").read_text())["sources"])


def test_build_writes_held_out_and_dev_corpora_from_the_cache(tmp_path: Path) -> None:
    remote = _fake_remote()
    fetched: list[str] = []

    def fetch(path: str) -> bytes:
        fetched.append(path)
        return remote[path]

    summary = build(tmp_path, fetch=fetch, dev_size=2, heldout2_size=2)

    dev = json.loads((tmp_path / "dev" / "manifest.json").read_text())
    assert _sources(tmp_path, "heldout") == ["100.srt"]
    assert _sources(tmp_path, "dev") == ["200.srt", "300.srt"]
    assert dev["sources"]["200.srt"]["priming_terms"] == ["Co 200"]
    assert (tmp_path / "dev" / "200.srt").read_text().startswith("1\n00:00:00,000 --> ")
    cases = json.loads((tmp_path / "dev" / "cases.json").read_text())
    assert [(c["source"], c["span"]) for c in cases] == [("200.srt", "cops"), ("300.srt", "cops")]
    assert summary["dev"]["cases"] == 2

    # Raw downloads are cached: a rebuild fetches nothing.
    fetched.clear()
    build(tmp_path, fetch=fetch, dev_size=2, heldout2_size=2)
    assert fetched == []


def test_heldout_2_is_a_seeded_draw_from_the_calls_in_no_other_split(
    tmp_path: Path,
) -> None:
    remote = _fake_remote()
    fetched: list[str] = []

    def fetch(path: str) -> bytes:
        fetched.append(path)
        return remote[path]

    summary = build(tmp_path, fetch=fetch, dev_size=2, heldout2_size=3)

    drawn = _sources(tmp_path, "heldout-2")
    assert drawn == ["400.srt", "500.srt", "800.srt"]  # pinned by the fixed seed
    assert not set(drawn) & set(_sources(tmp_path, "heldout") + _sources(tmp_path, "dev"))
    manifest = json.loads((tmp_path / "heldout-2" / "manifest.json").read_text())
    assert manifest["sources"]["800.srt"]["priming_terms"] == ["Co 800"]
    assert summary["heldout-2"]["calls"] == 3

    # The calls left over stay held back: never fetched. (Per-call files sit
    # in subdirectories; the metadata CSVs don't.)
    fetched_calls = {Path(p).name.split(".")[0] for p in fetched if "/" in p}
    assert fetched_calls == {"100", "200", "300", "400", "500", "800"}

    # The same draw on a rebuild.
    build(tmp_path, fetch=fetch, dev_size=2, heldout2_size=3)
    assert _sources(tmp_path, "heldout-2") == drawn


def test_build_refuses_a_heldout_2_larger_than_the_calls_left(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="heldout-2"):
        build(tmp_path, fetch=_fake_remote().__getitem__, dev_size=2, heldout2_size=7)


@pytest.mark.parametrize("split, call_id", [("dev", "300"), ("heldout-2", "800")])
def test_a_failed_build_leaves_the_previous_corpus_intact(
    tmp_path: Path, split: str, call_id: str
) -> None:
    remote = _fake_remote()
    build(tmp_path, fetch=remote.__getitem__, dev_size=2, heldout2_size=3)
    before = (tmp_path / split / "cases.json").read_text()
    srts = sorted(p.name for p in (tmp_path / split).glob("*.srt"))

    broken = dict(remote)
    broken[f"output/google/{call_id}.nlp"] = _hyp_words("").encode()  # nothing to align
    (tmp_path / "raw" / "output" / "google" / f"{call_id}.nlp").unlink()
    with pytest.raises(ValueError):
        build(tmp_path, fetch=broken.__getitem__, dev_size=2, heldout2_size=3)
    assert (tmp_path / split / "cases.json").read_text() == before
    assert sorted(p.name for p in (tmp_path / split).glob("*.srt")) == srts


def test_build_command_reports_each_split_with_the_caveat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = _fake_remote()
    real_build = earnings21.build
    monkeypatch.setattr(
        earnings21,
        "build",
        lambda: real_build(tmp_path, fetch=remote.__getitem__, dev_size=2, heldout2_size=3),
    )
    result = CliRunner().invoke(main, ["build-earnings21"])
    assert result.exit_code == 0, result.output
    assert "heldout: 1 calls, 1 cases" in result.output
    assert "dev: 2 calls, 2 cases" in result.output
    assert "heldout-2: 3 calls, 3 cases" in result.output
    assert "real-word 2" in result.output
    assert "Google's 2021 ASR" in result.output
