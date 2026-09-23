from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from caption_checker.cli import main
from caption_checker.detect import detect
from caption_checker.models import DetectConfig
from caption_checker.parser import parse, tokenize

DATA_DIR = Path(__file__).parent / "data"


def _flags(filename: str = "sample_lecture.srt", config: DetectConfig = DetectConfig()):
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


def test_doc_vocab_corrects_recurring_misspellings() -> None:
    """"Kalshi" recurs correctly 7x; "Kashi" and "Caushi" are each a
    consistent ASR mistake recurring 3x — enough to independently clear the
    recurrence threshold that promotes a term into doc_vocab. Without
    clustering near-duplicates together, each misspelling would get trusted
    as its own "correct" term instead of being corrected."""
    flags = _flags("doc_vocab_sample.srt")

    kashi_like = _covering(flags, "kashi", "caushi")
    assert kashi_like, "expected Kashi/Caushi misspellings to be flagged"
    assert all(f.candidates == ["Kalshi"] for f in kashi_like)

    spans = [f.span.lower() for f in flags]
    assert "kalshi" not in spans, "correctly-spelled Kalshi should never be flagged"


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
    flags = detect(cues, config=DetectConfig())
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


# --- CLI ---------------------------------------------------------------------


def test_cli_json_output_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["check", str(DATA_DIR / "sample_lecture.srt"),
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
        ["check", str(DATA_DIR / "sample_lecture.srt")],
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
    result = CliRunner().invoke(main, ["check", str(clean)])
    assert result.exit_code == 0
    assert "no likely caption errors" in result.output


def test_mid_sentence_name_not_corrected_to_common_word(tmp_path: Path) -> None:
    """"Steyer" shares a phonetic code with "story", but capitalized
    mid-sentence it reads as a name, not a mistranscribed common word. At a
    sentence start the capital says nothing, so the match still stands."""
    srt = tmp_path / "names.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:04,000\n"
        "Tell the story to Tom Steyer about it. Steyer knows.\n",
        encoding="utf-8",
    )
    flags = detect(parse(srt), config=DetectConfig())
    suggested = [f.global_indices[0] for f in flags if "story" in f.candidates]
    words = tokenize(parse(srt))
    assert [words[i].text for i in suggested] == ["Steyer"]
    assert words[suggested[0]].global_index == 8  # the sentence-initial one


def _detect_text(tmp_path: Path, text: str):
    srt = tmp_path / "t.srt"
    srt.write_text(f"1\n00:00:00,000 --> 00:00:04,000\n{text}\n", encoding="utf-8")
    return detect(parse(srt), config=DetectConfig())


def test_possessive_of_vocab_term_not_flagged(tmp_path: Path) -> None:
    assert not _detect_text(tmp_path, "We read OpenAI's report.")


def test_lost_apostrophe_on_vocab_term_suggests_possessive(tmp_path: Path) -> None:
    flags = _covering(_detect_text(tmp_path, "They doubted anthropics leadership."), "anthropics")
    assert flags and flags[0].candidates == ["Anthropic's"]


def test_miscased_brand_term_flagged(tmp_path: Path) -> None:
    flags = _detect_text(tmp_path, "She joined Deepseek and later DeepSeek again.")
    assert [f.span for f in flags] == ["Deepseek"]
    assert flags[0].candidates == ["DeepSeek"]


def test_split_word_ignores_filler_and_existing_terms(tmp_path: Path) -> None:
    """"uh when" joins to the sound of "Yann", and "why OpenAI" to "OpenAI"
    itself; neither is a split term."""
    flags = _detect_text(tmp_path, "So uh when did you see why OpenAI moved?")
    assert not [f for f in flags if f.detector.startswith("split_word")]
