from __future__ import annotations

from datetime import timedelta

from caption_checker.models import Cue
from caption_checker.normalize import (
    CONTEXT_WINDOW_WORDS,
    MAX_CONTEXT_WORDS,
    is_wordlike,
    span_contexts,
)
from caption_checker.parser import tokenize


def _cues(*texts: str) -> list[Cue]:
    return [
        Cue(index=i, start=timedelta(seconds=i), end=timedelta(seconds=i + 1), text=t)
        for i, t in enumerate(texts)
    ]


def test_short_acronym_and_its_plural_are_not_wordlike() -> None:
    for token in ("LLM", "LLMs", "GPU", "GPUs", "CEO", "CEOs"):
        assert not is_wordlike(token), f"{token!r} should be treated as a trusted acronym"


def test_ordinary_capitalized_words_ending_in_s_stay_wordlike() -> None:
    """A sentence-initial "As" or "Its" must not be mistaken for a plural
    acronym just because stripping the trailing "s" leaves a single
    uppercase letter -- only a genuine multi-letter all-caps core counts."""
    for token in ("As", "Its", "Is", "Us", "Ads", "Was"):
        assert is_wordlike(token), f"{token!r} should not be treated as an acronym"


def test_long_all_caps_run_is_still_wordlike() -> None:
    # Five or more letters is outside the "short acronym" exemption whether
    # or not it's plural -- an unusually long all-caps run is worth a look.
    assert is_wordlike("KUBERNETES")
    assert is_wordlike("KUBERNETESs")


def test_span_context_is_its_sentence_when_short() -> None:
    cues = _cues("The cat sat. A dog", "barked loudly. Then quiet.")
    words = tokenize(cues)
    context = span_contexts(cues, words)
    dog = next(w.global_index for w in words if w.text == "dog")
    assert context([dog]) == "A dog barked loudly."


def test_span_context_across_a_sentence_end_joins_both_sentences() -> None:
    cues = _cues("The cat sat. A dog barked.")
    words = tokenize(cues)
    context = span_contexts(cues, words)
    assert context([2, 3]) == "The cat sat. A dog barked."


def test_unpunctuated_transcript_gives_a_window_not_the_whole_text() -> None:
    """An auto-caption with no sentence punctuation is one "sentence"; a
    Flag's context must still be a window around its span (the review
    page showed the entire Transcript on every card)."""
    texts = [f"w{n}" for n in range(MAX_CONTEXT_WORDS * 5)]
    cues = _cues(*(" ".join(texts[i : i + 8]) for i in range(0, len(texts), 8)))
    words = tokenize(cues)
    context = span_contexts(cues, words)
    mid = len(words) // 2
    expected = texts[mid - CONTEXT_WINDOW_WORDS : mid + 1 + CONTEXT_WINDOW_WORDS]
    assert context([mid]) == " ".join(expected)
    # Clipped at the transcript's edges rather than wrapping or padding.
    assert context([0]) == " ".join(texts[: CONTEXT_WINDOW_WORDS + 1])
