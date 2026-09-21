from __future__ import annotations

from caption_checker.normalize import is_wordlike


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
