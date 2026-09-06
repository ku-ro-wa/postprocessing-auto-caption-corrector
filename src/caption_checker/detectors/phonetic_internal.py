"""Phonetic-vs-transcript detector: when the same phonetic code appears both as
a known-good word and as an unknown one elsewhere in the same transcript, the
unknown spelling is probably a mistranscription of the good one.

This is the plan's "term used correctly elsewhere in the document" rule. It does
not fire on the Session 1 sample (none of the targets appear correctly), but it
is the basis for the future skip-the-LLM shortcut.
"""

from __future__ import annotations

from collections import defaultdict

from wordfreq import zipf_frequency

from caption_checker.models import (
    DETECTOR_PHONETIC_INTERNAL,
    Cue,
    DetectConfig,
    Flag,
    Word,
)
from caption_checker.normalize import clean, is_wordlike
from caption_checker.phonetics import codes, similar
from caption_checker.vocab import Vocab

from .base import index_cues, make_flag


def _is_known_good(cleaned: str, vocab: Vocab, config: DetectConfig) -> bool:
    return (
        cleaned in vocab.terms
        or zipf_frequency(cleaned, "en") >= config.known_good_zipf_min
    )


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    **_context: object,
) -> list[Flag]:
    cues_by_index = index_cues(cues)

    # phonetic code -> {cleaned surface form -> one representative Word}
    by_code: dict[str, dict[str, Word]] = defaultdict(dict)
    for word in words:
        if not is_wordlike(word.text):
            continue
        cleaned = clean(word.text)
        if len(cleaned) < config.min_token_len or cleaned in config.stopwords:
            continue
        for code in codes(word.text, algo=config.phonetic_algo):
            by_code[code].setdefault(cleaned, word)

    flags: list[Flag] = []
    for surfaces in by_code.values():
        if len(surfaces) < 2:
            continue
        good = [s for s in surfaces if _is_known_good(s, vocab, config)]
        bad = [s for s in surfaces if not _is_known_good(s, vocab, config)]
        if not good or not bad:
            continue
        for bad_surface in bad:
            word = surfaces[bad_surface]
            ranked = sorted(
                good, key=lambda s: similar(bad_surface, s), reverse=True
            )
            flags.append(
                make_flag(
                    [word],
                    cues_by_index,
                    detector=DETECTOR_PHONETIC_INTERNAL,
                    reason=f'"{word.text}" sounds like "{ranked[0]}" used '
                    "elsewhere in this transcript",
                    candidates=ranked,
                    confidence=0.7,
                )
            )
    return flags
