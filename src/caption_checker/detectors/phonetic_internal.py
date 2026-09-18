"""Phonetic-vs-transcript detector: when the same phonetic code appears both as
a known-good word and as an unknown one elsewhere in the same transcript, the
unknown spelling is probably a mistranscription of the good one.

This is the plan's "term used correctly elsewhere in the document" rule. A
second pass below extends it to terms that are unknown outside this transcript
too (a recurring brand/guest name, say): ``doc_vocab`` supplies those as
"good" even without a matching phonetic code, since Double Metaphone can
diverge on near-misses that still read as the same botched word ("Caushi" vs
"Kalshi" -> KX vs KLX).
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
from caption_checker.vocab import DocVocab, Vocab

from .base import index_cues, make_flag


def _is_known_good(
    cleaned: str, vocab: Vocab, doc_vocab: DocVocab, config: DetectConfig
) -> bool:
    return (
        cleaned in vocab.terms
        or cleaned in doc_vocab
        or zipf_frequency(cleaned, "en") >= config.known_good_zipf_min
    )


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    *,
    doc_vocab: DocVocab | None = None,
    **_context: object,
) -> list[Flag]:
    cues_by_index = index_cues(cues)
    doc_vocab = doc_vocab or DocVocab()

    # phonetic code -> {cleaned surface form -> every occurrence}
    by_code: dict[str, dict[str, list[Word]]] = defaultdict(lambda: defaultdict(list))
    by_surface: dict[str, list[Word]] = defaultdict(list)
    for word in words:
        if not is_wordlike(word.text):
            continue
        cleaned = clean(word.text)
        if len(cleaned) < config.min_token_len or cleaned in config.stopwords:
            continue
        by_surface[cleaned].append(word)
        for code in codes(word.text, algo=config.phonetic_algo):
            by_code[code][cleaned].append(word)

    flags: list[Flag] = []
    flagged: set[int] = set()  # id(Word) already covered, so passes don't overlap

    def emit(word: Word, candidates: list[str], reason: str) -> None:
        if id(word) in flagged:
            return
        flagged.add(id(word))
        flags.append(
            make_flag(
                [word],
                cues_by_index,
                detector=DETECTOR_PHONETIC_INTERNAL,
                reason=reason,
                candidates=candidates,
                confidence=0.7,
            )
        )

    # Pass 1: fuzzy match against doc_vocab, checked first because it's
    # grounded in evidence specific to this transcript rather than a
    # possibly-coincidental dictionary phonetic-code collision (pass 2 would
    # otherwise grab "Kashi" via "cash"/"couch" before this pass gets a shot
    # at the far more likely "Kalshi"). Matches through any clustered
    # variant, not just the canonical spelling, so a chain like
    # "Caushi" -> "Kashi" -> "Kalshi" still resolves even though the two
    # ends aren't similar enough to match directly.
    if doc_vocab.variants:
        for cleaned, occurrences in by_surface.items():
            if _is_known_good(cleaned, vocab, doc_vocab, config):
                continue
            best_variant = max(
                doc_vocab.variants, key=lambda v: similar(cleaned, v)
            )
            if similar(cleaned, best_variant) < config.doc_vocab_fuzzy_min:
                continue
            canonical = doc_vocab.variants[best_variant]
            display = doc_vocab.display[canonical]
            for word in occurrences:
                emit(
                    word,
                    [display],
                    f'"{word.text}" is likely a mistranscription of '
                    f'"{display}", which recurs {doc_vocab.counts[canonical]} '
                    "times elsewhere in this transcript",
                )

    # Pass 2: exact phonetic-code match against a known-good sibling.
    for surfaces in by_code.values():
        if len(surfaces) < 2:
            continue
        good = [s for s in surfaces if _is_known_good(s, vocab, doc_vocab, config)]
        bad = [s for s in surfaces if not _is_known_good(s, vocab, doc_vocab, config)]
        if not good or not bad:
            continue
        for bad_surface in bad:
            ranked = sorted(
                good, key=lambda s: similar(bad_surface, s), reverse=True
            )
            for word in surfaces[bad_surface]:
                emit(
                    word,
                    ranked,
                    f'"{word.text}" sounds like "{ranked[0]}" used '
                    "elsewhere in this transcript",
                )

    return flags
