"""Phonetic-vs-vocabulary detector: a token that is not itself a known domain
term but sounds exactly like one ("cubernetes" -> "Kubernetes")."""

from __future__ import annotations

from wordfreq import zipf_frequency

from caption_checker.models import (
    DETECTOR_PHONETIC_VOCAB,
    Cue,
    DetectConfig,
    Flag,
    Word,
)
from caption_checker.normalize import clean, is_wordlike
from caption_checker.phonetics import codes, similar
from caption_checker.vocab import Vocab

from .base import index_cues, make_flag


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    **_context: object,
) -> list[Flag]:
    cues_by_index = index_cues(cues)
    flags: list[Flag] = []
    for word in words:
        if not is_wordlike(word.text):
            continue
        cleaned = clean(word.text)
        if cleaned in vocab.terms or cleaned in config.stopwords:
            continue
        # A term that is already a perfectly ordinary English word is unlikely
        # to be a mistranscription of a domain term.
        if zipf_frequency(cleaned, "en") >= config.known_good_zipf_min:
            continue

        matches: set[str] = set()
        for code in codes(word.text, algo=config.phonetic_algo):
            matches |= vocab.by_phonetic.get(code, set())
        if not matches:
            continue

        ranked = sorted(
            matches, key=lambda term: similar(cleaned, clean(term)), reverse=True
        )
        top = similar(cleaned, clean(ranked[0]))
        flags.append(
            make_flag(
                [word],
                cues_by_index,
                detector=DETECTOR_PHONETIC_VOCAB,
                reason=f'"{word.text}" sounds like domain term '
                f'"{ranked[0]}"',
                candidates=ranked,
                confidence=min(0.95, 0.6 + 0.35 * top),
            )
        )
    return flags
