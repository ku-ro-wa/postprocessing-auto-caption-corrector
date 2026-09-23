"""Anomaly detectors. Each module exposes
``find(words, cues, vocab, config, **context) -> list[Flag]``.

``ALL_DETECTORS`` is ordered cheapest-first.
"""

from __future__ import annotations

from . import (
    oov,
    phonetic_internal,
    phonetic_vocab,
    split_word,
)

ALL_DETECTORS = [oov, phonetic_vocab, phonetic_internal, split_word]

__all__ = ["ALL_DETECTORS"]
