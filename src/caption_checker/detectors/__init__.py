"""Anomaly detectors. Each module exposes
``find(words, cues, vocab, config, **context) -> list[Flag]``.

``ALL_DETECTORS`` is ordered cheapest-first; ``context_embedding`` runs last and
receives the flags found so far as ``existing=``.
"""

from __future__ import annotations

from . import (
    context_embedding,
    oov,
    phonetic_internal,
    phonetic_vocab,
    split_word,
)

LEXICAL_DETECTORS = [oov, phonetic_vocab, phonetic_internal, split_word]
ALL_DETECTORS = [*LEXICAL_DETECTORS, context_embedding]

__all__ = ["ALL_DETECTORS", "LEXICAL_DETECTORS"]
