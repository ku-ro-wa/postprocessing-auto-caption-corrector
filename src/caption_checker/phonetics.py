"""Phonetic encoding helpers, thin wrappers over ``metaphone`` / ``jellyfish``.

Double Metaphone is the primary code: it collapses ASR-style near-homophones
("cough ka" and "kafka" both encode to ``KFK``) far more reliably than plain
Metaphone or Soundex.
"""

from __future__ import annotations

import jellyfish
from metaphone import doublemetaphone

from caption_checker.normalize import clean


def code(word: str, algo: str = "metaphone") -> str:
    """Primary phonetic code for a word. ``algo`` selects the backend:
    ``"metaphone"`` (Double Metaphone, default) or ``"soundex"`` (coarser)."""
    cleaned = clean(word)
    if not cleaned:
        return ""
    if algo == "soundex":
        try:
            return jellyfish.soundex(cleaned)
        except (UnicodeEncodeError, ValueError):
            return ""
    primary, _secondary = doublemetaphone(cleaned)
    return primary


def codes(word: str, algo: str = "metaphone") -> set[str]:
    """All non-empty phonetic codes for a word. Double Metaphone can return a
    primary and a secondary encoding; both are worth indexing."""
    cleaned = clean(word)
    if not cleaned:
        return set()
    if algo == "soundex":
        one = code(cleaned, algo="soundex")
        return {one} if one else set()
    primary, secondary = doublemetaphone(cleaned)
    return {c for c in (primary, secondary) if c}


def similar(a: str, b: str) -> float:
    """0..1 similarity between two strings (Jaro-Winkler). Used to rank
    correction candidates that share a phonetic code."""
    if not a or not b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(a, b)
