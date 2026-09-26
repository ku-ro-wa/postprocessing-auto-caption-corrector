"""Shared token hygiene so every detector agrees on what a "word" is."""

from __future__ import annotations

import re

from caption_checker.models import Cue, Word

# Characters we strip from the edges of a token before any lookup. Keeps
# interior punctuation (e.g. "gRPC", "co-routine") intact.
_EDGE_PUNCT = "\"'`.,;:!?()[]{}<>«»•–—…“”‘’"
#: What a multi-Word span sheds at its edges: punctuation, and the spaces
#: left between it and the Words when a whole edge Word is punctuation.
_SPAN_EDGE = _EDGE_PUNCT + " \t\n"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DIGIT_RE = re.compile(r"\d")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def trim_edges(token: str) -> str:
    """Strip leading/trailing punctuation, preserving case and interior."""
    return token.strip(_EDGE_PUNCT)


def clean(token: str) -> str:
    """Lowercase and strip edge punctuation. Used for every dictionary /
    phonetic lookup; the original surface form stays on ``Word.text``."""
    return token.strip(_EDGE_PUNCT).casefold()


def is_wordlike(token: str) -> bool:
    """True when a token is worth running detectors against — rejects numbers,
    timestamps, single characters and short all-caps acronyms (including their
    plural, "LLMs" alongside "LLM" — wordfreq frequently has no data for the
    plural form of a recent acronym even when it trusts the singular)."""
    stripped = token.strip(_EDGE_PUNCT)
    if len(stripped) < 2:
        return False
    if _DIGIT_RE.search(stripped):
        return False
    if not _LETTER_RE.search(stripped):
        return False
    if stripped.isupper() and len(stripped) <= 4:
        return False
    # A plural of a short acronym ("LLMs") loses the all-upper check above to
    # its lowercase "s". Require 2+ letters in the core so an ordinary,
    # sentence-capitalized word ending in "s" ("As", "Its") can't match --
    # those have a lowercase letter before the "s" and so aren't all-upper.
    core = stripped[:-1] if stripped.endswith("s") else stripped
    if 2 <= len(core) <= 4 and core.isupper():
        return False
    return True


def sentences(cues: list[Cue], words: list[Word]) -> list[tuple[str, list[int]]]:
    """Join cue text across cue boundaries, split into sentences, and pair each
    sentence with the ``global_index`` values of the words it contains.

    Word association is positional: words keep their transcript order, so we
    walk them in lockstep with the flattened sentence stream.
    """
    joined = " ".join(cue.text.replace("\n", " ") for cue in cues)
    raw = _SENTENCE_SPLIT_RE.split(joined)

    result: list[tuple[str, list[int]]] = []
    cursor = 0
    for sentence in raw:
        sentence = sentence.strip()
        if not sentence:
            continue
        count = len(sentence.split())
        indices = [w.global_index for w in words[cursor : cursor + count]]
        cursor += count
        result.append((sentence, indices))

    # Any trailing words (no sentence-final punctuation) join the last sentence.
    if cursor < len(words) and result:
        last_sentence, last_indices = result[-1]
        last_indices.extend(w.global_index for w in words[cursor:])
        result[-1] = (last_sentence, last_indices)

    return result
