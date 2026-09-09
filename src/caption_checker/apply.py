"""Write accepted corrections into cue text by character-offset splice.

Per ADR-0001: replace exactly the substring the flag covers -- the first
Word's character offset within its cue, out to the end of the last Word, with
the same outer-punctuation trim ``span_text`` applied -- and leave every other
byte of the cue untouched. No re-tokenise-and-rejoin. A multi-token span
("con sensus") collapses to one word in a single operation.
"""

from __future__ import annotations

from dataclasses import replace

from caption_checker.models import Cue, Flag, Word
from caption_checker.normalize import _EDGE_PUNCT
from caption_checker.parser import tokenize


def apply_corrections(
    cues: list[Cue], accepted: list[tuple[Flag, str]]
) -> list[Cue]:
    """Return a new cue list with every ``(flag, replacement)`` spliced in.

    Corrections in the same cue are applied right-to-left so earlier character
    offsets stay valid as the text shrinks or grows.
    """
    words_by_gi = {w.global_index: w for w in tokenize(cues)}
    cues_by_index = {c.index: c for c in cues}

    edits_by_cue: dict[int, list[tuple[int, int, str]]] = {}
    for flag, replacement in accepted:
        cue_text = cues_by_index[flag.cue_index].text
        start, end = _span_range(flag, words_by_gi, cue_text)
        edits_by_cue.setdefault(flag.cue_index, []).append(
            (start, end, replacement)
        )

    out: list[Cue] = []
    for cue in cues:
        edits = edits_by_cue.get(cue.index)
        if not edits:
            out.append(cue)
            continue
        text = cue.text
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        out.append(replace(cue, text=text))
    return out


def _span_range(
    flag: Flag, words_by_gi: dict[int, Word], cue_text: str
) -> tuple[int, int]:
    """Character range within ``cue_text`` the flag's span occupies, matching
    the outer-punctuation trim done when the span string was built."""
    first = words_by_gi[flag.global_indices[0]]
    last = words_by_gi[flag.global_indices[-1]]
    raw_start = first.char_offset
    raw_end = last.char_offset + len(last.text)

    raw = cue_text[raw_start:raw_end]
    lead = len(raw) - len(raw.lstrip(_EDGE_PUNCT))
    trail = len(raw) - len(raw.rstrip(_EDGE_PUNCT))
    # Degenerate all-punctuation span: ``span_text`` keeps the original there,
    # so splice the whole raw range rather than an empty one.
    if lead + trail >= raw_end - raw_start:
        return raw_start, raw_end
    return raw_start + lead, raw_end - trail
