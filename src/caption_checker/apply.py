"""Write accepted corrections into cue text by character-offset splice.

Per ADR-0001: replace exactly the substring the flag covers -- the first
Word's character offset within its cue, out to the end of the last Word, with
the same outer-punctuation trim ``span_text`` applied -- and leave every other
byte of the cue untouched. No re-tokenise-and-rejoin. A multi-token span
("con sensus") collapses to one word in a single operation.

A span that crosses a Cue boundary is written whole into the Cue it starts in,
from its first Word to the end of that Cue's text; its remaining Words are
cut from the following Cue(s), with the whitespace they leave at the start of
a Cue tidied away. A Cue left with no text is removed (the SRT serializer
renumbers). Timings never change.
"""

from __future__ import annotations

from dataclasses import replace

from caption_checker.models import Cue, Flag, Word
from caption_checker.normalize import _SPAN_EDGE
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
    cut: set[int] = set()  # Cues that lost leading Words to an earlier Cue
    for flag, replacement in accepted:
        for cue_index, start, end, text in _span_edits(
            flag, replacement, words_by_gi, cues_by_index
        ):
            edits_by_cue.setdefault(cue_index, []).append((start, end, text))
            if cue_index != flag.cue_index:
                cut.add(cue_index)

    out: list[Cue] = []
    for cue in cues:
        edits = edits_by_cue.get(cue.index)
        if not edits:
            out.append(cue)
            continue
        text = cue.text
        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        if cue.index in cut:
            text = text.lstrip()
            if not text:
                continue
        out.append(replace(cue, text=text))
    return out


def _span_edits(
    flag: Flag,
    replacement: str,
    words_by_gi: dict[int, Word],
    cues_by_index: dict[int, Cue],
) -> list[tuple[int, int, int, str]]:
    """``(cue index, start, end, text)`` splices that write one correction:
    the replacement into the first Cue, and a cut of the span's Words from
    each later Cue it reaches. A cut starts at 0 so any whitespace before
    the Words goes with them."""
    words = [words_by_gi[gi] for gi in flag.global_indices]
    first, last = words[0], words[-1]
    if first.cue_index == last.cue_index:
        start, end = _span_range(first, last, cues_by_index[first.cue_index].text)
        return [(first.cue_index, start, end, replacement)]

    by_cue: dict[int, list[Word]] = {}
    for w in words:
        by_cue.setdefault(w.cue_index, []).append(w)
    edits = []
    for cue_index, part in by_cue.items():
        text = cues_by_index[cue_index].text
        end = part[-1].char_offset + len(part[-1].text)
        if cue_index == first.cue_index:
            raw = text[first.char_offset : end]
            start = end - len(raw.lstrip(_SPAN_EDGE))
            edits.append((cue_index, start, end, replacement))
        else:
            if cue_index == last.cue_index:
                end = len(text[:end].rstrip(_SPAN_EDGE))
            edits.append((cue_index, 0, end, ""))
    return edits


def _span_range(first: Word, last: Word, cue_text: str) -> tuple[int, int]:
    """Character range within ``cue_text`` from ``first`` to ``last``,
    matching the outer-punctuation trim done when the span string was built."""
    raw_start = first.char_offset
    raw_end = last.char_offset + len(last.text)

    raw = cue_text[raw_start:raw_end]
    lead = len(raw) - len(raw.lstrip(_SPAN_EDGE))
    trail = len(raw) - len(raw.rstrip(_SPAN_EDGE))
    # Degenerate all-punctuation span: ``span_text`` keeps the original there,
    # so splice the whole raw range rather than an empty one.
    if lead + trail >= raw_end - raw_start:
        return raw_start, raw_end
    return raw_start + lead, raw_end - trail


def cues_spanned(flag: Flag, cues: list[Cue], words_by_gi: dict[int, Word]) -> list[Cue]:
    """Every Cue the flag's span touches, first to last -- one, unless the
    span crosses a Cue boundary."""
    order = [c.index for c in cues]
    last = words_by_gi[flag.global_indices[-1]].cue_index
    return cues[order.index(flag.cue_index) : order.index(last) + 1]
