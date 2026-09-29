"""Write accepted corrections into cue text by character-offset splice --
for Export (``apply_corrections``), and for a view of the Cues as Export
would write them, each Flag's span marked (``splice``).

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

from dataclasses import dataclass, field, replace
from typing import Sequence

from caption_checker.models import Cue, Flag, Word
from caption_checker.normalize import _SPAN_EDGE
from caption_checker.parser import tokenize


def apply_corrections(
    cues: list[Cue], accepted: list[tuple[Flag, str]]
) -> list[Cue]:
    """Return a new cue list with every ``(flag, replacement)`` spliced in."""
    return [
        replace(s.cue, text=s.text)
        for s in splice(cues, accepted)
        if s.merged_into is None
    ]


# One splice in a Cue: start, end, the text written there (``None`` for a
# marked span, left as it is) and the Flag it belongs to.
_Edit = tuple[int, int, "str | None", Flag]


@dataclass
class SplicedCue:
    """One Cue after :func:`splice`: its text in ``pieces``, each piece tagged
    with the Flag whose span it is (``None`` between spans). ``merged_into``
    is the index of the Cue an accepted cross-Cue span was written into when
    that left this Cue with no text -- a Cue Export removes."""

    cue: Cue
    pieces: list[tuple[str, Flag | None]]
    merged_into: int | None = None
    #: Where each piece's text sits in ``cue.text`` (parallel to ``pieces``):
    #: the range a span replaced, or the range an untouched piece was.
    origins: list[tuple[int, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(text for text, _ in self.pieces)


def splice(
    cues: list[Cue],
    accepted: list[tuple[Flag, str]],
    marked: Sequence[Flag] = (),
) -> list[SplicedCue]:
    """Every Cue, in order, with each ``(flag, replacement)`` spliced in and
    each ``marked`` Flag's span left as it is -- both tagged in the pieces.

    An accepted edit that overlaps one earlier in its Cue is skipped rather
    than splicing over text it already replaced; a mark that overlaps any
    edit, or an earlier mark, is skipped too.
    """
    words_by_gi = {w.global_index: w for w in tokenize(cues)}
    cues_by_index = {c.index: c for c in cues}

    accepted_by_cue: dict[int, list[_Edit]] = {}
    marked_by_cue: dict[int, list[_Edit]] = {}
    cut_into: dict[int, int] = {}  # Cue that lost leading Words -> the Cue they went to
    for flag, replacement in accepted:
        for cue_index, start, end, text in _span_edits(
            flag, replacement, words_by_gi, cues_by_index
        ):
            accepted_by_cue.setdefault(cue_index, []).append((start, end, text, flag))
            if cue_index != flag.cue_index:
                cut_into[cue_index] = flag.cue_index
    for flag in marked:
        for cue_index, start, end, _ in _span_edits(flag, "", words_by_gi, cues_by_index):
            marked_by_cue.setdefault(cue_index, []).append((start, end, None, flag))

    out = []
    for cue in cues:
        # Accepted edits are chosen first, so a mark never changes what's written.
        edits = _non_overlapping(accepted_by_cue.get(cue.index, []), [])
        edits += _non_overlapping(marked_by_cue.get(cue.index, []), edits)
        pieces: list[tuple[str, Flag | None]] = []
        origins: list[tuple[int, int]] = []
        at = 0
        for start, end, written, flag in sorted(edits, key=lambda e: e[0]):
            kept = cue.text[start:end]
            if written is None:  # marked: any whitespace a cut takes stays outside
                start += len(kept) - len(kept.lstrip())
                kept = kept.lstrip()
            pieces.append((cue.text[at:start], None))
            origins.append((at, start))
            pieces.append((kept if written is None else written, flag))
            origins.append((start, end))
            at = end
        pieces.append((cue.text[at:], None))
        origins.append((at, len(cue.text)))
        kept_pieces = [(p, o) for p, o in zip(pieces, origins) if p[0]]
        pieces = [p for p, _ in kept_pieces]
        origins = [o for _, o in kept_pieces]

        merged_into = None
        if cue.index in cut_into:
            pieces, origins = _lstrip(pieces, origins)
            if not pieces:
                merged_into = cut_into[cue.index]
        out.append(SplicedCue(cue, pieces, merged_into, origins))
    return out


def _non_overlapping(edits: list[_Edit], taken: list[_Edit]) -> list[_Edit]:
    """``edits`` in order of start, less any that overlap one in ``taken`` or
    one kept before it."""
    kept: list[_Edit] = []
    for edit in sorted(edits, key=lambda e: e[0]):
        if all(edit[1] <= t[0] or edit[0] >= t[1] for t in [*taken, *kept]):
            kept.append(edit)
    return kept


def _lstrip(
    pieces: list[tuple[str, Flag | None]], origins: list[tuple[int, int]]
) -> tuple[list[tuple[str, Flag | None]], list[tuple[int, int]]]:
    while pieces and not pieces[0][0].strip():
        pieces, origins = pieces[1:], origins[1:]
    if pieces:
        text, flag = pieces[0]
        stripped = text.lstrip()
        first = origins[0]
        if flag is None:  # an untouched piece keeps its offsets in step
            first = (first[0] + len(text) - len(stripped), first[1])
        pieces = [(stripped, flag), *pieces[1:]]
        origins = [first, *origins[1:]]
    return pieces, origins


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
