"""Turn a reviewer's edit of one Cue's text into the splices that record it
(ADR 0009). Pure text work: no storage, and no Flag records are built here.

The edit is diffed at Word level against the Cue as it would export, and each
changed stretch becomes a *span*: a range of the exported text and what it
becomes. A span covers at least one Word, and its outer punctuation is the
same before and after, because Export trims a span's outer punctuation
(ADR 0001). An insertion, a deletion or a punctuation-only change is widened
to a neighbouring Word until both hold.

Spans are then grouped into *regions* against the Flags already in the Cue: a
region sitting exactly inside one Flag's span updates that Flag's text, and
any other region is a new Flag that supersedes the Flags it overlaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from caption_checker.apply import SplicedCue
from caption_checker.models import Flag
from caption_checker.normalize import _SPAN_EDGE

_WORD_RE = re.compile(r"\S+")


@dataclass
class Region:
    """A stretch of the exported Cue text that the edit rewrites.

    ``start``/``end`` are offsets in the exported text and ``replacement`` is
    what that stretch becomes. ``inside`` is the Flag whose span the region
    fills exactly: its Review Decision text becomes ``replacement``.
    Otherwise the region is a new Flag: ``flags`` are the Flags it overlaps
    and ``origin`` is the range of the Cue's own text it covers.
    """

    start: int
    end: int
    replacement: str
    flags: list[Flag] = field(default_factory=list)
    origin: tuple[int, int] = (0, 0)
    inside: Flag | None = None


@dataclass
class EditPlan:
    regions: list[Region]
    #: What couldn't be saved, one sentence each.
    unsaved: list[str]


def plan_edit(
    spliced: SplicedCue,
    new_text: str,
    *,
    live: Callable[[Flag], bool],
    crosses_cues: Callable[[Flag], bool],
) -> EditPlan:
    """Regions that turn ``spliced.text`` into ``new_text``. ``live`` says
    whether a Flag in the Cue takes part in the overlap rules, and
    ``crosses_cues`` whether its span runs into another Cue, which a
    reviewer's edit can't merge with."""
    old_text = spliced.text
    spans, unsaved = _spans(old_text, new_text)
    regions = _regions(spliced, spans, live)

    saved: list[Region] = []
    for region in regions:
        if region.inside is None and any(crosses_cues(f) for f in region.flags):
            unsaved.append(
                f"Not saved: {_quote(old_text[region.start : region.end])} overlaps "
                "a fix that runs across captions."
            )
        else:
            saved.append(region)
    return EditPlan(saved, unsaved)


def apply_regions(text: str, regions: list[Region]) -> str:
    """``text`` with each region's range replaced."""
    for r in sorted(regions, key=lambda r: r.start, reverse=True):
        text = text[: r.start] + r.replacement + text[r.end :]
    return text


def _quote(text: str) -> str:
    return "“" + " ".join(text.split()) + "”"


# --- Word-level diff ---------------------------------------------------------------

_Span = tuple[int, int, str]  # start and end in the exported text, replacement


def _lead(text: str) -> str:
    return text[: len(text) - len(text.lstrip(_SPAN_EDGE))]


def _trail(text: str) -> str:
    return text[len(text.rstrip(_SPAN_EDGE)) :]


def _spans(old: str, new: str) -> tuple[list[_Span], list[str]]:
    old_toks = [m.span() for m in _WORD_RE.finditer(old)]
    new_toks = [m.span() for m in _WORD_RE.finditer(new)]
    opcodes = SequenceMatcher(
        None,
        [old[s:e] for s, e in old_toks],
        [new[s:e] for s, e in new_toks],
        autojunk=False,
    ).get_opcodes()

    pair: dict[int, int] = {}  # unchanged old Word -> the same Word in the new text
    changed: dict[int, tuple[int, int, int, int]] = {}  # changed old Word -> its op
    for tag, i1, i2, j1, j2 in opcodes:
        for i in range(i1, i2):
            if tag == "equal":
                pair[i] = j1 + (i - i1)
            else:
                changed[i] = (i1, i2, j1, j2)

    def raw(text: str, toks: list[tuple[int, int]], i: int, j: int) -> str:
        return text[toks[i][0] : toks[j - 1][1]] if j > i else ""

    def grow_left(a: int, c: int) -> tuple[int, int] | None:
        if a == 0:
            return None
        if a - 1 in pair:
            return a - 1, pair[a - 1]
        return changed[a - 1][0], changed[a - 1][2]

    def grow_right(b: int, d: int) -> tuple[int, int] | None:
        if b == len(old_toks):
            return None
        if b in pair:
            return b + 1, pair[b] + 1
        return changed[b][1], changed[b][3]

    def widen(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int] | None:
        """The op's Word ranges (old ``a:b``, new ``c:d``), grown until they
        make a span, or ``None`` when a Cue's edge stops that."""
        while True:
            o, n = raw(old, old_toks, a, b), raw(new, new_toks, c, d)
            if not o.strip(_SPAN_EDGE) or not n.strip(_SPAN_EDGE):
                # No Word of its own on one side: a neighbour either side does.
                grown = grow_left(a, c)
                if grown is not None:
                    a, c = grown
                    continue
                grown = grow_right(b, d)
                if grown is None:
                    return None
                b, d = grown
                continue
            if _lead(o) != _lead(n):
                grown = grow_left(a, c)
                if grown is None:
                    return None
                a, c = grown
                continue
            if _trail(o) != _trail(n):
                grown = grow_right(b, d)
                if grown is None:
                    return None
                b, d = grown
                continue
            return a, b, c, d

    ranges: list[list[int]] = []
    unsaved: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        grown = widen(i1, i2, j1, j2)
        if grown is None:
            unsaved.append(
                f"Not saved: {_quote(raw(old, old_toks, i1, i2))} → "
                f"{_quote(raw(new, new_toks, j1, j2))}. A change that only touches "
                "punctuation at the very start or end of a caption can't be saved, "
                "and a caption can't be left without a word."
            )
        else:
            ranges.append(list(grown))

    # Widening can make neighbouring spans meet: they become one.
    merged: list[list[int]] = []
    for a, b, c, d in sorted(ranges):
        if merged and (a < merged[-1][1] or c < merged[-1][3]):
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][3] = max(merged[-1][3], d)
        else:
            merged.append([a, b, c, d])

    spans: list[_Span] = []
    for a, b, c, d in merged:
        o, n = raw(old, old_toks, a, b), raw(new, new_toks, c, d)
        lead, trail = _lead(o), _trail(o)
        spans.append(
            (
                old_toks[a][0] + len(lead),
                old_toks[b - 1][1] - len(trail),
                n[len(lead) : len(n) - len(trail)],
            )
        )
    return spans, unsaved


# --- Regions against the Cue's Flags -----------------------------------------------


@dataclass
class _Piece:
    start: int  # in the exported text
    end: int
    flag: Flag | None
    origin: tuple[int, int]  # in the Cue's own text


def _regions(
    spliced: SplicedCue, spans: list[_Span], live: Callable[[Flag], bool]
) -> list[Region]:
    pieces: list[_Piece] = []
    at = 0
    for (text, flag), origin in zip(spliced.pieces, spliced.origins):
        # A Flag out of the overlap rules still keeps its piece when its text
        # is not the Cue's own, so exported offsets stay mapped to the Cue's.
        kept = flag if flag and (live(flag) or len(text) != origin[1] - origin[0]) else None
        pieces.append(_Piece(at, at + len(text), kept, origin))
        at += len(text)

    # Each span grows to cover the Flag pieces it touches; regions that then
    # overlap become one.
    grown: list[tuple[int, int, list[_Span], list[_Piece]]] = []
    for s, e, replacement in spans:
        touched = [p for p in pieces if p.flag and p.start < e and s < p.end]
        grown.append(
            (
                min([s, *(p.start for p in touched)]),
                max([e, *(p.end for p in touched)]),
                [(s, e, replacement)],
                touched,
            )
        )
    grown.sort(key=lambda g: g[0])
    merged: list[tuple[int, int, list[_Span], list[_Piece]]] = []
    for start, end, edits, touched in grown:
        if merged and start < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (
                prev[0],
                max(prev[1], end),
                prev[2] + edits,
                prev[3] + [p for p in touched if p not in prev[3]],
            )
        else:
            merged.append((start, end, edits, touched))

    text = spliced.text
    regions = []
    for start, end, edits, touched in merged:
        replacement = text[start:end]
        for s, e, r in sorted(edits, reverse=True):
            replacement = replacement[: s - start] + r + replacement[e - start :]
        inside = (
            touched[0].flag
            if len(touched) == 1 and (touched[0].start, touched[0].end) == (start, end)
            else None
        )
        regions.append(
            Region(
                start,
                end,
                replacement,
                flags=[p.flag for p in touched if p.flag],
                origin=(_origin(pieces, start, left=True), _origin(pieces, end, left=False)),
                inside=inside,
            )
        )
    return regions


def _origin(pieces: list[_Piece], at: int, *, left: bool) -> int:
    """The offset in the Cue's own text of an exported-text offset that sits
    in untouched text or on the edge of a Flag's span."""
    for p in pieces:
        if (p.start <= at < p.end) if left else (p.start < at <= p.end):
            if p.flag is None:
                return p.origin[0] + (at - p.start)
            return p.origin[0] if left else p.origin[1]
    return pieces[-1].origin[1] if pieces else 0
