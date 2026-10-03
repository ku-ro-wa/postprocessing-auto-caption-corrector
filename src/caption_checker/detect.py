"""Detection orchestrator: run every detector, merge overlapping flags, attach
sentence context."""

from __future__ import annotations

from caption_checker.detectors import ALL_DETECTORS
from caption_checker.detectors.base import span_text
from caption_checker.models import Cue, DetectConfig, Flag, Word
from caption_checker.normalize import _SPAN_EDGE, sentences
from caption_checker.parser import tokenize
from caption_checker.vocab import Vocab, build_doc_vocab, load_vocab


def detect(
    cues: list[Cue],
    *,
    vocab: Vocab | None = None,
    config: DetectConfig | None = None,
) -> list[Flag]:
    config = config or DetectConfig()
    vocab = vocab or load_vocab(algo=config.phonetic_algo)
    words = tokenize(cues)
    doc_vocab = build_doc_vocab(
        words, vocab, config, min_count=config.doc_vocab_min_count
    )

    raw: list[Flag] = []
    for detector in ALL_DETECTORS:
        raw.extend(detector.find(words, cues, vocab, config, doc_vocab=doc_vocab))

    merged = _merge(raw, words)
    _attach_context(merged, cues, words)
    return merged


def _merge(flags: list[Flag], words: list[Word]) -> list[Flag]:
    """Collapse flags whose word spans overlap into one combined flag. Each
    flag covers a contiguous range of ``global_index`` values, so this is an
    interval merge over ``(min, max)`` spans."""
    if not flags:
        return []

    order = sorted(flags, key=lambda f: (min(f.global_indices), max(f.global_indices)))
    groups: list[list[Flag]] = [[order[0]]]
    reach = max(order[0].global_indices)
    for flag in order[1:]:
        if min(flag.global_indices) <= reach:
            groups[-1].append(flag)
            reach = max(reach, max(flag.global_indices))
        else:
            groups.append([flag])
            reach = max(flag.global_indices)

    return sorted(
        (_combine(group, words) for group in groups),
        key=lambda f: (f.cue_index, min(f.global_indices)),
    )


def _combine(group: list[Flag], words: list[Word]) -> Flag:
    """One Flag over the whole extent of ``group``. Every Candidate replaces
    that whole span, as an accepted Review Decision's text does on Export: one
    from a narrower Flag is rewritten with the merged span's other Words
    around it ("Nome" -> "Noam" inside "Nome Brown" gives "Noam Brown")."""
    if len(group) == 1:
        return group[0]

    covered = [i for f in group for i in f.global_indices]
    indices = list(range(min(covered), max(covered) + 1))
    detectors = sorted({d for f in group for d in f.detector.split("+")})

    candidates: list[str] = []
    for flag in sorted(group, key=lambda f: f.confidence, reverse=True):
        for cand in flag.candidates:
            cand = _widen(cand, flag, indices, words)
            if cand not in candidates:
                candidates.append(cand)

    reasons = "; ".join(
        dict.fromkeys(f.reason for f in group)  # dedup, keep order
    )
    agree_bonus = 0.05 * (len(detectors) - 1)
    confidence = min(0.98, max(f.confidence for f in group) + agree_bonus)

    return Flag(
        span=span_text([words[i] for i in indices]),
        global_indices=indices,
        cue_index=min(f.cue_index for f in group),
        start=min(f.start for f in group),
        end=max(f.end for f in group),
        detector="+".join(detectors),
        reason=reasons,
        candidates=candidates,
        confidence=round(confidence, 3),
    )


def _widen(candidate: str, flag: Flag, indices: list[int], words: list[Word]) -> str:
    """``candidate``, a replacement for ``flag``'s Words, as a replacement for
    all of ``indices``: the Words either side kept, along with any punctuation
    ``flag``'s own span trimmed that falls inside the wider one ("Treyus,"
    -> "trace," before "why"). Only the wider span's outer edges are trimmed,
    as ``span_text`` does."""
    first, last = min(flag.global_indices), max(flag.global_indices)
    own = " ".join(words[i].text for i in range(first, last + 1))
    core = own.strip(_SPAN_EDGE)
    lead, trail = "", ""
    if core:
        lead = own[: len(own) - len(own.lstrip(_SPAN_EDGE))]
        trail = own[len(own.rstrip(_SPAN_EDGE)) :]
    before = " ".join(words[i].text for i in indices if i < first)
    after = " ".join(words[i].text for i in indices if i > last)
    if before:
        candidate = f"{before} {lead}".lstrip(_SPAN_EDGE) + candidate
    if after:
        candidate += f"{trail} {after}".rstrip(_SPAN_EDGE)
    return candidate


def _attach_context(flags: list[Flag], cues: list[Cue], words: list) -> None:
    sents = sentences(cues, words)
    by_index: dict[int, str] = {}
    for sent_text, idxs in sents:
        for gi in idxs:
            by_index[gi] = sent_text
    for flag in flags:
        for gi in flag.global_indices:
            if gi in by_index:
                flag.context = by_index[gi]
                break
