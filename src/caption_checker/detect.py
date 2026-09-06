"""Detection orchestrator: run every detector, merge overlapping flags, attach
sentence context."""

from __future__ import annotations

from caption_checker.detectors import LEXICAL_DETECTORS, context_embedding
from caption_checker.models import Cue, DetectConfig, Flag
from caption_checker.normalize import sentences
from caption_checker.parser import tokenize
from caption_checker.vocab import Vocab, load_vocab


def detect(
    cues: list[Cue],
    *,
    vocab: Vocab | None = None,
    config: DetectConfig | None = None,
) -> list[Flag]:
    config = config or DetectConfig()
    vocab = vocab or load_vocab(algo=config.phonetic_algo)
    words = tokenize(cues)

    raw: list[Flag] = []
    for detector in LEXICAL_DETECTORS:
        raw.extend(detector.find(words, cues, vocab, config))

    if config.enable_embeddings:
        raw.extend(
            context_embedding.find(words, cues, vocab, config, existing=list(raw))
        )

    merged = _merge(raw)
    _attach_context(merged, cues, words)
    return merged


def _merge(flags: list[Flag]) -> list[Flag]:
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
        (_combine(group) for group in groups),
        key=lambda f: (f.cue_index, min(f.global_indices)),
    )


def _combine(group: list[Flag]) -> Flag:
    if len(group) == 1:
        return group[0]

    indices = sorted({i for f in group for i in f.global_indices})
    detectors = sorted({d for f in group for d in f.detector.split("+")})
    base = max(group, key=lambda f: len(f.global_indices))

    candidates: list[str] = []
    for flag in sorted(group, key=lambda f: f.confidence, reverse=True):
        for cand in flag.candidates:
            if cand not in candidates:
                candidates.append(cand)

    reasons = "; ".join(
        dict.fromkeys(f.reason for f in group)  # dedup, keep order
    )
    agree_bonus = 0.05 * (len(detectors) - 1)
    confidence = min(0.98, max(f.confidence for f in group) + agree_bonus)

    return Flag(
        span=base.span,
        global_indices=indices,
        cue_index=min(f.cue_index for f in group),
        start=min(f.start for f in group),
        end=max(f.end for f in group),
        detector="+".join(detectors),
        reason=reasons,
        candidates=candidates,
        confidence=round(confidence, 3),
    )


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
