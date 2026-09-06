"""Split-word detector: an ASR term broken across adjacent tokens
("con sensus" -> "consensus", "cough ka" -> "Kafka").

Slides a 2..N token window within a single cue, concatenates the cleaned
tokens, and flags when the join either sounds exactly like a known domain term
or is itself a common English word.
"""

from __future__ import annotations

from wordfreq import zipf_frequency

from caption_checker.models import (
    DETECTOR_SPLIT_WORD,
    Cue,
    DetectConfig,
    Flag,
    Word,
)
from caption_checker.normalize import clean, is_wordlike
from caption_checker.phonetics import codes, similar
from caption_checker.vocab import Vocab

from .base import index_cues, make_flag, span_text


def _looks_broken(part: str, config: DetectConfig) -> bool:
    return (
        len(part) <= config.split_component_max_len
        or zipf_frequency(part, "en") < config.known_good_zipf_min
    )


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    **_context: object,
) -> list[Flag]:
    cues_by_index = index_cues(cues)
    flags: list[Flag] = []
    consumed: set[int] = set()

    for size in range(2, config.split_max_window + 1):
        for i in range(len(words) - size + 1):
            window = words[i : i + size]
            idxs = {w.global_index for w in window}
            if idxs & consumed:
                continue
            if len({w.cue_index for w in window}) != 1:
                continue
            if not all(is_wordlike(w.text) for w in window):
                continue

            parts = [clean(w.text) for w in window]
            if any(not p for p in parts):
                continue
            if any(p in config.stopwords for p in parts):
                continue
            if not all(_looks_broken(p, config) for p in parts):
                continue

            join = "".join(parts)
            if len(join) <= config.min_token_len:
                continue

            flag = _match(window, join, parts, vocab, config, cues_by_index)
            if flag is not None:
                flags.append(flag)
                consumed |= idxs

    return flags


def _match(
    window: list[Word],
    join: str,
    parts: list[str],
    vocab: Vocab,
    config: DetectConfig,
    cues_by_index: dict[int, Cue],
) -> Flag | None:
    disp = span_text(window)
    join_codes = codes(join, algo=config.phonetic_algo)

    # (a) sounds exactly like a curated domain term / phrase
    hits: set[str] = set()
    for code in join_codes:
        hits |= vocab.by_phonetic.get(code, set())
        hits |= vocab.phrases_by_phonetic.get(code, set())
    hits = {h for h in hits if "".join(clean(p) for p in h.split()) != join}
    if hits:
        ranked = sorted(
            hits,
            key=lambda t: similar(join, "".join(clean(p) for p in t.split())),
            reverse=True,
        )
        return make_flag(
            window,
            cues_by_index,
            detector=DETECTOR_SPLIT_WORD,
            reason=f'"{disp}" joined sounds like "{ranked[0]}"',
            candidates=ranked,
            confidence=0.82,
        )

    # (b) the join is itself a common word the ASR split in two
    if zipf_frequency(join, "en") >= config.split_common_zipf_min and any(
        zipf_frequency(p, "en") < config.known_good_zipf_min for p in parts
    ):
        return make_flag(
            window,
            cues_by_index,
            detector=DETECTOR_SPLIT_WORD,
            reason=f'"{disp}" is likely the single word "{join}"',
            candidates=[join],
            confidence=0.75,
        )

    return None
