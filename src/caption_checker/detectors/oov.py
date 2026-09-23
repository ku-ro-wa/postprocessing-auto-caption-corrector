"""Out-of-vocabulary detector: tokens that are neither common English nor a
known domain term. Cheap first-pass filter (catches "cubernetes", "sensus")."""

from __future__ import annotations

from wordfreq import zipf_frequency

from caption_checker.models import DETECTOR_OOV, Cue, DetectConfig, Flag, Word
from caption_checker.normalize import clean, is_wordlike
from caption_checker.vocab import DocVocab, Vocab

from .base import index_cues, make_flag


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    *,
    doc_vocab: DocVocab | None = None,
    **_context: object,
) -> list[Flag]:
    cues_by_index = index_cues(cues)
    doc_vocab = doc_vocab or DocVocab()
    flags: list[Flag] = []
    for word in words:
        if not is_wordlike(word.text):
            continue
        cleaned = clean(word.text)
        if len(cleaned) < config.min_token_len:
            continue
        if cleaned in config.stopwords or cleaned in vocab.terms or cleaned in doc_vocab:
            continue
        zipf = zipf_frequency(cleaned, "en")
        if zipf > config.oov_zipf_max:
            continue
        # 0.0 -> 0.9 confidence; anything with a faint frequency signal lower.
        confidence = 0.9 if zipf == 0.0 else max(0.4, 0.9 - zipf / 3.0)
        suspects = doc_vocab.suspects.get(cleaned, [])
        reason = f'"{word.text}" is not a common word or known term' + (
            "" if zipf == 0.0 else f" (rare, zipf {zipf:.1f})"
        )
        if suspects:
            reason += f'; it recurs, but always like a misheard "{suspects[0]}"'
        flags.append(
            make_flag(
                [word],
                cues_by_index,
                detector=DETECTOR_OOV,
                reason=reason,
                confidence=confidence,
                candidates=suspects,
            )
        )
    return flags
