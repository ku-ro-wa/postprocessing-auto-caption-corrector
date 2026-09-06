"""Context-embedding detector (soft tier): flags words that are a poor semantic
fit for the sentence they sit in, using a local sentence-transformers model.

Heuristic and lower-confidence by design; it mostly reinforces the lexical
detectors. Disabled by ``--no-embeddings`` or when the optional dependency is
missing.
"""

from __future__ import annotations

import sys

from wordfreq import zipf_frequency

from caption_checker.models import (
    DETECTOR_CONTEXT_EMBEDDING,
    Cue,
    DetectConfig,
    Flag,
    Word,
)
from caption_checker.normalize import clean, is_wordlike, sentences
from caption_checker.vocab import Vocab

from .base import index_cues, make_flag

_MODEL_CACHE: dict[str, object] = {}
_WARNED = False


def _load_model(name: str):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def find(
    words: list[Word],
    cues: list[Cue],
    vocab: Vocab,
    config: DetectConfig,
    *,
    existing: list[Flag] | None = None,
    **_context: object,
) -> list[Flag]:
    global _WARNED
    if not config.enable_embeddings:
        return []
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        if not _WARNED:
            print(
                "caption-checker: sentence-transformers not installed; skipping "
                "the context-embedding detector. Install the 'embeddings' extra "
                "or pass --no-embeddings to silence this.",
                file=sys.stderr,
            )
            _WARNED = True
        return []

    import numpy as np

    sents = sentences(cues, words)
    if len(sents) < 3:
        return []

    already: set[int] = set()
    for flag in existing or []:
        already.update(flag.global_indices)

    words_by_index = {w.global_index: w for w in words}
    candidates: list[tuple[Word, str]] = []
    for sent_text, idxs in sents:
        for gi in idxs:
            word = words_by_index.get(gi)
            if word is None or not is_wordlike(word.text):
                continue
            cleaned = clean(word.text)
            if (
                len(cleaned) < config.min_token_len
                or cleaned in config.stopwords
                or cleaned in vocab.terms
            ):
                continue
            rare = (
                zipf_frequency(cleaned, "en")
                <= config.embedding_candidate_zipf_max
            )
            if rare or gi in already:
                candidates.append((word, sent_text))

    if len(candidates) < 3:
        return []

    model = _load_model(config.embedding_model)
    word_texts = [clean(w.text) for w, _ in candidates]
    ctx_texts = [
        (sent.replace(w.text, "", 1).strip() or sent) for w, sent in candidates
    ]
    emb_w = model.encode(word_texts, normalize_embeddings=True)
    emb_c = model.encode(ctx_texts, normalize_embeddings=True)
    sims = np.sum(np.asarray(emb_w) * np.asarray(emb_c), axis=1)

    mean = float(sims.mean())
    std = float(sims.std())
    if std < 1e-6:
        return []

    cues_by_index = index_cues(cues)
    flags: list[Flag] = []
    for (word, _sent), sim in zip(candidates, sims):
        z = (float(sim) - mean) / std
        if z <= config.embedding_sim_z:
            flags.append(
                make_flag(
                    [word],
                    cues_by_index,
                    detector=DETECTOR_CONTEXT_EMBEDDING,
                    reason=f'"{word.text}" is a weak semantic fit for its '
                    f"sentence (context z={z:.1f})",
                    candidates=[],
                    confidence=min(0.6, 0.3 + 0.1 * abs(z)),
                )
            )
    return flags
