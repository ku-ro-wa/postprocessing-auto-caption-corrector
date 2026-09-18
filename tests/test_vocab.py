from __future__ import annotations

from pathlib import Path

from caption_checker.models import DetectConfig
from caption_checker.parser import parse, tokenize
from caption_checker.vocab import build_doc_vocab, load_vocab

DATA_DIR = Path(__file__).parent / "data"
NO_EMBED = DetectConfig(enable_embeddings=False)


def test_build_doc_vocab_clusters_recurring_misspellings() -> None:
    """A deterministic ASR mistake repeats too, so "Kashi" (x3) and "Caushi"
    (x3) each independently clear the recurrence threshold alongside the
    correct "Kalshi" (x7). Only the most frequent spelling should survive as
    its own canonical entry; the rest should fold into it."""
    cues = parse(DATA_DIR / "doc_vocab_sample.srt")
    words = tokenize(cues)
    vocab = load_vocab()

    doc_vocab = build_doc_vocab(words, vocab, NO_EMBED)

    assert doc_vocab.counts == {"kalshi": 13}
    assert doc_vocab.variants["kashi"] == "kalshi"
    assert doc_vocab.variants["caushi"] == "kalshi"
    assert doc_vocab.display["kalshi"] == "Kalshi"


def test_build_doc_vocab_ignores_one_off_terms() -> None:
    cues = parse(DATA_DIR / "doc_vocab_sample.srt")
    words = tokenize(cues)
    vocab = load_vocab()

    doc_vocab = build_doc_vocab(words, vocab, NO_EMBED)

    # "Polymarket" only appears once in the fixture: not enough recurrence
    # to be trusted as this document's own vocabulary.
    assert "polymarket" not in doc_vocab.counts
    assert "polymarket" not in doc_vocab.variants
