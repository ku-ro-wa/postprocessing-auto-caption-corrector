from __future__ import annotations

from pathlib import Path

from caption_checker.models import DetectConfig
from caption_checker.parser import parse, tokenize
from caption_checker.vocab import build_doc_vocab, load_vocab

DATA_DIR = Path(__file__).parent / "data"


def test_build_doc_vocab_clusters_recurring_misspellings() -> None:
    """A deterministic ASR mistake repeats too, so "Kashi" (x3) and "Caushi"
    (x3) each independently clear the recurrence threshold alongside the
    correct "Kalshi" (x7). Only the most frequent spelling should survive as
    its own canonical entry; the rest should fold into it."""
    cues = parse(DATA_DIR / "doc_vocab_sample.srt")
    words = tokenize(cues)
    vocab = load_vocab()

    doc_vocab = build_doc_vocab(words, vocab, DetectConfig())

    assert doc_vocab.counts == {"kalshi": 13}
    assert doc_vocab.variants["kashi"] == "kalshi"
    assert doc_vocab.variants["caushi"] == "kalshi"
    assert doc_vocab.display["kalshi"] == "Kalshi"


def test_build_doc_vocab_ignores_one_off_terms() -> None:
    cues = parse(DATA_DIR / "doc_vocab_sample.srt")
    words = tokenize(cues)
    vocab = load_vocab()

    doc_vocab = build_doc_vocab(words, vocab, DetectConfig())

    # "Polymarket" only appears once in the fixture: not enough recurrence
    # to be trusted as this document's own vocabulary.
    assert "polymarket" not in doc_vocab.counts
    assert "polymarket" not in doc_vocab.variants


def _srt(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "doc.srt"
    path.write_text(f"1\n00:00:00,000 --> 00:00:09,000\n{text}\n", encoding="utf-8")
    return path


def test_build_doc_vocab_distrusts_near_miss_of_a_known_word(tmp_path: Path) -> None:
    """"Corsera" x3 is a consistent mishearing of "Coursera", which wordfreq
    knows, so it shouldn't be trusted as the doc's own term. "Polymarket" x3
    has no such neighbour and stays trusted."""
    text = " ".join(["I took a Corsera course on Polymarket."] * 3)
    words = tokenize(parse(_srt(tmp_path, text)))

    doc_vocab = build_doc_vocab(words, load_vocab(), DetectConfig())

    assert "polymarket" in doc_vocab
    assert "corsera" not in doc_vocab
    assert "corsera" not in doc_vocab.variants
    assert "Coursera" in doc_vocab.suspects["corsera"]


def test_build_doc_vocab_trusts_near_miss_once_it_recurs_enough(tmp_path: Path) -> None:
    """"Kalshi" sounds like the rarer "kalish", but repeated this often it's
    the doc's own term, not a misspelling."""
    config = DetectConfig()
    text = " ".join(["Bets on Kalshi."] * config.doc_vocab_suspect_min_count)
    words = tokenize(parse(_srt(tmp_path, text)))

    doc_vocab = build_doc_vocab(words, load_vocab(), config)

    assert "kalshi" in doc_vocab
    assert not doc_vocab.suspects
