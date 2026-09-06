"""Domain vocabulary loading and phonetic indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from caption_checker.normalize import clean
from caption_checker.phonetics import codes

DEFAULT_VOCAB_PATH = Path(__file__).parent / "data" / "domain_vocab.txt"


@dataclass
class Vocab:
    """A set of known-good domain terms plus a phonetic-code index into them."""

    #: cleaned single-token surface forms (for exact membership checks)
    terms: set[str] = field(default_factory=set)
    #: original-cased display forms, keyed by cleaned form
    display: dict[str, str] = field(default_factory=dict)
    #: phonetic code -> original-cased terms that encode to it (single tokens)
    by_phonetic: dict[str, set[str]] = field(default_factory=dict)
    #: joined phonetic code -> original-cased multi-word terms ("leaderelection")
    phrases_by_phonetic: dict[str, set[str]] = field(default_factory=dict)

    def __contains__(self, token: str) -> bool:
        return clean(token) in self.terms


def _read_terms(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def load_vocab(
    extra: Path | str | None = None, *, algo: str = "metaphone"
) -> Vocab:
    """Load the bundled default vocabulary, merged with an optional user file."""
    raw_terms = _read_terms(DEFAULT_VOCAB_PATH)
    if extra is not None:
        raw_terms.extend(_read_terms(Path(extra)))

    vocab = Vocab()
    for term in raw_terms:
        parts = term.split()
        if len(parts) == 1:
            cleaned = clean(term)
            if not cleaned:
                continue
            vocab.terms.add(cleaned)
            vocab.display.setdefault(cleaned, term)
            for c in codes(term, algo=algo):
                vocab.by_phonetic.setdefault(c, set()).add(term)
        else:
            joined = "".join(clean(p) for p in parts)
            for c in codes(joined, algo=algo):
                vocab.phrases_by_phonetic.setdefault(c, set()).add(term)

    return vocab
