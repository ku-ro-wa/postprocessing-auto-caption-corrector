"""Domain vocabulary loading and phonetic indexing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

from caption_checker.models import DetectConfig, Word
from caption_checker.normalize import clean, is_wordlike, trim_edges
from caption_checker.phonetics import codes, similar

DEFAULT_VOCAB_PATH = Path(__file__).parent / "data" / "domain_vocab.txt"

# Deep enough to reach names and brands like "shreyas" (rank ~136k).
_KNOWN_WORDS_TOP_N = 150_000


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


@dataclass
class DocVocab:
    """Terms that recur often enough in one specific transcript, spelled
    consistently, to trust even though wordfreq / the curated domain vocab
    don't know them — e.g. a channel's recurring brand, guest, or company
    name. Built once per transcript by ``build_doc_vocab``."""

    #: canonical cleaned surface form -> number of occurrences (merged
    #: across every near-duplicate misspelling clustered into it)
    counts: dict[str, int] = field(default_factory=dict)
    #: canonical cleaned surface form -> the first original-cased spelling seen
    display: dict[str, str] = field(default_factory=dict)
    #: every raw candidate spelling that hit the recurrence threshold
    #: (canonical forms included) -> the canonical term it was clustered
    #: into. Lets fuzzy-matching bridge through an intermediate misspelling
    #: ("Caushi" -> "Kashi" -> "Kalshi") even when the two ends aren't
    #: similar enough to match directly.
    variants: dict[str, str] = field(default_factory=dict)
    #: recurring terms *not* trusted because a known word could explain them
    #: as a consistent misspelling (cleaned surface form -> those known
    #: words, cased like the transcript's spelling). Offered as corrections.
    suspects: dict[str, list[str]] = field(default_factory=dict)

    def __contains__(self, cleaned: str) -> bool:
        return cleaned in self.counts

    def __bool__(self) -> bool:
        return bool(self.counts)

    def __iter__(self):
        return iter(self.counts)


def build_doc_vocab(
    words: list[Word],
    vocab: Vocab,
    config: DetectConfig,
    *,
    min_count: int = 3,
) -> DocVocab:
    """Count OOV tokens across the whole transcript; anything spelled the
    same way at least ``min_count`` times is probably this document's own
    term rather than a one-off ASR fluke, so later detectors can treat it as
    known-good and as a correction candidate for near-miss misspellings.

    A deterministic ASR mistake repeats too, so several near-duplicate
    misspellings of the same term can each independently clear the
    recurrence threshold ("Kashi" x4, "Caushi" x4, alongside the correct
    "Kalshi" x10). Candidates are merged into clusters (highest count first,
    Jaro-Winkler similarity, transitive) so only the most frequent spelling
    in each cluster is trusted as canonical; the rest stay correctable.
    """
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for word in words:
        if not is_wordlike(word.text):
            continue
        cleaned = clean(word.text)
        if len(cleaned) < config.min_token_len or cleaned in config.stopwords:
            continue
        if cleaned in vocab.terms or zipf_frequency(cleaned, "en") >= config.known_good_zipf_min:
            continue  # already known-good; doc_vocab is only for the unknowns
        counts[cleaned] += 1
        display.setdefault(cleaned, trim_edges(word.text) or word.text)

    candidates = sorted(
        (c for c, n in counts.items() if n >= min_count),
        key=lambda c: -counts[c],
    )

    canonical_counts: dict[str, int] = {}
    canonical_display: dict[str, str] = {}
    owner: dict[str, str] = {}
    seen_order: list[str] = []
    for cand in candidates:
        match = next(
            (
                owner[seen]
                for seen in seen_order
                if similar(cand, seen) >= config.doc_vocab_fuzzy_min
            ),
            None,
        )
        if match is None:
            canonical_counts[cand] = counts[cand]
            canonical_display[cand] = display[cand]
            owner[cand] = cand
        else:
            canonical_counts[match] += counts[cand]
            owner[cand] = match
        seen_order.append(cand)

    # A low-count cluster whose canonical spelling is a near-miss of a word
    # wordfreq knows is more likely that word, misheard the same way each
    # time ("Corsera" x3 -> "Coursera"). A term with no such neighbour
    # ("Polymarket"), or one repeated enough to outweigh it ("Kalshi" x20 vs
    # "kalish"), stays trusted.
    suspects: dict[str, list[str]] = {}
    for canon, n in list(canonical_counts.items()):
        if n >= config.doc_vocab_suspect_min_count:
            continue
        if zipf_frequency(canon, "en") > config.oov_zipf_max:
            continue
        neighbours = known_neighbours(canon, config)
        if not neighbours:
            continue
        shown = canonical_display.pop(canon)
        del canonical_counts[canon]
        if shown[:1].isupper():
            neighbours = [n[:1].upper() + n[1:] for n in neighbours]
        for variant in [v for v, o in owner.items() if o == canon]:
            del owner[variant]
            suspects[variant] = neighbours

    return DocVocab(
        counts=canonical_counts,
        display=canonical_display,
        variants=owner,
        suspects=suspects,
    )


@lru_cache(maxsize=1)
def _known_words_by_phonetic() -> dict[str, list[str]]:
    """Phonetic code -> wordfreq's most common English words encoding to it.
    Built lazily (~0.6s) the first time a doc-vocab term needs checking."""
    index: dict[str, list[str]] = {}
    for word in top_n_list("en", _KNOWN_WORDS_TOP_N):
        if not word.replace("'", "").isalpha():
            continue
        for c in codes(word):
            index.setdefault(c, []).append(word)
    return index


def known_neighbours(term: str, config: DetectConfig, limit: int = 3) -> list[str]:
    """The common English words ``term`` could be a mishearing of, most
    similar first. Similarity alone can't pick the right one ("corser" edges
    out "coursera"), so callers pass a few on for the LLM to choose from in
    context. Always matches on Double Metaphone, whatever
    ``config.phonetic_algo`` says, since that's how the index is built."""
    index = _known_words_by_phonetic()
    scored: dict[str, float] = {}
    for c in codes(term):
        for word in index.get(c, []):
            if word == term:
                continue
            score = similar(term, word)
            if score < config.known_neighbour_jw_min:
                continue
            if zipf_frequency(word, "en") < config.known_neighbour_zipf_min:
                continue
            scored[word] = score
    return sorted(scored, key=lambda w: -scored[w])[:limit]


def possessive_base(cleaned: str) -> str | None:
    """``"openai"`` for ``"openai's"`` (straight or curly apostrophe), else
    None. wordfreq and the vocab list only hold base forms, so a possessive
    of a known term would otherwise read as out-of-vocabulary."""
    for suffix in ("'s", "’s"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            return cleaned[: -len(suffix)]
    return None


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
