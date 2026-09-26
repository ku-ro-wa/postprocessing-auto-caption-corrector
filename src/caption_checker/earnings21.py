"""Earnings-21 Auto-labelled corpus importer (ADR 0006).

Google's 2021 ASR output for each call becomes the transcript under test
(written as SRT), Rev's verbatim reference is the answer key, and every
disagreement between them becomes a Scored corpus case -- classified, and
dropped where it's alignment noise rather than a caption error. Everything
is derived into a gitignored cache and never committed: the dataset is
CC BY-SA 4.0. See CONTEXT.md's Auto-labelled corpus, Dev set and Held-out set.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable

from wordfreq import zipf_frequency

from caption_checker.models import Cue, DetectConfig
from caption_checker.normalize import clean, is_wordlike
from caption_checker.parser import parse, serialize, tokenize

REMOTE_BASE = (
    "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21/"
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "earnings21"

CAVEAT = (
    "Auto-labelled corpus: errors are Google's 2021 ASR, not today's YouTube; "
    "Rev's references are verbatim style; labels are noisy -- trust relative "
    "comparisons, not absolute numbers."
)

#: Entity types that make a disagreement a formatting difference, not a
#: misrecognition: "four" vs "4", "we're" vs "we are".
FORMAT_ENTITIES = frozenset(
    "CARDINAL YEAR MONEY PERCENT DATE TIME ORDINAL QUANTITY CONTRACTION "
    "ALPHANUMERIC ABBREVIATION WEBSITE".split()
)
#: Entity types that earn a case the extra ``entity`` tag.
NAMED_ENTITIES = frozenset("PERSON ORG PRODUCT GPE NORP".split())
FILLERS = frozenset("uh um uhm mm mhm hmm er ah eh huh".split())
#: Regions longer than this (normalised tokens, either side) are alignment
#: drift or crosstalk rather than a misheard phrase.
MAX_REGION_TOKENS = 6
#: Region labels that never become a case; counted in the build stats.
DROPPED = ("deletion", "filler", "drift")

#: Seed for drawing ``heldout-2`` for #28's comparison. The value is
#: arbitrary; it is fixed so anyone rebuilding the corpus gets the same calls.
HELDOUT2_SEED = 28

_MIN_CUE_SECONDS = 0.01
_NON_SPEECH_RE = re.compile(r"^(<[^>]*>|\*+)$")  # "<crosstalk>", "*"
_KEEP_RE = re.compile(r"[^\w'.]")
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class NlpToken:
    """One row of an Earnings-21 ``.nlp`` file. ``text`` is the token as it
    appears on screen (with its trailing punctuation); ``word`` is the bare
    token."""

    word: str
    speaker: str
    start: float | None
    end: float | None
    punctuation: str = ""
    entity_types: frozenset[str] = frozenset()

    @property
    def text(self) -> str:
        return self.word + self.punctuation


def parse_nlp(text: str, wer_tags: dict[str, dict] | None = None) -> list[NlpToken]:
    """Parse a hypothesis (``output/<vendor>/*.nlp``) or reference
    (``transcripts/nlp_references/*.nlp``) file. A reference's ``wer_tags``
    column is resolved to entity types through the call's
    ``.wer_tag.json`` map; non-speech markers are dropped."""
    wer_tags = wer_tags or {}
    rows = csv.DictReader(io.StringIO(text), delimiter="|", quoting=csv.QUOTE_NONE)
    tokens: list[NlpToken] = []
    for row in rows:
        word = row["token"]
        if not word or _NON_SPEECH_RE.match(word):
            continue
        tag_ids = ast.literal_eval(row["wer_tags"]) if row.get("wer_tags") else []
        tokens.append(
            NlpToken(
                word=word,
                speaker=row["speaker"],
                start=_seconds(row["ts"]),
                end=_seconds(row["endTs"]),
                punctuation=row["punctuation"] or "",
                entity_types=frozenset(
                    wer_tags[t]["entity_type"] for t in tag_ids if t in wer_tags
                ),
            )
        )
    return tokens


def _seconds(value: str | None) -> float | None:
    """A timing column, or None when it's empty or malformed (one Rev
    reference row carries its punctuation in ``endTs``)."""
    try:
        return float(value) if value else None
    except ValueError:
        return None


def to_cues(
    tokens: list[NlpToken], max_seconds: float = 7.0, max_chars: int = 42
) -> list[Cue]:
    """Group timed hypothesis tokens into Cues the way a caption track would:
    a new cue on a speaker change, after sentence-final punctuation, or
    before the cue would run past ``max_seconds`` or ``max_chars``."""
    groups: list[list[NlpToken]] = []
    for tok in tokens:
        current = groups[-1] if groups else None
        if current is not None:
            text = " ".join(t.text for t in [*current, tok])
            first, last = current[0], current[-1]
            if (
                tok.speaker != last.speaker
                or last.punctuation[-1:] in (".", "?", "!")
                or len(text) > max_chars
                or (_end(tok) - _start(first)) > max_seconds
            ):
                current = None
        if current is None:
            groups.append([tok])
        else:
            current.append(tok)
    # Cases address Words by position, so the SRT must read back in token
    # order: starts strictly increase (serialisation sorts by start, and one
    # Google output repeats a sentence with its original timings), and no cue
    # is zero-length (dropped on read-back).
    cues: list[Cue] = []
    previous = -_MIN_CUE_SECONDS
    for i, g in enumerate(groups, start=1):
        start = max(_start(g[0]), previous + _MIN_CUE_SECONDS)
        end = max(_end(g[-1]), start + _MIN_CUE_SECONDS)
        cues.append(
            Cue(
                index=i,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                text=" ".join(t.text for t in g),
            )
        )
        previous = start
    return cues


def _start(tok: NlpToken) -> float:
    return tok.start if tok.start is not None else (tok.end or 0.0)


def _end(tok: NlpToken) -> float:
    return tok.end if tok.end is not None else _start(tok)


def _normalise(word: str) -> list[str]:
    """Alignment tokens for one word: lowercased, punctuation stripped,
    hyphens split, so "Long-term," aligns with "long term"."""
    out = []
    for part in word.lower().replace("-", " ").split():
        part = _KEEP_RE.sub("", part).strip("'.")
        if part:
            out.append(part)
    return out


@dataclass
class _Region:
    """A run of disagreement, as indices into the original token lists."""

    hyp: set[int] = field(default_factory=set)
    ref: set[int] = field(default_factory=set)
    hyp_norm: list[str] = field(default_factory=list)
    ref_norm: list[str] = field(default_factory=list)


def _flatten(tokens: list[NlpToken]) -> tuple[list[str], list[int]]:
    norm: list[str] = []
    origin: list[int] = []
    for i, tok in enumerate(tokens):
        for part in _normalise(tok.word):
            norm.append(part)
            origin.append(i)
    return norm, origin


def _align(hyp: list[NlpToken], ref: list[NlpToken]) -> list[_Region]:
    """Non-equal regions between hypothesis and reference, widened to whole
    original tokens and merged where they share one (a hyphenated word split
    for alignment)."""
    import jiwer  # only the importer pays for it

    hyp_norm, hyp_origin = _flatten(hyp)
    ref_norm, ref_origin = _flatten(ref)
    if not hyp_norm or not ref_norm:
        raise ValueError("cannot align an empty hypothesis or reference")
    chunks = jiwer.process_words(" ".join(ref_norm), " ".join(hyp_norm)).alignments[0]

    regions: list[_Region] = []
    previous_equal = True
    for chunk in chunks:
        if chunk.type == "equal":
            previous_equal = True
            continue
        if previous_equal:
            regions.append(_Region())
        previous_equal = False
        region = regions[-1]
        for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
            region.hyp.add(hyp_origin[j])
            region.hyp_norm.append(hyp_norm[j])
        for j in range(chunk.ref_start_idx, chunk.ref_end_idx):
            region.ref.add(ref_origin[j])
            region.ref_norm.append(ref_norm[j])

    merged: list[_Region] = []
    for region in regions:
        last = merged[-1] if merged else None
        if last is not None and (
            (region.hyp and last.hyp and min(region.hyp) <= max(last.hyp))
            or (region.ref and last.ref and min(region.ref) <= max(last.ref))
        ):
            last.hyp |= region.hyp
            last.ref |= region.ref
            last.hyp_norm += region.hyp_norm
            last.ref_norm += region.ref_norm
        else:
            merged.append(region)
    return merged


def _classify(
    region: _Region,
    hyp: list[NlpToken],
    entity_types: frozenset[str],
    config: DetectConfig,
) -> str:
    """A region's ``kind``, or one of ``DROPPED`` when it isn't a case."""
    if not region.hyp:
        return "deletion"  # nothing on screen to flag
    if all(t in FILLERS for t in region.hyp_norm + region.ref_norm):
        return "filler"
    if max(len(region.hyp_norm), len(region.ref_norm)) > MAX_REGION_TOKENS:
        return "drift"
    hyp_joined = "".join(region.hyp_norm)
    ref_joined = "".join(region.ref_norm)
    if (
        entity_types & FORMAT_ENTITIES
        or _DIGIT_RE.search(hyp_joined + ref_joined)
        or hyp_joined == ref_joined
    ):
        return "format"
    if all(t in config.stopwords for t in region.hyp_norm + region.ref_norm):
        return "function-word"
    for i in region.hyp:
        word = hyp[i].word
        if is_wordlike(word) and clean(word) not in config.stopwords:
            if zipf_frequency(clean(word), "en") <= config.oov_zipf_max:
                return "non-word"
    return "real-word"


def _unique_context(tokens: list[str], start: int, end: int) -> str:
    """The shortest window of ``tokens`` around ``[start, end)`` that occurs
    exactly once in the file, with the span's first occurrence inside it at
    the right offset -- what the Scored corpus's ``context`` needs."""
    span = tokens[start:end]
    for extra in range(len(tokens)):
        for left in range(extra + 1):
            lo, hi = start - left, end + (extra - left)
            if lo < 0 or hi > len(tokens):
                continue
            window = tokens[lo:hi]
            if _first(window, span) != left:
                continue
            if _count(tokens, window) == 1:
                return " ".join(window)
    return " ".join(tokens)


def _first(tokens: list[str], needle: list[str]) -> int:
    n = len(needle)
    return next((i for i in range(len(tokens) - n + 1) if tokens[i : i + n] == needle), -1)


def _count(tokens: list[str], needle: list[str]) -> int:
    n, head = len(needle), needle[0]
    return sum(
        1
        for i in range(len(tokens) - n + 1)
        if tokens[i] == head and tokens[i : i + n] == needle
    )


@dataclass
class CallCorpus:
    """One call's contribution: the transcript under test and its cases."""

    cues: list[Cue]
    cases: list[dict]
    stats: Counter[str]


def build_call(
    source: str,
    hyp: list[NlpToken],
    ref: list[NlpToken],
    config: DetectConfig | None = None,
) -> CallCorpus:
    """Align one call's hypothesis against its reference and emit
    ``scored_corpus.json``-shaped cases: ``span`` is the hypothesis words as
    captioned, ``candidate`` the reference words, ``context`` the shortest
    hypothesis window unique in the file, ``entity`` whether the region
    touches a named entity."""
    config = config or DetectConfig()
    cleaned = [clean(t.text) for t in hyp]
    cases: list[dict] = []
    stats: Counter[str] = Counter()
    for region in _align(hyp, ref):
        entity_types = frozenset(t for i in region.ref for t in ref[i].entity_types)
        kind = _classify(region, hyp, entity_types, config)
        stats[kind] += 1
        if kind in DROPPED:
            continue
        lo, hi = min(region.hyp), max(region.hyp) + 1
        case: dict = {
            "source": source,
            "span": " ".join(t.text for t in hyp[lo:hi]),
            "verdict": "should-flag",
        }
        if region.ref:
            case["candidate"] = " ".join(
                ref[i].word for i in range(min(region.ref), max(region.ref) + 1)
            )
        case["context"] = _unique_context(cleaned, lo, hi)
        case["kind"] = kind
        case["entity"] = bool(entity_types & NAMED_ENTITIES)
        cases.append(case)
    return CallCorpus(cues=to_cues(hyp), cases=cases, stats=stats)


def _check_round_trip(path: Path, hyp: list[NlpToken]) -> None:
    """Cases address the SRT's Words by position, so the written file must
    read back as exactly the hypothesis tokens."""
    words = [w.text for w in tokenize(parse(path))]
    expected = [t.text for t in hyp]
    if words != expected:
        at = next(
            (i for i, (a, b) in enumerate(zip(words, expected)) if a != b),
            min(len(words), len(expected)),
        )
        raise ValueError(f"{path.name}: SRT round trip diverges at word {at}")


Fetch = Callable[[str], bytes]


def fetch_remote(path: str) -> bytes:
    """Download one file of the dataset's ``earnings21/`` directory."""
    import ssl
    from urllib.request import urlopen

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(REMOTE_BASE + path, context=context, timeout=60) as resp:
        return resp.read()


def _cached(cache_dir: Path, path: str, fetch: Fetch) -> str:
    local = cache_dir / "raw" / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(fetch(path))
    return local.read_text(encoding="utf-8")


def _metadata(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def build(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    fetch: Fetch = fetch_remote,
    dev_size: int = 5,
    heldout2_size: int = 10,
) -> dict[str, dict]:
    """Build the Held-out (``eval-10``), Dev (the first ``dev_size`` other
    calls, in metadata order) and second Held-out (``heldout-2``: a draw of
    ``heldout2_size`` of the calls left, by ID, seeded with ``HELDOUT2_SEED``)
    corpora under ``cache_dir``. Raw downloads are cached under ``raw/``;
    each split directory is rebuilt from them and holds one SRT per call,
    ``cases.json`` and a ``manifest.json`` listing every source with its
    Priming terms (the company name). Remaining calls are held back and never
    fetched. Returns per-split summary counts."""
    calls = _metadata(_cached(cache_dir, "earnings21-file-metadata.csv", fetch))
    eval10 = {
        row["file_id"]
        for row in _metadata(_cached(cache_dir, "eval10-file-metadata.csv", fetch))
    }
    rest = [row for row in calls if row["file_id"] not in eval10]
    dev = rest[:dev_size]
    left = sorted(row["file_id"] for row in rest[dev_size:])
    if heldout2_size > len(left):
        raise ValueError(
            f"heldout-2 needs {heldout2_size} calls but only {len(left)} are left"
        )
    # Drawn from sorted IDs, so the upstream CSV's row order can't change it.
    drawn = set(random.Random(HELDOUT2_SEED).sample(left, heldout2_size))
    splits = {
        "heldout": [row for row in calls if row["file_id"] in eval10],
        "dev": dev,
        "heldout-2": [row for row in calls if row["file_id"] in drawn],
    }

    summary: dict[str, dict] = {}
    for split, rows in splits.items():
        # Built aside and swapped in, so a failure never leaves a half-built
        # split for the eval command to score.
        out = cache_dir / f".{split}.partial"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        cases: list[dict] = []
        stats: Counter[str] = Counter()
        sources: dict[str, dict] = {}
        for row in rows:
            call_id = row["file_id"]
            source = f"{call_id}.srt"
            tags = json.loads(
                _cached(cache_dir, f"transcripts/wer_tags/{call_id}.wer_tag.json", fetch)
            )
            hyp = parse_nlp(_cached(cache_dir, f"output/google/{call_id}.nlp", fetch))
            ref = parse_nlp(
                _cached(cache_dir, f"transcripts/nlp_references/{call_id}.nlp", fetch),
                tags,
            )
            call = build_call(source, hyp, ref)
            (out / source).write_text(serialize(call.cues, format="srt"), encoding="utf-8")
            _check_round_trip(out / source, hyp)
            cases.extend(call.cases)
            stats.update(call.stats)
            sources[source] = {
                "priming_terms": [row["company_name"]],
                "sector": row["sector"],
            }
        (out / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(
            json.dumps(
                {"caveat": CAVEAT, "sources": sources, "stats": dict(sorted(stats.items()))},
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.rmtree(cache_dir / split, ignore_errors=True)
        out.rename(cache_dir / split)
        summary[split] = {"calls": len(rows), "cases": len(cases), **stats}
    return summary
