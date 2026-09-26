"""The Read-through (ADR 0006): an LLM pass that reads the whole transcript in
chunks and returns Flags with their Corrections in one call per chunk --
including errors no Detector raised.

Each chunk goes out as numbered Words (``[12]word``), with the local
detectors' Flags in it as hints (every hint gets a verdict back) and the run's
Priming terms. The reply names spans by word index, so every result maps back
onto real Words and becomes an ordinary :class:`~caption_checker.models.Flag`
plus :class:`~caption_checker.corrector.Correction` -- apply, sidecar and
review don't know the difference. Not a Detector: it depends on their output.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

from caption_checker.corrector import (
    Correction,
    CorrectorError,
    OpenRouterClient,
    Spend,
)
from caption_checker.detectors.base import index_cues, make_flag
from caption_checker.models import DEFAULT_MODEL, Cue, Flag, Word
from caption_checker.normalize import clean, sentences
from caption_checker.parser import tokenize

DETECTOR_READ_THROUGH = "read_through"

#: Words per chunk before the next sentence end closes it. Big enough that a
#: chunk carries its own topic, small enough that the model reads every word.
CHUNK_WORDS = 400
#: Plain-text lead-in and tail around each chunk, for context only.
CONTEXT_WORDS = 40
#: Concurrent requests per transcript.
WORKERS = 8
#: An error the Read-through found by itself (not a hint's verdict) is kept
#: only at or above this confidence. Set on the Scored corpus (Dev): below
#: it, new finds were mostly style and grammar edits (28 of 33).
MIN_CONFIDENCE = 0.9


# --- data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Hint:
    """A local Flag inside a chunk, as the model sees it."""

    id: str
    start: int
    end: int
    span: str
    candidates: list[str]
    reason: str

    def as_dict(self) -> dict:
        return {
            "hint": self.id,
            "start": self.start,
            "end": self.end,
            "span": self.span,
            "candidates": self.candidates,
            "reason": self.reason,
        }


@dataclass
class ChunkRequest:
    """One request: numbered ``(global index, text)`` Words to read, plain
    context either side, the hints inside it and the Priming terms."""

    words: list[tuple[int, str]]
    hints: list[Hint] = field(default_factory=list)
    before: str = ""
    after: str = ""
    priming_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkVerdict:
    """One item of a reply: a word range, and a replacement or None (not an
    error). ``hint`` names the hint it answers, if any."""

    start: int
    end: int
    replacement: str | None
    confidence: float
    rationale: str = ""
    hint: str | None = None


@dataclass
class ReadItem:
    """A Flag the Read-through returned. ``correction`` is None only when its
    chunk failed twice -- the hint is kept, unjudged. ``hint`` is the local
    Flag it answers (``flag`` itself unless the verdict widened it), or None
    for an error the Read-through found by itself."""

    flag: Flag
    correction: Correction | None
    hint: Flag | None = None


@dataclass
class ReadThroughResult:
    items: list[ReadItem]
    calls: int
    chunk_count: int
    failed_chunks: int = 0


class Reader(Protocol):
    @property
    def spend(self) -> Spend:
        """What the reader's requests have cost so far."""
        ...

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]: ...


# --- chunking --------------------------------------------------------------


def plan_chunks(
    cues: list[Cue],
    flags: list[Flag],
    *,
    priming_terms: Sequence[str] = (),
    chunk_words: int = CHUNK_WORDS,
    context_words: int = CONTEXT_WORDS,
) -> list[ChunkRequest]:
    """Cut the transcript into chunks of about ``chunk_words`` Words, each
    closed at the next sentence end (or at twice the size, if none comes),
    and never splitting a hint. A hint rides with the chunk its first Word is
    in."""
    words = tokenize(cues)
    starts_at = {min(f.global_indices): f for f in flags}
    reach_of = {min(f.global_indices): max(f.global_indices) for f in flags}

    bounds: list[tuple[int, int]] = []  # [lo, hi) into words
    lo = 0
    while lo < len(words):
        hi, reach = lo, lo
        while hi < len(words):
            reach = max(reach, reach_of.get(words[hi].global_index, 0))
            hi += 1
            size = hi - lo
            if words[hi - 1].global_index < reach:
                continue  # inside a hint
            if size >= 2 * chunk_words:
                break
            if size >= chunk_words and words[hi - 1].text.endswith((".", "?", "!")):
                break
        bounds.append((lo, hi))
        lo = hi

    ordered = sorted(flags, key=lambda f: min(f.global_indices))
    hint_id = {id(f): f"h{i}" for i, f in enumerate(ordered)}
    chunks: list[ChunkRequest] = []
    for lo, hi in bounds:
        inside = words[lo:hi]
        hints = [
            _hint(hint_id[id(starts_at[w.global_index])], starts_at[w.global_index])
            for w in inside
            if w.global_index in starts_at
        ]
        chunks.append(
            ChunkRequest(
                words=[(w.global_index, w.text) for w in inside],
                hints=hints,
                before=" ".join(w.text for w in words[max(0, lo - context_words) : lo]),
                after=" ".join(w.text for w in words[hi : hi + context_words]),
                priming_terms=list(priming_terms),
            )
        )
    return chunks


def _hint(hid: str, flag: Flag) -> Hint:
    return Hint(
        id=hid,
        start=min(flag.global_indices),
        end=max(flag.global_indices),
        span=flag.span,
        candidates=list(flag.candidates),
        reason=flag.reason,
    )


# --- prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You proofread a transcript made by automatic speech recognition (ASR). ASR
errors are words the recogniser misheard: a wrong real word that sounds like
the right one ("stationary" for "stationery", "Jensen Hang" for "Jensen
Huang"), a garbled name or term ("cubernetes"), a word split in two or two
words run together ("con sensus", "chad GPT" for "ChatGPT"), or a dropped or
inserted short word that breaks the sentence.

You get one chunk of the transcript as numbered words, "[index]word", with a
little plain text before and after it for context only. You also get:
- priming terms: names and terms known to occur in this recording. A span
  that sounds like one is very likely a misrecognition of it.
- hints: spans a cheap detector flagged, with its reason and candidate
  fixes. Hints are often false alarms; judge each one on its merits.

Find every error in the numbered words. An ASR error is an acoustic
confusion: the replacement must sound like the words it replaces when spoken
aloud, and the transcribed words must be something a person would not
plausibly have said there. Spontaneous speech is messy -- people misspeak,
use the wrong tense or number, drop articles, restart sentences -- and none
of that is an ASR error. Do NOT flag or change:
- the speaker's grammar: singular/plural, tense, agreement, missing or extra
  articles and prepositions ("expectation" for "expectations", "child" for
  "a child", "publish" for "published");
- the speaker's word choice, when the words make sense as said
  ("economical" for "economic", "misnomer" for "misconception");
- disfluencies, fillers ("um", "uh"), repetitions and false starts;
- spelling variants, hyphenation, punctuation, capitalisation, contractions
  or number formatting;
- words that are plausible as spoken even if unfamiliar to you (a new
  product, model, company or person) -- fix a name only when it is a clear
  mishearing, ideally of a priming term;
- anything in the before/after text.
Never add words the speaker did not say (not "Jensen" -> "Jensen Huang",
not "business advice" -> "or business advice"). When unsure, leave it
alone: a wrong flag costs a reviewer's time.

Reply with ONLY a JSON object {"verdicts": [...]}, each verdict a JSON
array of eight fields, in this order:
  [start, end, span, replacement, cause, confidence, hint, why]
- start, end: first and last word index (inclusive);
- span: those words exactly as written;
- replacement: what the speaker said, or null when it is not an error;
- cause: "misheard" when the recogniser wrote something other than what was
  said; "grammar" when the words are what the speaker said but
  ungrammatical; "style" for anything else. Only a "misheard" change is a
  correction;
- confidence: 0..1;
- hint: the hint id this verdict answers, or null;
- why: under ten words.
Give exactly one verdict per hint, with its id as "hint"; its replacement is
null when the hint is not an error, and its start/end may widen the hint to
cover the whole error. Beyond the hints, list ONLY errors you find, with a
null hint -- never list words you judged correct. The replacement replaces
exactly the words from start to end -- keep surrounding punctuation out.

Example chunk: [0]We [1]run [2]it [3]on [4]cubernetes, [5]and [6]the [7]con
[8]sensus [9]layer [10]is [11]raft. Hints: [{"hint": "h0", "start": 11,
"end": 11, "span": "raft", "candidates": ["rift"], "reason": "sounds like
\\"rift\\" used elsewhere"}]
Example reply: {"verdicts": [
 [11, 11, "raft", null, "misheard", 0.9, "h0", "Raft is a consensus algorithm"],
 [4, 4, "cubernetes", "Kubernetes", "misheard", 0.97, null, "container platform"],
 [7, 8, "con sensus", "consensus", "misheard", 0.95, null, "one word split in two"]]}
"""


def build_messages(request: ChunkRequest) -> list[dict]:
    terms = "; ".join(request.priming_terms) or "(none)"
    numbered = " ".join(f"[{gi}]{text}" for gi, text in request.words)
    hints = json.dumps([h.as_dict() for h in request.hints], ensure_ascii=False)
    user = "\n\n".join(
        [
            f"Priming terms: {terms}",
            f"Before (context only): {request.before or '(start of transcript)'}",
            f"Chunk:\n{numbered}",
            f"After (context only): {request.after or '(end of transcript)'}",
            f"Hints: {hints}",
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --- reply parsing -----------------------------------------------------------


def parse_reply(content: str, request: ChunkRequest) -> list[ChunkVerdict]:
    """Turn a reply into Verdicts on the request's Words.

    A verdict whose range disagrees with its ``span`` text is moved to the
    nearest place that text occurs; one that can't be placed at all is
    dropped (a hint's falls back to the hint's own range). A replacement is
    no correction when its ``cause`` is anything but ``misheard`` (the
    speaker's own grammar or style), or when it only changes case or
    punctuation -- and, for a new find, spacing; a change to a hint's word
    boundaries counts. A hint's verdict with no correction becomes
    not-an-error; any other is dropped. Raises
    :class:`CorrectorError` -- a failed chunk -- on anything that isn't a
    JSON list of verdicts, or when a hint has no verdict."""
    text = content.strip()
    if text.startswith("```"):  # ```json ... ```
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise CorrectorError(f"reply is not JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("verdicts")
    if not isinstance(raw, list):
        raise CorrectorError("reply has no list of verdicts")

    positions = [gi for gi, _ in request.words]
    tokens = [clean(t) for _, t in request.words]
    hints = {h.id: h for h in request.hints}
    out: list[ChunkVerdict] = []
    answered: set[str] = set()
    for item in raw:
        if isinstance(item, list):  # the compact positional form
            item = dict(zip(_FIELDS, item))
        if not isinstance(item, dict):
            continue
        hint_id = item.get("hint")
        hint = hints.get(str(hint_id)) if hint_id is not None else None
        if hint is not None and hint.id in answered:
            continue
        placed = _place(item, positions, tokens)
        if placed is None and hint is not None:
            placed = (hint.start, hint.end)
        if placed is None:
            continue
        replacement = item.get("replacement")
        span_text = " ".join(
            t for gi, t in request.words if placed[0] <= gi <= placed[1]
        )
        # A hint's span is already suspect, so moving its word boundaries
        # ("con sensus" -> "consensus") is a Correction; a new find must
        # change what a listener would hear, since the model proposes many
        # boundary-only changes to words that were right.
        normalise = _words if hint is not None else _letters
        if replacement is not None and (
            item.get("cause", "misheard") != "misheard"
            or normalise(str(replacement)) == normalise(span_text)
        ):
            if hint is None:
                continue
            replacement = None
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        out.append(
            ChunkVerdict(
                start=placed[0],
                end=placed[1],
                replacement=None if replacement is None else str(replacement),
                confidence=max(0.0, min(1.0, confidence)),
                rationale=str(item.get("rationale", "")),
                hint=hint.id if hint is not None else None,
            )
        )
        if hint is not None:
            answered.add(hint.id)

    missing = sorted(set(hints) - answered)
    if missing:
        raise CorrectorError(f"reply has no verdict for hint(s) {missing}")
    return out


#: Field order of a compact, positional verdict.
_FIELDS = (
    "start", "end", "span", "replacement", "cause", "confidence", "hint", "rationale",
)


def _letters(text: str) -> str:
    """``text`` with case, punctuation and spacing gone -- what's left is
    what a listener would hear."""
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _words(text: str) -> tuple[str, ...]:
    """``text`` with case and punctuation gone but its word breaks kept --
    what a reader would see as its words ("Anthropic's" is two)."""
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


def _place(
    item: dict, positions: list[int], tokens: list[str]
) -> tuple[int, int] | None:
    """The ``(start, end)`` global indices a verdict covers, checked against
    its ``span`` text when it has one."""
    start, end = item.get("start"), item.get("end")
    span = [clean(t) for t in str(item.get("span") or "").split()]
    target = 0
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and start in positions
        and end in positions
        and start <= end
    ):
        target = lo = positions.index(start)
        hi = positions.index(end) + 1
        if not span or tokens[lo:hi] == span:
            return start, end
    if not span:
        return None
    n = len(span)
    hits = [i for i in range(len(tokens) - n + 1) if tokens[i : i + n] == span]
    if not hits:
        return None
    best = min(hits, key=lambda i: abs(i - target))
    return positions[best], positions[best + n - 1]


# --- orchestrator ------------------------------------------------------------


def read_through(
    cues: list[Cue],
    flags: list[Flag],
    reader: Reader,
    *,
    priming_terms: Sequence[str] = (),
    chunk_words: int = CHUNK_WORDS,
    max_calls: int | None = None,
    workers: int = WORKERS,
    min_confidence: float = MIN_CONFIDENCE,
) -> ReadThroughResult:
    """Read ``cues`` chunk by chunk with the local ``flags`` as hints. Each
    chunk is retried once on a malformed reply; a chunk that fails twice
    keeps its hints unjudged. ``max_calls`` caps requests, retries included.
    New finds under ``min_confidence`` are discarded; every hint keeps its
    verdict whatever the confidence."""
    chunks = plan_chunks(
        cues, flags, priming_terms=priming_terms, chunk_words=chunk_words
    )
    words = tokenize(cues)
    by_gi = {w.global_index: w for w in words}
    hint_flags = {min(f.global_indices): f for f in flags}
    calls = 0
    lock = threading.Lock()

    def ask(chunk: ChunkRequest) -> list[ChunkVerdict] | None:
        nonlocal calls
        for _attempt in (1, 2):
            with lock:
                if max_calls is not None and calls >= max_calls:
                    return None
                calls += 1
            try:
                return reader.read(chunk)
            except CorrectorError:
                continue
        return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        replies = list(pool.map(ask, chunks))

    context_of = {gi: text for text, idxs in sentences(cues, words) for gi in idxs}
    cues_by_index = index_cues(cues)
    items: list[ReadItem] = []
    failed = 0
    for n, (chunk, verdicts) in enumerate(zip(chunks, replies)):
        if verdicts is None:
            failed += 1
            items.extend(
                ReadItem(hint_flags[h.start], None, hint_flags[h.start])
                for h in chunk.hints
            )
            continue
        answered: list[ReadItem] = []  # verdicts on hints
        found: list[ReadItem] = []
        for k, v in enumerate(verdicts):
            if v.hint is None and v.confidence < min_confidence:
                continue
            hint = next((h for h in chunk.hints if h.id == v.hint), None)
            span_words = [by_gi[gi] for gi in range(v.start, v.end + 1)]
            flag = _flag_for(v, hint, span_words, hint_flags, cues_by_index)
            flag.context = flag.context or _context(span_words, context_of)
            correction = Correction(
                id=f"c{n}.{k}",
                replacement=v.replacement,
                confidence=v.confidence,
                rationale=v.rationale,
            )
            if hint is None:
                found.append(ReadItem(flag, correction))
            else:
                answered.append(
                    ReadItem(flag, correction, hint_flags[hint.start])
                )
        items.extend(_without_overlaps(answered, found))

    items.sort(key=lambda i: min(i.flag.global_indices))
    return ReadThroughResult(
        items=items,
        calls=calls,
        chunk_count=len(chunks),
        failed_chunks=failed,
    )


def _context(span_words: list[Word], context_of: dict[int, str]) -> str:
    """The sentence a span sits in -- or, for one that runs across a sentence
    end (as a span across a Cue boundary may), every sentence it touches."""
    parts: list[str] = []
    for w in span_words:
        text = context_of.get(w.global_index, "")
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def _flag_for(
    v: ChunkVerdict,
    hint: Hint | None,
    span_words: list[Word],
    hint_flags: dict[int, Flag],
    cues_by_index: dict[int, Cue],
) -> Flag:
    """The hint's own Flag when the verdict keeps its range; otherwise a new
    Flag, credited to the hint's detector too when it widened a hint."""
    if hint is not None and (v.start, v.end) == (hint.start, hint.end):
        return hint_flags[hint.start]
    if hint is not None:
        base = hint_flags[hint.start]
        detector = f"{base.detector}+{DETECTOR_READ_THROUGH}"
        reason = f"{base.reason}; widened by the Read-through: {v.rationale}"
    else:
        detector = DETECTOR_READ_THROUGH
        reason = v.rationale or "the Read-through judged this a misrecognition"
    return make_flag(
        span_words,
        cues_by_index,
        detector=detector,
        reason=reason,
        confidence=v.confidence,
        candidates=[v.replacement] if v.replacement is not None else [],
    )


def _without_overlaps(
    answered: list[ReadItem], found: list[ReadItem]
) -> list[ReadItem]:
    """One verdict per Word, since a Word can only be spliced once. Every
    hint keeps its verdict -- on its own span if a widened one would collide
    with another verdict or another hint's own span -- and the Read-through's
    own finds fill the gaps, most confident first."""
    kept: list[ReadItem] = []
    taken: set[int] = set()
    hint_words = [set(i.hint.global_indices) for i in answered if i.hint is not None]
    for item in answered:
        if item.hint is not None and item.flag is not item.hint:
            own = set(item.hint.global_indices)
            others = set().union(*(w for w in hint_words if w != own))
            if (taken | others).intersection(item.flag.global_indices):
                item = replace(item, flag=item.hint)
        taken.update(item.flag.global_indices)
        kept.append(item)
    for item in sorted(found, key=lambda i: -(i.correction.confidence if i.correction else 0.0)):
        if taken.intersection(item.flag.global_indices):
            continue
        taken.update(item.flag.global_indices)
        kept.append(item)
    return kept


# --- readers -------------------------------------------------------------------


class StubReader:
    """Deterministic Reader for tests and offline runs.

    By default it answers every hint with its top candidate (or not-an-error
    when it has none) and finds nothing else. The keyword arguments script
    the rest:

    - ``extra`` -- ``{span text: replacement}`` errors to report wherever
      that text occurs in a chunk.
    - ``null_spans`` -- hints to answer not-an-error.
    - ``replacement_for`` / ``confidence_for`` -- per-span overrides.
    - ``widen`` -- ``{hint span: wider span text}`` to answer a hint with.
    - ``garbage`` -- raise :class:`CorrectorError` on every call.
    """

    def __init__(
        self,
        *,
        extra: dict[str, str] | None = None,
        null_spans: set[str] | None = None,
        replacement_for: dict[str, str] | None = None,
        confidence_for: dict[str, float] | None = None,
        widen: dict[str, str] | None = None,
        garbage: bool = False,
        default_confidence: float = 0.9,
    ) -> None:
        self.extra = extra or {}
        self.null_spans = null_spans or set()
        self.replacement_for = replacement_for or {}
        self.confidence_for = confidence_for or {}
        self.widen = widen or {}
        self.garbage = garbage
        self.default_confidence = default_confidence
        self.calls = 0
        self.requests: list[ChunkRequest] = []
        self.spend = Spend()  # a stub costs nothing
        self._lock = threading.Lock()

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
        with self._lock:
            self.calls += 1
            self.requests.append(request)
        if self.garbage:
            raise CorrectorError("stub: garbage reply")
        out: list[ChunkVerdict] = []
        for h in request.hints:
            span = self.widen.get(h.span, h.span)
            where = _find(span, request) if span != h.span else (h.start, h.end)
            if where is None:
                where = (h.start, h.end)
            replacement: str | None = self.replacement_for.get(
                span, h.candidates[0] if h.candidates else None
            )
            if h.span in self.null_spans:
                replacement = None
            out.append(self._verdict(span, where, replacement, h.id))
        for span, replacement in self.extra.items():
            where = _find(span, request)
            if where is not None:
                out.append(self._verdict(span, where, replacement, None))
        return out

    def _verdict(
        self, span: str, where: tuple[int, int], replacement: str | None, hint: str | None
    ) -> ChunkVerdict:
        return ChunkVerdict(
            start=where[0],
            end=where[1],
            replacement=replacement,
            confidence=self.confidence_for.get(span, self.default_confidence),
            rationale="stub",
            hint=hint,
        )


def _find(span: str, request: ChunkRequest) -> tuple[int, int] | None:
    return _place(
        {"span": span},
        [gi for gi, _ in request.words],
        [clean(t) for _, t in request.words],
    )


class OpenRouterReader:
    """The real Reader: one OpenRouter chat call per chunk."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.client = OpenRouterClient(model_id, api_key=api_key)

    @property
    def spend(self) -> Spend:
        return self.client.spend

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
        content = self.client.chat(
            build_messages(request),
            timeout=180,
            response_format={"type": "json_object"},
        )
        return parse_reply(content, request)


def build_reader(model_id: str) -> OpenRouterReader:
    """Factory the CLI and eval call; tests monkeypatch this to inject a stub."""
    return OpenRouterReader(model_id)
