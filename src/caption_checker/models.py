from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


def format_timestamp(td: timedelta) -> str:
    """``HH:MM:SS,mmm`` -- the SRT-style stamp used in reports and the eval
    table (kept format-neutral here so both ``check`` and ``correct`` share it)."""
    total_ms = round(td.total_seconds() * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Cue:
    index: int
    start: timedelta
    end: timedelta
    text: str


@dataclass
class Word:
    text: str
    cue_index: int
    char_offset: int
    global_index: int


@dataclass
class Flag:
    """A span of transcript text a detector believes is a likely ASR error."""

    span: str
    global_indices: list[int]
    cue_index: int
    start: timedelta
    end: timedelta
    detector: str
    reason: str
    candidates: list[str] = field(default_factory=list)
    confidence: float = 0.0
    context: str = ""


def flag_to_dict(flag: Flag) -> dict:
    """JSON shape of a Flag -- the ``check --format json`` contract and the
    per-flag record inside a ``correct`` sidecar."""
    return {
        "span": flag.span,
        "global_indices": flag.global_indices,
        "cue_index": flag.cue_index,
        "start": flag.start.total_seconds(),
        "end": flag.end.total_seconds(),
        "detector": flag.detector,
        "reason": flag.reason,
        "candidates": flag.candidates,
        "confidence": flag.confidence,
        "context": flag.context,
    }


# Default OpenRouter model for the ``correct`` pass: a Gemini Flash-class slug,
# pinned here and overridable with ``correct --model SLUG``.
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


# Detector names, also the order flags are reported in.
DETECTOR_OOV = "oov"
DETECTOR_PHONETIC_VOCAB = "phonetic_vocab"
DETECTOR_PHONETIC_INTERNAL = "phonetic_internal"
DETECTOR_SPLIT_WORD = "split_word"
DETECTOR_CONTEXT_EMBEDDING = "context_embedding"


_DEFAULT_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with without from into
    over under again further is are was were be been being do does did have has had
    i you he she it we they me him her us them my your his its our their this that
    these those as so than too very can will just not no nor only own same s t
    we're we'll today next week
    """.split()
)


@dataclass
class DetectConfig:
    """Tunable thresholds for the detection pass. Defaults aim for the plan's
    ~5-10% pre-filter rate before the (future) LLM tier."""

    oov_zipf_max: float = 0.0
    min_token_len: int = 3
    split_max_window: int = 3
    split_component_max_len: int = 5
    split_common_zipf_min: float = 3.0
    phonetic_algo: str = "metaphone"
    known_good_zipf_min: float = 3.0
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_sim_z: float = -1.5
    embedding_candidate_zipf_max: float = 2.5
    enable_embeddings: bool = True
    #: Internal-match bypass (see ``correct.py``): a pure ``phonetic_internal``
    #: flag with several candidates is only applied without an LLM call when the
    #: top candidate's Jaro-Winkler score beats the runner-up's by at least this
    #: much. A lone candidate always clears the bar.
    bypass_jw_margin: float = 0.15
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS
