from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


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
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS
