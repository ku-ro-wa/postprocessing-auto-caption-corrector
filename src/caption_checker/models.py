from __future__ import annotations

from dataclasses import dataclass
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
