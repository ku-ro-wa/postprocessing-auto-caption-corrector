from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import srt
import webvtt

from caption_checker.models import Cue, Word

_WORD_RE = re.compile(r"\S+")


def parse(path: str | Path) -> list[Cue]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return _parse_srt(path)
    if suffix == ".vtt":
        return _parse_vtt(path)
    raise ValueError(f"Unsupported caption format: {suffix}")


def serialize(cues: list[Cue], format: str) -> str:
    if format == "srt":
        return _serialize_srt(cues)
    if format == "vtt":
        return _serialize_vtt(cues)
    raise ValueError(f"Unsupported caption format: {format}")


def tokenize(cues: list[Cue]) -> list[Word]:
    words = []
    global_index = 0
    for cue in cues:
        for match in _WORD_RE.finditer(cue.text):
            words.append(
                Word(
                    text=match.group(),
                    cue_index=cue.index,
                    char_offset=match.start(),
                    global_index=global_index,
                )
            )
            global_index += 1
    return words


def _parse_srt(path: Path) -> list[Cue]:
    subs = list(srt.parse(path.read_text(encoding="utf-8")))
    return [
        Cue(index=sub.index, start=sub.start, end=sub.end, text=sub.content)
        for sub in subs
    ]


def _parse_vtt(path: Path) -> list[Cue]:
    vtt = webvtt.WebVTT.read(str(path))
    return [
        Cue(
            index=i,
            start=_vtt_timestamp_to_timedelta(caption.start),
            end=_vtt_timestamp_to_timedelta(caption.end),
            text=caption.text,
        )
        for i, caption in enumerate(vtt.captions, start=1)
    ]


def _serialize_srt(cues: list[Cue]) -> str:
    subs = [
        srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=cue.text)
        for cue in cues
    ]
    return srt.compose(subs)


def _serialize_vtt(cues: list[Cue]) -> str:
    vtt = webvtt.WebVTT(
        captions=[
            webvtt.Caption(
                start=_timedelta_to_vtt_timestamp(cue.start),
                end=_timedelta_to_vtt_timestamp(cue.end),
                text=cue.text,
            )
            for cue in cues
        ]
    )
    return vtt.content


def _timedelta_to_vtt_timestamp(td: timedelta) -> str:
    total_ms = round(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _vtt_timestamp_to_timedelta(ts: str) -> timedelta:
    hours, minutes, rest = ts.split(":")
    seconds, _, ms = rest.partition(".")
    return timedelta(
        hours=int(hours),
        minutes=int(minutes),
        seconds=int(seconds),
        milliseconds=int(ms or 0),
    )
