"""The Example (#65): a Transcript run through Correct once, offline, and
saved with the app, so a first-time visitor can see a finished review
without uploading anything.

Saved as a directory shaped like a stored Transcript's:

    <example_dir>/original.<ext>
    <example_dir>/state.json

:func:`capture` writes it from a Transcript in a server's data directory;
:func:`copy_into` gives a visitor their own copy in their Session, so one
visitor's Review Decisions never show up for another. No model is called
either way, so opening the Example is never metered by the Free tier.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from caption_checker.models import DETECTOR_REVIEWER
from caption_checker.normalize import span_contexts
from caption_checker.parser import tokenize
from caption_checker.web.models import (
    ReviewDecision,
    TranscriptRecord,
    record_from_dict,
    record_to_dict,
)
from caption_checker.web.storage import Storage

#: Where the app looks for the Example it offers.
EXAMPLE_DIR = Path(__file__).parent / "example"

STATE_FILENAME = "state.json"


class ExampleError(ValueError):
    """Raised when a Transcript can't be saved as the Example."""


def _original(example_dir: Path) -> Path | None:
    matches = sorted(example_dir.glob("original.*"))
    return matches[0] if matches else None


def is_available(example_dir: Path | None) -> bool:
    """Whether an Example is saved in ``example_dir``."""
    return (
        example_dir is not None
        and (example_dir / STATE_FILENAME).is_file()
        and _original(example_dir) is not None
    )


def load(example_dir: Path) -> TranscriptRecord:
    """The Example saved in ``example_dir``, belonging to no Session."""
    data = json.loads((example_dir / STATE_FILENAME).read_text(encoding="utf-8"))
    return record_from_dict(data)


def capture(
    storage: Storage, transcript_id: str, example_dir: Path, *, credit: str = ""
) -> TranscriptRecord:
    """Save Transcript ``transcript_id`` from ``storage`` as the Example in
    ``example_dir``, replacing any saved before, and return what was saved.

    The run is kept -- Flags, Corrections, chunk counts, Video link -- but
    not the review: every Review Decision goes back to pending, and the
    Flags the reviewer raised by editing Cues are dropped, so each visitor
    starts the review fresh. Each Flag's context is cut again from the
    original, since a Flag made before a change to how contexts are cut
    still holds the old one. ``credit`` is shown on every copy, for a
    Source video whose licence asks for one. The record belongs to no
    Session until copied.
    """
    record = storage.find_transcript(transcript_id)
    original = storage.original_path(record.session_id, record.id) if record else None
    if record is None or original is None:
        raise ExampleError(f"No transcript {transcript_id!r} in {storage.root}")
    if not record.corrected:
        raise ExampleError(
            f"Correct hasn't been run on transcript {transcript_id!r}; "
            "the Example shows a finished run."
        )

    # Records from before the Read-through can hold fewer Corrections than
    # Flags; pad them so the two stay aligned by index.
    corrections = list(record.corrections) + [None] * (len(record.flags) - len(record.corrections))
    kept = [i for i, flag in enumerate(record.flags) if flag.detector != DETECTOR_REVIEWER]
    cues = storage.load_cues(record.session_id, record.id)
    context = span_contexts(cues, tokenize(cues))
    saved = replace(
        record,
        session_id="",
        flags=[
            replace(record.flags[i], context=context(record.flags[i].global_indices))
            for i in kept
        ],
        corrections=[corrections[i] for i in kept],
        decisions=[ReviewDecision() for _ in kept],
        correct_error=None,
        last_activity="",
        is_example=False,
        credit=credit.strip(),
    )

    example_dir.mkdir(parents=True, exist_ok=True)
    for old in example_dir.glob("original.*"):
        old.unlink()
    shutil.copyfile(original, example_dir / original.name)
    (example_dir / STATE_FILENAME).write_text(
        json.dumps(record_to_dict(saved), indent=2), encoding="utf-8"
    )
    return load(example_dir)


def copy_into(storage: Storage, session_id: str, example_dir: Path) -> TranscriptRecord:
    """Copy the Example in ``example_dir`` into Session ``session_id`` as a
    new Transcript, marked as an Example and created now; the retention
    sweep then treats it like any other."""
    original = _original(example_dir)
    if original is None:
        raise FileNotFoundError(f"No Example saved in {example_dir}")
    saved = load(example_dir)
    record = replace(
        saved,
        id=uuid4().hex,
        session_id=session_id,
        created_at=storage.clock().isoformat(),
        is_example=True,
    )
    suffix = original.suffix.lstrip(".")
    shutil.copyfile(original, storage.staging_path(session_id, record.id, suffix))
    storage.save_transcript(record)
    return record
