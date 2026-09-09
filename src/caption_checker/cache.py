"""Decision cache: a persistent JSON store mapping ``(cleaned span, model id)``
to the correction that model returned, so a mistranscription that recurs -- in
the same file or across a lecture series -- is not paid for twice.

Keyed on the span alone, not its surrounding context (CONTEXT.md): the working
assumption is that a given garble resolves the same way every time. ``--no-cache``
turns both read and write off; the mitigation for a genuinely ambiguous garble
is that cached corrections still go through review.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from caption_checker.normalize import clean


def default_cache_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "caption-checker" / "corrections.json"


@dataclass
class CachedCorrection:
    replacement: str | None
    confidence: float
    rationale: str = ""


class DecisionCache:
    """JSON file of ``{cleaned_span: {model_id: correction}}``.

    Construct with :meth:`load`. A disabled cache (``--no-cache``) answers every
    lookup with ``None`` and drops every write, so callers need no branching.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        enabled: bool = True,
        entries: dict[str, dict[str, dict]] | None = None,
    ) -> None:
        self.path = path
        self.enabled = enabled
        self._entries: dict[str, dict[str, dict]] = entries or {}
        self._dirty = False

    @classmethod
    def load(cls, path: Path | None, *, enabled: bool = True) -> DecisionCache:
        if not enabled or path is None:
            return cls(path, enabled=enabled)
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, dict) else {}
        except (FileNotFoundError, ValueError):
            entries = {}
        return cls(Path(path), enabled=True, entries=entries)

    def get(self, span: str, model_id: str) -> CachedCorrection | None:
        if not self.enabled:
            return None
        record = self._entries.get(clean(span), {}).get(model_id)
        if record is None:
            return None
        return CachedCorrection(
            replacement=record.get("replacement"),
            confidence=float(record.get("confidence", 0.0)),
            rationale=record.get("rationale", ""),
        )

    def set(
        self, span: str, model_id: str, correction: CachedCorrection
    ) -> None:
        if not self.enabled:
            return
        self._entries.setdefault(clean(span), {})[model_id] = {
            "replacement": correction.replacement,
            "confidence": correction.confidence,
            "rationale": correction.rationale,
        }
        self._dirty = True

    def save(self) -> None:
        if not self.enabled or self.path is None or not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._dirty = False
