"""The LLM correction backend, behind a small protocol.

``Corrector.correct`` takes a batch of :class:`FlagContext` and returns a
:class:`Correction` per flag. Two implementations ship:

- :class:`OpenRouterCorrector` -- the real one. Reads ``OPENROUTER_API_KEY``
  from ``.env`` (via ``python-dotenv``) or the environment, talks to the
  OpenRouter chat-completions endpoint over the standard library, and parses
  the JSON reply with :mod:`caption_checker.prompt`.
- :class:`StubCorrector` -- deterministic, offline, for tests.

The detection pipeline never imports this module, and importing this module
never imports an HTTP client: :class:`OpenRouterCorrector` pulls ``urllib`` in
lazily when it actually calls out.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from caption_checker.models import DEFAULT_MODEL

__all__ = [
    "DEFAULT_MODEL",
    "Correction",
    "Corrector",
    "CorrectorError",
    "FlagContext",
    "MissingAPIKeyError",
    "OpenRouterCorrector",
    "StubCorrector",
    "build_corrector",
    "ensure_ids_match",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class FlagContext:
    """Everything one flag carries into the LLM request."""

    id: str
    span: str
    sentence: str
    candidates: list[str]
    detector: str
    reason: str
    nearby: tuple[str, str]  # cue before, cue after
    related: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "span": self.span,
            "sentence": self.sentence,
            "candidates": self.candidates,
            "detector": self.detector,
            "reason": self.reason,
            "nearby_before": self.nearby[0],
            "nearby_after": self.nearby[1],
            "related": self.related,
        }


@dataclass
class Correction:
    """The model's verdict on one flag. ``replacement is None`` is a
    not-an-error verdict."""

    id: str
    replacement: str | None
    confidence: float
    rationale: str = ""


class CorrectorError(RuntimeError):
    """The model reply could not be turned into corrections for this batch."""


def ensure_ids_match(got: set[str], want: set[str]) -> None:
    """Raise :class:`CorrectorError` unless the reply covers exactly the
    requested flag ids -- the check both the reply parser and the orchestrator
    apply before trusting a batch."""
    if got != want:
        raise CorrectorError(
            f"reply ids {sorted(got)} do not match request {sorted(want)}"
        )


class MissingAPIKeyError(RuntimeError):
    """No OpenRouter credential is configured."""


class Corrector(Protocol):
    def correct(self, batch: list[FlagContext]) -> list[Correction]: ...


class StubCorrector:
    """Deterministic corrector for tests and offline runs.

    By default it answers each flag with its top detector candidate at a fixed
    confidence, or a not-an-error verdict when the flag has no candidate. The
    keyword arguments script the exceptions a test needs:

    - ``null_spans`` -- return a not-an-error verdict for these spans.
    - ``garbage_spans`` -- raise :class:`CorrectorError` for any batch touching
      one of these spans (exercises the retry-then-skip path).
    - ``confidence_for`` -- ``{span: confidence}`` overrides.
    - ``replacement_for`` -- ``{span: replacement}`` overrides.
    """

    def __init__(
        self,
        *,
        default_confidence: float = 0.9,
        null_spans: set[str] | None = None,
        garbage_spans: set[str] | None = None,
        confidence_for: dict[str, float] | None = None,
        replacement_for: dict[str, str] | None = None,
    ) -> None:
        self.default_confidence = default_confidence
        self.null_spans = null_spans or set()
        self.garbage_spans = garbage_spans or set()
        self.confidence_for = confidence_for or {}
        self.replacement_for = replacement_for or {}
        self.calls: list[list[FlagContext]] = []

    def correct(self, batch: list[FlagContext]) -> list[Correction]:
        self.calls.append(list(batch))
        if any(fc.span in self.garbage_spans for fc in batch):
            raise CorrectorError("stub: garbage batch")
        out: list[Correction] = []
        for fc in batch:
            if fc.span in self.null_spans:
                out.append(Correction(fc.id, None, 0.2, "stub: not an error"))
                continue
            replacement = self.replacement_for.get(fc.span)
            if replacement is None:
                replacement = fc.candidates[0] if fc.candidates else None
            confidence = self.confidence_for.get(
                fc.span, self.default_confidence
            )
            verdict = "stub: no candidate" if replacement is None else "stub"
            out.append(Correction(fc.id, replacement, confidence, verdict))
        return out


class OpenRouterCorrector:
    """Real corrector, backed by OpenRouter's OpenAI-compatible endpoint."""

    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, api_key: str | None = None
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key or _load_api_key()
        if not self._api_key:
            raise MissingAPIKeyError(
                "no OpenRouter credential: set OPENROUTER_API_KEY in the "
                "environment or in a .env file"
            )

    def correct(self, batch: list[FlagContext]) -> list[Correction]:
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        from caption_checker.prompt import build_messages, parse_response

        body = json.dumps(
            {
                "model": self.model_id,
                "messages": build_messages(batch),
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise CorrectorError(f"OpenRouter request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CorrectorError(f"unexpected OpenRouter reply: {exc}") from exc
        return parse_response(content, batch)


def build_corrector(model_id: str) -> Corrector:
    """Factory the CLI calls; tests monkeypatch this to inject a stub."""
    return OpenRouterCorrector(model_id)


def _load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - python-dotenv is a hard dep
        pass
    return os.environ.get("OPENROUTER_API_KEY") or None
