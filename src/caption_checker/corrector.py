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
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from caption_checker.models import DEFAULT_MODEL

if TYPE_CHECKING:
    import ssl

__all__ = [
    "DEFAULT_MODEL",
    "Correction",
    "Corrector",
    "CorrectorError",
    "FlagContext",
    "MissingAPIKeyError",
    "OpenRouterClient",
    "OpenRouterCorrector",
    "Spend",
    "StubCorrector",
    "build_corrector",
    "ensure_ids_match",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


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


@dataclass
class Spend:
    """What a run's LLM requests cost, as OpenRouter reported it. ``cost_usd``
    stays None if any reply came back without a cost figure."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0

    def add(self, usage: dict) -> None:
        self.requests += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        if self.cost_usd is not None and isinstance(cost, (int, float)):
            self.cost_usd += float(cost)
        else:
            self.cost_usd = None


class OpenRouterClient:
    """One model on OpenRouter's OpenAI-compatible chat endpoint: the
    transport both the per-flag :class:`OpenRouterCorrector` and the
    Read-through share. Tallies every reply's usage into ``spend``."""

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
        # Only checked once per instance, on the first `chat()` call --
        # callers reuse one instance across every chunk of a run.
        self._model_checked = False
        self._lock = threading.Lock()
        self.spend = Spend()

    def chat(
        self, messages: list[dict], *, timeout: float = 60, **options: object
    ) -> str:
        """Send one chat turn and return the reply text. ``options`` are
        extra request fields (e.g. ``response_format``)."""
        import ssl
        from http.client import HTTPException
        from urllib.error import HTTPError, URLError
        from urllib.request import Request, urlopen

        import certifi

        # Some Python installs (notably python.org's macOS builds) ship
        # without a wired-up system trust store, so the stdlib's default
        # SSL context can't verify OpenRouter's certificate. Point it at
        # certifi's bundle explicitly rather than relying on the
        # environment being set up right.
        context = ssl.create_default_context(cafile=certifi.where())
        with self._lock:
            if not self._model_checked:
                self._check_model_available(context)
                self._model_checked = True

        body = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "temperature": 0,
                "usage": {"include": True},
                **options,
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
            with urlopen(request, timeout=timeout, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # exc's own str() is just the generic reason phrase (e.g. "Not
            # Found") -- OpenRouter's actual explanation (bad model slug,
            # no credit, moderation, ...) is in the JSON body.
            detail = exc.read().decode("utf-8", errors="replace").strip()
            try:
                detail = json.loads(detail)["error"]["message"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            raise CorrectorError(
                f"OpenRouter request failed: HTTP {exc.code} {exc.reason}: {detail}"
            ) from exc
        except (URLError, HTTPException, OSError, ValueError) as exc:
            # HTTPException: a reply cut off mid-body (IncompleteRead);
            # OSError covers TimeoutError and dropped connections.
            raise CorrectorError(f"OpenRouter request failed: {exc!r}") from exc

        with self._lock:
            self.spend.add(payload.get("usage") or {})
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CorrectorError(f"unexpected OpenRouter reply: {exc}") from exc
        if not isinstance(content, str):
            # e.g. a reasoning model that spent its whole budget thinking
            raise CorrectorError("OpenRouter reply has no text")
        return content

    def _check_model_available(self, context: ssl.SSLContext) -> None:
        """Catalogue drift guard: OpenRouter periodically retires dated model
        slugs (e.g. ``google/gemini-2.0-flash-001`` disappeared in 2026-09),
        which otherwise only surfaces as a bare 404 from the completions
        endpoint. Check the public model list first so a retired slug fails
        fast with a pointer to what replaced it, instead of a cryptic error
        from mid-batch. Best-effort: if the catalogue check itself can't be
        reached, fall through and let the real request's own error handling
        take over rather than blocking the whole run on it.
        """
        from urllib.error import URLError
        from urllib.request import urlopen

        try:
            with urlopen(OPENROUTER_MODELS_URL, timeout=15, context=context) as response:
                payload = json.loads(response.read().decode("utf-8"))
            available = {m["id"] for m in payload["data"]}
        except (URLError, TimeoutError, ValueError, KeyError, TypeError):
            return

        if self.model_id in available:
            return

        suggestions = _similar_models(self.model_id, available)
        hint = (
            f" Similar models still available: {', '.join(suggestions)}."
            if suggestions
            else ""
        )
        raise CorrectorError(
            f"model {self.model_id!r} is no longer on OpenRouter.{hint} "
            "Set --model (CLI) or OPENROUTER_MODEL (web) to a current slug."
        )


class OpenRouterCorrector:
    """Real corrector, backed by OpenRouter's OpenAI-compatible endpoint."""

    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, api_key: str | None = None
    ) -> None:
        self.model_id = model_id
        self.client = OpenRouterClient(model_id, api_key=api_key)

    def correct(self, batch: list[FlagContext]) -> list[Correction]:
        from caption_checker.prompt import build_messages, parse_response

        return parse_response(self.client.chat(build_messages(batch)), batch)


def _similar_models(model_id: str, available: set[str]) -> list[str]:
    """Same-vendor models, ranked by how many of ``model_id``'s dash/dot/colon
    -separated words they share (e.g. ``flash``, ``lite``) -- a cheap stand-in
    for "closest replacement" that needs no extra API call."""
    vendor = model_id.split("/", 1)[0]
    keywords = [w for w in re.split(r"[/:._-]", model_id) if w and w != vendor]

    def shared_keywords(candidate: str) -> int:
        return sum(1 for w in keywords if w in candidate)

    same_vendor = [
        m for m in available if m.startswith(f"{vendor}/") and not m.startswith("~")
    ]
    same_vendor.sort(key=lambda m: (-shared_keywords(m), m))
    return same_vendor[:3]


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
