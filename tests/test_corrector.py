"""The Corrector protocol layer: the stub, the real backend's guard rails, and
the reply parser. No network anywhere."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from caption_checker.corrector import (
    OPENROUTER_MODELS_URL,
    OPENROUTER_URL,
    Correction,
    CorrectorError,
    FlagContext,
    MissingAPIKeyError,
    OpenRouterCorrector,
    StubCorrector,
    _similar_models,
)
from caption_checker.prompt import parse_response


def _ctx(cid: str, span: str, candidates: list[str]) -> FlagContext:
    return FlagContext(
        id=cid,
        span=span,
        sentence=f"a sentence with {span} in it",
        candidates=candidates,
        detector="oov",
        reason="reason",
        nearby=("", ""),
        related=[],
    )


def test_stub_is_deterministic_and_counts_calls() -> None:
    stub = StubCorrector()
    batch = [_ctx("f0", "sensus", ["consensus"]), _ctx("f1", "rey", [])]

    first = stub.correct(batch)
    second = stub.correct(batch)

    assert first == second
    assert [c.replacement for c in first] == ["consensus", None]
    assert len(stub.calls) == 2


def test_stub_null_and_garbage_scripting() -> None:
    stub = StubCorrector(null_spans={"lease"}, garbage_spans={"boom"})
    assert stub.correct([_ctx("f0", "lease", ["least"])])[0].replacement is None
    with pytest.raises(CorrectorError):
        stub.correct([_ctx("f0", "boom", [])])


def test_openrouter_without_key_raises_named_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "caption_checker.corrector._load_api_key", lambda: None
    )
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        OpenRouterCorrector("some/model")


class _FakeResponse:
    """Minimal stand-in for the context manager `urlopen` returns."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _patch_urlopen(monkeypatch, *, models: list[str], completion_reply: str | None) -> list[str]:
    """Fakes both OpenRouter endpoints the corrector talks to, keyed by URL.
    Returns the list of URLs `urlopen` was called with, so tests can assert
    on call count (e.g. the model check running only once per instance)."""
    calls: list[str] = []
    models_body = json.dumps({"data": [{"id": m} for m in models]}).encode()
    completion_body = (
        json.dumps(
            {"choices": [{"message": {"content": completion_reply}}]}
        ).encode()
        if completion_reply is not None
        else b""
    )

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ARG001
        url = request if isinstance(request, str) else request.full_url
        calls.append(url)
        if url == OPENROUTER_MODELS_URL:
            return _FakeResponse(models_body)
        assert url == OPENROUTER_URL
        return _FakeResponse(completion_body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def test_similar_models_ranks_same_vendor_by_shared_keywords() -> None:
    available = {
        "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "google/gemini-3.1-flash-lite",
        "openai/gpt-4o-mini",
        "~google/gemini-flash-latest",
    }
    got = _similar_models("google/gemini-2.0-flash-001", available)
    assert got[0] in {"google/gemini-2.5-flash", "google/gemini-2.5-flash-lite"}
    assert all(m.startswith("google/") for m in got)
    assert "~google/gemini-flash-latest" not in got


def test_correct_fails_fast_on_retired_model(monkeypatch) -> None:
    calls = _patch_urlopen(
        monkeypatch, models=["google/gemini-2.5-flash"], completion_reply=None
    )
    corrector = OpenRouterCorrector("google/gemini-2.0-flash-001", api_key="k")

    with pytest.raises(CorrectorError, match="no longer on OpenRouter"):
        corrector.correct([_ctx("f0", "sensus", ["consensus"])])

    assert calls == [OPENROUTER_MODELS_URL]  # never reached the completions call


def test_correct_proceeds_when_model_is_current(monkeypatch) -> None:
    reply = '[{"id":"f0","replacement":"consensus","confidence":0.9}]'
    calls = _patch_urlopen(
        monkeypatch, models=["google/gemini-2.5-flash"], completion_reply=reply
    )
    corrector = OpenRouterCorrector("google/gemini-2.5-flash", api_key="k")

    got = corrector.correct([_ctx("f0", "sensus", ["consensus"])])

    assert got[0].replacement == "consensus"
    assert calls == [OPENROUTER_MODELS_URL, OPENROUTER_URL]


def test_model_check_runs_once_per_instance(monkeypatch) -> None:
    reply = '[{"id":"f0","replacement":"consensus","confidence":0.9}]'
    calls = _patch_urlopen(
        monkeypatch, models=["google/gemini-2.5-flash"], completion_reply=reply
    )
    corrector = OpenRouterCorrector("google/gemini-2.5-flash", api_key="k")

    corrector.correct([_ctx("f0", "sensus", ["consensus"])])
    corrector.correct([_ctx("f0", "sensus", ["consensus"])])

    assert calls.count(OPENROUTER_MODELS_URL) == 1


def test_model_check_is_best_effort_on_catalogue_fetch_failure(monkeypatch) -> None:
    """If the catalogue check itself can't be reached, the real request
    still gets a chance to run rather than blocking the whole pass on it."""
    from urllib.error import URLError

    reply = '[{"id":"f0","replacement":"consensus","confidence":0.9}]'

    def fake_urlopen(request, timeout=None, context=None):  # noqa: ARG001
        url = request if isinstance(request, str) else request.full_url
        if url == OPENROUTER_MODELS_URL:
            raise URLError("network unreachable")
        return _FakeResponse(
            json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    corrector = OpenRouterCorrector("google/gemini-2.5-flash", api_key="k")

    got = corrector.correct([_ctx("f0", "sensus", ["consensus"])])

    assert got[0].replacement == "consensus"


def _modules_after_import(*imports: str) -> set[str]:
    code = (
        "import sys;"
        + "".join(f"import {name};" for name in imports)
        + "print('\\n'.join(sorted(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_detection_never_imports_an_llm_sdk() -> None:
    loaded = _modules_after_import("caption_checker.detect")
    assert {"openai", "httpx", "requests"}.isdisjoint(loaded)


def test_check_path_does_not_import_the_correction_stack() -> None:
    """`check` stays free and dependency-light (ADR-0002): importing the CLI
    must not drag in the orchestrator or the LLM backend."""
    loaded = _modules_after_import("caption_checker.cli")
    assert "caption_checker.correct" not in loaded
    assert "caption_checker.corrector" not in loaded


def test_importing_corrector_pulls_no_http_client() -> None:
    """The SDK guard: constructing a request is where urllib comes in, not
    import time -- so importing the module stays cheap and side-effect free."""
    loaded = _modules_after_import("caption_checker.corrector")
    assert {"openai", "httpx", "requests", "urllib.request"}.isdisjoint(loaded)


def test_parse_response_matches_ids_in_order() -> None:
    batch = [_ctx("f0", "sensus", ["consensus"]), _ctx("f1", "rey", [])]
    reply = (
        '[{"id":"f1","replacement":null,"confidence":0.4,"rationale":"ok"},'
        '{"id":"f0","replacement":"consensus","confidence":0.9}]'
    )
    got = parse_response(reply, batch)
    assert [c.id for c in got] == ["f0", "f1"]
    assert got[0] == Correction("f0", "consensus", 0.9, "")


def test_parse_response_rejects_non_json() -> None:
    with pytest.raises(CorrectorError):
        parse_response("sorry, I cannot help", [_ctx("f0", "x", [])])


def test_parse_response_rejects_id_mismatch() -> None:
    batch = [_ctx("f0", "x", [])]
    with pytest.raises(CorrectorError):
        parse_response('[{"id":"other","replacement":"y","confidence":1}]', batch)


def test_parse_response_strips_code_fence() -> None:
    batch = [_ctx("f0", "x", ["y"])]
    reply = '```json\n[{"id":"f0","replacement":"y","confidence":0.8}]\n```'
    assert parse_response(reply, batch)[0].replacement == "y"
