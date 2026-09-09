"""The Corrector protocol layer: the stub, the real backend's guard rails, and
the reply parser. No network anywhere."""

from __future__ import annotations

import subprocess
import sys

import pytest

from caption_checker.corrector import (
    Correction,
    CorrectorError,
    FlagContext,
    MissingAPIKeyError,
    OpenRouterCorrector,
    StubCorrector,
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
