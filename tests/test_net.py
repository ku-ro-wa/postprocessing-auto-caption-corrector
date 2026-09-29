from __future__ import annotations

import ssl

import certifi
import pytest

from caption_checker.net import open_url


def test_open_url_verifies_against_certifis_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, float, ssl.SSLContext]] = []

    def fake_urlopen(request: object, *, timeout: float, context: ssl.SSLContext) -> str:
        calls.append((request, timeout, context))
        return "response"

    loaded: list[str] = []
    real_load = ssl.SSLContext.load_verify_locations
    monkeypatch.setattr(
        ssl.SSLContext,
        "load_verify_locations",
        lambda self, cafile=None, *a, **k: (loaded.append(cafile), real_load(self, cafile, *a, **k))[1],
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert open_url("https://example.com/", timeout=5) == "response"

    [(request, timeout, context)] = calls
    assert (request, timeout) == ("https://example.com/", 5)
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert loaded == [certifi.where()]
