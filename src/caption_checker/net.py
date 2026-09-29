"""HTTPS through the standard library, verified against certifi's CA bundle.

Some Python installs (notably python.org's macOS builds) ship without a
wired-up system trust store, so the stdlib's default SSL context can't
verify any server's certificate. Every outbound request goes through
:func:`open_url`, which points it at certifi's bundle explicitly rather
than relying on the environment being set up right.

``urllib`` is imported lazily, on the call, so importing this module never
imports an HTTP client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from http.client import HTTPResponse
    from urllib.request import Request


def open_url(request: str | Request, *, timeout: float) -> HTTPResponse:
    """``urllib.request.urlopen(request)`` with certifi's CA bundle. Raises
    what ``urlopen`` raises."""
    import ssl
    from urllib import request as urllib_request

    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    return urllib_request.urlopen(request, timeout=timeout, context=context)
