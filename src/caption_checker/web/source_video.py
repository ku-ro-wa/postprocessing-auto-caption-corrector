"""A Transcript's Source video: parsing a pasted YouTube link down to the
11-character video ID that's all we store, and building links back to it.

The video itself is never fetched — the review page plays it in YouTube's
own embedded player (see ``templates/transcript.html``). Only its title and
channel are looked up, once, from YouTube oEmbed (:func:`lookup_metadata`),
to offer as Priming terms.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from http.client import HTTPException
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

# How much of the video around a Flag's span a timestamp click plays:
# from ``start - PLAY_LEAD_SECONDS`` (clamped at 0) to ``end + PLAY_TAIL_SECONDS``.
PLAY_LEAD_SECONDS = 1.0
PLAY_TAIL_SECONDS = 1.0

# A lookup that takes longer than this is abandoned and the link saved
# without metadata: it runs inside the save request.
OEMBED_TIMEOUT_SECONDS = 3.0

_OEMBED_ENDPOINT = "https://www.youtube.com/oembed"

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
_SHORT_HOSTS = {"youtu.be"}


class InvalidVideoLinkError(ValueError):
    """Raised when pasted input isn't a YouTube link or video ID we accept."""


def parse_video_id(raw: str) -> str:
    """The video ID from a ``youtube.com/watch?v=``, ``m.youtube.com``,
    ``youtu.be/``, ``/shorts/`` or ``/embed/`` link, or a bare ID. Other
    parameters (``&t=``, ``&list=``, …) are discarded."""
    text = raw.strip()
    if _VIDEO_ID.fullmatch(text):
        return text

    candidate = _id_from_url(text if "://" in text else f"https://{text}")
    if candidate is None or not _VIDEO_ID.fullmatch(candidate):
        raise InvalidVideoLinkError(
            f"Couldn't find a YouTube video in {text!r}. Paste a youtube.com or "
            "youtu.be link, or the 11-character video ID."
        )
    return candidate


def _id_from_url(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.split("/") if s]

    if host in _SHORT_HOSTS:
        return segments[0] if len(segments) == 1 else None
    if host in _YOUTUBE_HOSTS:
        if segments == ["watch"]:
            return parse_qs(parts.query).get("v", [None])[0]
        if len(segments) == 2 and segments[0] in ("shorts", "embed"):
            return segments[1]
    return None


def watch_url(video_id: str, *, at: int | None = None) -> str:
    """The video's youtube.com page, optionally starting ``at`` seconds in."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    return url if at is None else f"{url}&t={at}s"


def span_watch_url(video_id: str, start: float) -> str:
    """Where a Flag's timestamp links when the video can't play embedded:
    its span's play window start, in whole seconds."""
    return watch_url(video_id, at=max(0, math.floor(start - PLAY_LEAD_SECONDS)))


@dataclass(frozen=True)
class VideoMetadata:
    """What YouTube oEmbed says about a Source video: its ``title`` and its
    ``author_name``, which is the channel."""

    title: str | None
    channel: str | None


#: ``fetch(url, timeout)`` returns the response body, or raises ``OSError``
#: (``URLError``/``HTTPError``/``TimeoutError`` included) or
#: ``http.client.HTTPException`` when the request fails.
Fetch = Callable[[str, float], bytes]

#: Looks up a video ID's metadata; ``None`` when it can't be had.
MetadataLookup = Callable[[str], "VideoMetadata | None"]


def _urlopen_fetch(url: str, timeout: float) -> bytes:
    import ssl
    from urllib.request import urlopen

    import certifi

    # Same trust-store workaround as ``OpenRouterCorrector.chat``.
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(url, timeout=timeout, context=context) as response:
        return response.read()


def lookup_metadata(video_id: str, *, fetch: Fetch = _urlopen_fetch) -> VideoMetadata | None:
    """The video's title and channel from one oEmbed request, or ``None``
    when the request fails (timeout, network error, 401/404 for a private,
    deleted or embed-disabled video) or the reply has neither. Never raises
    for any of those: a Source video is saved with or without its metadata."""
    url = f"{_OEMBED_ENDPOINT}?{urlencode({'url': watch_url(video_id), 'format': 'json'})}"
    try:
        data = json.loads(fetch(url, OEMBED_TIMEOUT_SECONDS))
    except (OSError, HTTPException, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    def nonblank(key: str) -> str | None:
        value = data.get(key)
        return (value.strip() or None) if isinstance(value, str) else None

    metadata = VideoMetadata(title=nonblank("title"), channel=nonblank("author_name"))
    return metadata if metadata.title or metadata.channel else None
