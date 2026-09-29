from __future__ import annotations

import json
from email.message import Message
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest

from caption_checker.web.source_video import (
    OEMBED_TIMEOUT_SECONDS,
    InvalidVideoLinkError,
    VideoMetadata,
    lookup_metadata,
    parse_video_id,
    span_watch_url,
    watch_url,
)

VIDEO_ID = "dQw4w9WgXcQ"


class TestParseVideoId:
    @pytest.mark.parametrize(
        "raw",
        [
            f"https://www.youtube.com/watch?v={VIDEO_ID}",
            f"https://youtube.com/watch?v={VIDEO_ID}",
            f"http://www.youtube.com/watch?v={VIDEO_ID}",
            f"www.youtube.com/watch?v={VIDEO_ID}",
            f"https://m.youtube.com/watch?v={VIDEO_ID}",
            f"https://youtu.be/{VIDEO_ID}",
            f"youtu.be/{VIDEO_ID}",
            f"https://www.youtube.com/shorts/{VIDEO_ID}",
            f"https://www.youtube.com/embed/{VIDEO_ID}",
            VIDEO_ID,
            f"  {VIDEO_ID}\n",
        ],
    )
    def test_accepted_forms(self, raw: str) -> None:
        assert parse_video_id(raw) == VIDEO_ID

    @pytest.mark.parametrize(
        "raw",
        [
            f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s",
            f"https://www.youtube.com/watch?list=PL123&v={VIDEO_ID}&index=3",
            f"https://youtu.be/{VIDEO_ID}?t=42",
            f"https://youtu.be/{VIDEO_ID}?si=abcdef",
            f"https://www.youtube.com/shorts/{VIDEO_ID}?feature=share",
            f"https://www.youtube.com/embed/{VIDEO_ID}?start=10",
            f"https://www.youtube.com/watch?v={VIDEO_ID}#comments",
        ],
    )
    def test_other_parameters_are_discarded(self, raw: str) -> None:
        assert parse_video_id(raw) == VIDEO_ID

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "not a link",
            "dQw4w9WgXc",  # 10 chars
            "dQw4w9WgXcQQ",  # 12 chars
            "dQw4w9WgX!Q",
            f"https://vimeo.com/{VIDEO_ID}",
            f"https://www.youtube.com.evil.example/watch?v={VIDEO_ID}",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/watch?v=short",
            f"https://www.youtube.com/channel/{VIDEO_ID}",
            "https://www.youtube.com/",
            "https://youtu.be/",
            f"https://youtu.be/{VIDEO_ID}/extra",
            f"ftp://youtu.be/{VIDEO_ID}",
        ],
    )
    def test_rejects(self, raw: str) -> None:
        with pytest.raises(InvalidVideoLinkError):
            parse_video_id(raw)


def test_watch_url() -> None:
    assert watch_url(VIDEO_ID) == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert watch_url(VIDEO_ID, at=41) == f"https://www.youtube.com/watch?v={VIDEO_ID}&t=41s"


@pytest.mark.parametrize(
    ("start", "expected_t"),
    [(42.7, 41), (1.0, 0), (0.4, 0), (0.0, 0)],
)
def test_span_watch_url_starts_one_second_early_floored_and_clamped(
    start: float, expected_t: int
) -> None:
    assert span_watch_url(VIDEO_ID, start) == watch_url(VIDEO_ID, at=expected_t)


class TestLookupMetadata:
    """#38: the Source video's title and channel, from YouTube oEmbed."""

    def test_success_reads_title_and_author_name(self) -> None:
        calls: list[tuple[str, float]] = []

        def fetch(url: str, timeout: float) -> bytes:
            calls.append((url, timeout))
            return json.dumps(
                {"title": "Raft in 10 minutes", "author_name": "Distributed Dan", "type": "video"}
            ).encode()

        assert lookup_metadata(VIDEO_ID, fetch=fetch) == VideoMetadata(
            title="Raft in 10 minutes", channel="Distributed Dan"
        )
        assert calls == [
            (
                "https://www.youtube.com/oembed?url=https%3A%2F%2Fwww.youtube.com%2F"
                f"watch%3Fv%3D{VIDEO_ID}&format=json",
                OEMBED_TIMEOUT_SECONDS,
            )
        ]

    @pytest.mark.parametrize(
        "error",
        [
            HTTPError("https://www.youtube.com/oembed", 401, "Unauthorized", Message(), None),
            HTTPError("https://www.youtube.com/oembed", 404, "Not Found", Message(), None),
            URLError("nodename nor servname provided"),
            TimeoutError("timed out"),
            OSError("connection reset"),
            IncompleteRead(b""),
        ],
        ids=["401", "404", "network", "timeout", "reset", "incomplete"],
    )
    def test_a_failed_request_means_no_metadata(self, error: Exception) -> None:
        def fetch(url: str, timeout: float) -> bytes:
            raise error

        assert lookup_metadata(VIDEO_ID, fetch=fetch) is None

    @pytest.mark.parametrize(
        "body",
        [b"<html>not json</html>", b"[]", b"{}", json.dumps({"title": 3}).encode()],
        ids=["html", "list", "empty", "wrong-type"],
    )
    def test_an_unusable_reply_means_no_metadata(self, body: bytes) -> None:
        assert lookup_metadata(VIDEO_ID, fetch=lambda url, timeout: body) is None

    def test_a_missing_channel_keeps_the_title(self) -> None:
        body = json.dumps({"title": "Raft in 10 minutes"}).encode()
        assert lookup_metadata(VIDEO_ID, fetch=lambda url, timeout: body) == VideoMetadata(
            title="Raft in 10 minutes", channel=None
        )
