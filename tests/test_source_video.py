from __future__ import annotations

import pytest

from caption_checker.web.source_video import (
    InvalidVideoLinkError,
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
