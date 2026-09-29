from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from caption_checker.readthrough import Reader, StubReader
from caption_checker.web.app import create_app
from caption_checker.web.models import TranscriptRecord
from caption_checker.web.source_video import MetadataLookup, VideoMetadata
from caption_checker.web.storage import Storage

DATA_DIR = Path(__file__).parent / "data"
SAMPLE = DATA_DIR / "sample_lecture.srt"
SAMPLE_VTT = DATA_DIR / "sample_lecture.vtt"


def _storage_for(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "data")


def _no_metadata(video_id: str) -> VideoMetadata | None:
    return None


def _make_client(
    tmp_path: Path,
    reader: Reader | None = None,
    video_lookup: MetadataLookup = _no_metadata,
) -> TestClient:
    app = create_app(_storage_for(tmp_path), reader=reader, video_lookup=video_lookup)
    return TestClient(app)


def _upload(
    client: TestClient,
    filename: str = "sample_lecture.srt",
    path: Path = SAMPLE,
    data: dict[str, str] | None = None,
) -> str:
    with path.open("rb") as f:
        response = client.post(
            "/transcripts",
            files={"file": (filename, f, "text/plain")},
            data=data,
            follow_redirects=False,
        )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


class TestSessionCookie:
    def test_index_sets_session_cookie(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        response = client.get("/")
        assert response.status_code == 200
        assert "cc_session" in response.cookies

    def test_reused_cookie_is_not_reissued(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.get("/")
        response = client.get("/")
        assert "set-cookie" not in response.headers


class TestUpload:
    def test_upload_scans_and_redirects_to_review_page(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}")
        assert page.status_code == 200
        assert "cubernetes" in page.text.lower()

    def test_index_lists_uploaded_transcript_with_flag_count(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        _upload(client)

        index = client.get("/")
        assert "sample_lecture.srt" in index.text

    def test_index_summary_shows_flag_count_and_reviewed_progress(
        self, tmp_path: Path
    ) -> None:
        # sample_lecture.srt yields 4 flags; review two of them.
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "consensus"},
        )
        client.post(
            f"/transcripts/{transcript_id}/flags/1/decision",
            data={"action": "reject"},
        )

        index = client.get("/")
        assert "2 / 4" in index.text

    def test_index_lists_every_transcript_uploaded_in_the_session(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        _upload(client, filename="first.srt")
        _upload(client, filename="second.srt")

        index = client.get("/")
        assert "first.srt" in index.text
        assert "second.srt" in index.text

    def test_index_links_to_transcripts_review_page(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        index = client.get("/")
        assert f'href="/transcripts/{transcript_id}"' in index.text

    def test_invalid_extension_shows_error_without_creating_transcript(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        response = client.post(
            "/transcripts",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.text.lower()

        index = client.get("/")
        assert "notes.txt" not in index.text

    def test_invalid_srt_content_shows_error(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        response = client.post(
            "/transcripts",
            files={"file": ("broken.srt", b"not an srt file at all", "text/plain")},
        )
        assert response.status_code == 400


class TestSessionIsolation:
    def test_other_session_cannot_see_transcript(self, tmp_path: Path) -> None:
        owner = _make_client(tmp_path)
        transcript_id = _upload(owner)

        stranger = _make_client(tmp_path)
        response = stranger.get(f"/transcripts/{transcript_id}")
        assert response.status_code == 404

    def test_unknown_transcript_id_is_404(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.get("/")  # establish a session
        response = client.get("/transcripts/does-not-exist")
        assert response.status_code == 404

    def test_index_excludes_other_sessions_transcripts(self, tmp_path: Path) -> None:
        # Two distinct TestClients == two distinct cookie jars == two
        # "browsers," each getting its own Session cookie.
        owner = _make_client(tmp_path)
        _upload(owner, filename="owner_only.srt")

        stranger = _make_client(tmp_path)
        _upload(stranger, filename="stranger_only.srt")

        owner_index = owner.get("/")
        assert "owner_only.srt" in owner_index.text
        assert "stranger_only.srt" not in owner_index.text

        stranger_index = stranger.get("/")
        assert "stranger_only.srt" in stranger_index.text
        assert "owner_only.srt" not in stranger_index.text


class TestReviewDecisions:
    def test_accept_decision_returns_partial_marked_accepted(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "Kubernetes"},
        )
        assert response.status_code == 200
        assert "accepted" in response.text.lower()
        assert "Kubernetes" in response.text

    def test_reject_decision_returns_partial_marked_rejected(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "reject"},
        )
        assert response.status_code == 200
        assert "rejected" in response.text.lower()

    def test_decision_persists_across_requests(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "Kubernetes"},
        )
        page = client.get(f"/transcripts/{transcript_id}")
        assert page.status_code == 200
        assert "1 reviewed" in page.text

    def test_unknown_flag_id_is_404(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/flags/9999/decision",
            data={"action": "accept"},
        )
        assert response.status_code == 404

    def test_review_page_defaults_accept_text_to_top_local_candidate(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}")
        assert 'value="consensus"' in page.text

    def test_unknown_action_is_400(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "frobnicate"},
        )
        assert response.status_code == 400


class TestCorrectPass:
    def test_correct_with_no_key_shows_actionable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Other tests in the suite trigger the CLI's load_dotenv(), which can
        # leak the real OPENROUTER_API_KEY into os.environ for the rest of
        # the process -- clear it explicitly (see test_corrector.py's
        # version of this same fixture-order issue).
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(f"/transcripts/{transcript_id}/correct", data={}, follow_redirects=True)
        assert response.status_code == 200
        assert "api key" in response.text.lower()

    def test_correct_with_session_key_populates_corrections(self, tmp_path: Path) -> None:
        stub = StubReader(replacement_for={"cubernetes": "Kubernetes"})
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        )
        assert response.status_code == 200
        # Per-flag-row content, not just the page-level summary line -- a
        # stale Correction field name in the template would still leave
        # "confirmed" in the summary but silently blank/wrong per row.
        assert 'LLM: confirmed' in response.text
        assert '→ "Kubernetes"' in response.text

    def test_correct_summary_shows_confirmed_and_dismissed_counts(
        self, tmp_path: Path
    ) -> None:
        # sample_lecture.srt yields 4 flags: "con sensus", "cough ka" (x2),
        # "cubernetes". Dismiss the first three, confirm the last.
        stub = StubReader(
            null_spans={"con sensus", "cough ka"},
            replacement_for={"cubernetes": "Kubernetes"},
        )
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "confirmed 1" in response.text.lower()
        assert "dismissed 3" in response.text.lower()

    def test_correct_does_not_rerun_once_corrected(self, tmp_path: Path) -> None:
        stub = StubReader(replacement_for={"cubernetes": "Kubernetes"})
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)
        client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        )

        page = client.get(f"/transcripts/{transcript_id}")
        assert "re-running isn't supported" in page.text.lower()
        assert 'name="api_key"' not in page.text

    def test_correct_failure_is_retriable_not_a_500(self, tmp_path: Path) -> None:
        stub = StubReader(garbage=True)
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert "error" in response.text.lower() or "failed" in response.text.lower()
        assert 'name="api_key"' in response.text  # retriable

    def test_form_has_a_priming_terms_field(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}")

        assert 'name="priming_terms"' in page.text

    def test_priming_terms_field_is_used_for_the_run(self, tmp_path: Path) -> None:
        stub = StubReader()
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)

        client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test", "priming_terms": "Kafka, Raft\nKubernetes"},
        )

        assert stub.requests[0].priming_terms == ["Kafka", "Raft", "Kubernetes"]

    def test_read_through_finds_show_as_reviewable_flags(self, tmp_path: Path) -> None:
        stub = StubReader(extra={"leader election": "leader elections"})
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)

        page = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        )

        assert "read_through" in page.text
        assert "5 flags" in page.text  # 4 local + 1 found
        assert "found 1 new" in page.text
        assert 'hx-post="/transcripts/%s/flags/4/decision"' % transcript_id in page.text

    def test_failed_chunks_are_reported_on_the_page(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, reader=StubReader())
        transcript_id = _upload(client)
        storage = _storage_for(tmp_path)
        session_id = client.cookies["cc_session"]
        record = storage.load_transcript(session_id, transcript_id)
        assert record is not None
        record.corrected_at, record.chunk_count, record.failed_chunks = "t", 3, 2
        storage.save_transcript(record)

        page = client.get(f"/transcripts/{transcript_id}")

        assert "2 of 3 chunks failed" in page.text
        assert "left unjudged" in page.text


class TestExport:
    def test_export_reflects_only_accepted_decisions(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "Kubernetes"},
        )
        response = client.get(f"/transcripts/{transcript_id}/export")
        assert response.status_code == 200
        assert "Kubernetes" in response.text
        assert 'filename="sample_lecture.corrected.srt"' in response.headers["content-disposition"]

    def test_export_available_before_full_review(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.get(f"/transcripts/{transcript_id}/export")
        assert response.status_code == 200

    def test_export_reflects_mix_of_accepted_rejected_and_pending_flags(
        self, tmp_path: Path
    ) -> None:
        # sample_lecture.srt yields 4 flags: 0 "con sensus" (cue 2),
        # 1 "cough ka" (cue 5), 2 "cough ka" (cue 6), 3 "cubernetes" (cue 7).
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "consensus"},
        )
        client.post(
            f"/transcripts/{transcript_id}/flags/1/decision",
            data={"action": "reject"},
        )
        # Flag 2 (cue 6) and flag 3 (cue 7) are left pending.

        response = client.get(f"/transcripts/{transcript_id}/export")
        assert response.status_code == 200
        body = response.text

        # Accepted flag's replacement text lands in its Cue.
        assert "con sensus algorithms" not in body
        assert "consensus algorithms" in body
        # Rejected flag's Cue is untouched.
        assert "We also need to talk about cough ka" in body
        # Pending flags' Cues are untouched.
        assert "Many companies use cough ka for event driven architectures" in body
        assert "cubernetes and container orchestration" in body

    def test_export_vtt_upload_downloads_as_vtt(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, filename="sample_lecture.vtt", path=SAMPLE_VTT)

        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": "consensus"},
        )
        response = client.get(f"/transcripts/{transcript_id}/export")

        assert response.status_code == 200
        assert response.text.startswith("WEBVTT")
        assert "consensus algorithms" in response.text
        assert 'filename="sample_lecture.corrected.vtt"' in response.headers["content-disposition"]


class TestCrossCueFlag:
    def test_a_find_across_cues_is_highlighted_and_exported(self, tmp_path: Path) -> None:
        path = tmp_path / "cross.srt"
        path.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nwe reached con\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nsensus\n\n"
            "3\n00:00:04,000 --> 00:00:06,000\nquickly.\n",
            encoding="utf-8",
        )
        client = _make_client(tmp_path, reader=StubReader(extra={"con sensus": "consensus"}))
        transcript_id = _upload(client, "cross.srt", path)

        page = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        ).text
        # the context runs across the Cue boundary, with the whole span marked
        assert "we reached <mark>con sensus</mark> quickly." in page
        assert "cues 1–2" in page

        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": ""},
        )
        exported = client.get(f"/transcripts/{transcript_id}/export").text
        assert exported == (
            "1\n00:00:00,000 --> 00:00:02,000\nwe reached consensus\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nquickly.\n\n"
        )


class TestDelete:
    def test_delete_removes_transcript(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(f"/transcripts/{transcript_id}/delete", follow_redirects=False)
        assert response.status_code == 303

        follow_up = client.get(f"/transcripts/{transcript_id}")
        assert follow_up.status_code == 404

    def test_delete_removes_transcript_from_index_list(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(f"/transcripts/{transcript_id}/delete")

        index = client.get("/")
        assert "sample_lecture.srt" not in index.text

    def test_delete_removes_stored_original_file(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        storage = _storage_for(tmp_path)
        session_id = client.cookies["cc_session"]
        assert storage.original_path(session_id, transcript_id) is not None

        client.post(f"/transcripts/{transcript_id}/delete")

        assert storage.original_path(session_id, transcript_id) is None


class TestReadThroughConfiguration:
    """#40: the configuration comes from env vars, resolved at startup."""

    def test_a_bad_configuration_fails_at_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from caption_checker.readthrough import ConfigError

        monkeypatch.setenv("OPENROUTER_CONFIG", "no-such-name")
        with pytest.raises(ConfigError, match="no-such-name"):
            create_app(_storage_for(tmp_path))

    def test_correct_runs_the_env_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from caption_checker.readthrough import CONFIGS
        from caption_checker.web import service

        built = []

        def reader(config, *, api_key):
            built.append(config)
            return StubReader()

        monkeypatch.setattr(service, "OpenRouterReader", reader)
        monkeypatch.setenv("OPENROUTER_CONFIG", "flash-v4")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        client.post(f"/transcripts/{transcript_id}/correct", follow_redirects=False)
        assert built == [CONFIGS["flash-v4"]]

    def test_both_env_vars_set_fails_at_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from caption_checker.readthrough import ConfigError

        monkeypatch.setenv("OPENROUTER_CONFIG", "flash-v4")
        monkeypatch.setenv("OPENROUTER_MODEL", "some/slug")
        with pytest.raises(ConfigError, match="not both"):
            create_app(_storage_for(tmp_path))


VIDEO_ID = "dQw4w9WgXcQ"
PLAYER_SCRIPT = "https://www.youtube.com/iframe_api"


def _record(tmp_path: Path, client: TestClient, transcript_id: str) -> TranscriptRecord:
    record = _storage_for(tmp_path).load_transcript(client.cookies["cc_session"], transcript_id)
    assert record is not None
    return record


def _video_id(tmp_path: Path, client: TestClient, transcript_id: str) -> str | None:
    return _record(tmp_path, client, transcript_id).video_id


class TestSourceVideo:
    """#37: an optional YouTube link per Transcript, played on the review page."""

    def test_upload_without_a_link_has_no_source_video(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        assert _video_id(tmp_path, client, transcript_id) is None

    def test_upload_form_offers_a_link_field(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        assert 'name="video_link"' in client.get("/").text

    def test_upload_with_a_link_stores_only_the_video_id(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(
            client, data={"video_link": f"https://youtu.be/{VIDEO_ID}?t=42"}
        )
        assert _video_id(tmp_path, client, transcript_id) == VIDEO_ID

    def test_upload_with_a_bad_link_shows_error_and_saves_nothing(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        with SAMPLE.open("rb") as f:
            response = client.post(
                "/transcripts",
                files={"file": ("sample_lecture.srt", f, "text/plain")},
                data={"video_link": "https://vimeo.com/12345"},
            )
        assert response.status_code == 400
        assert "youtube" in response.text.lower()
        assert "sample_lecture.srt" not in client.get("/").text

    def test_set_change_and_remove_on_the_review_page(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/video",
            data={"video_link": f"https://www.youtube.com/watch?v={VIDEO_ID}"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert _video_id(tmp_path, client, transcript_id) == VIDEO_ID

        client.post(f"/transcripts/{transcript_id}/video", data={"video_link": "abcdefghijk"})
        assert _video_id(tmp_path, client, transcript_id) == "abcdefghijk"

        response = client.post(
            f"/transcripts/{transcript_id}/video/delete", follow_redirects=False
        )
        assert response.status_code == 303
        assert _video_id(tmp_path, client, transcript_id) is None

    def test_a_bad_link_on_the_review_page_shows_error_and_keeps_the_old_one(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        response = client.post(
            f"/transcripts/{transcript_id}/video", data={"video_link": "not a link"}
        )
        assert response.status_code == 400
        assert "not a link" in response.text
        assert _video_id(tmp_path, client, transcript_id) == VIDEO_ID

    def test_other_session_cannot_set_the_link(self, tmp_path: Path) -> None:
        owner = _make_client(tmp_path)
        transcript_id = _upload(owner)

        stranger = _make_client(tmp_path)
        response = stranger.post(
            f"/transcripts/{transcript_id}/video", data={"video_link": VIDEO_ID}
        )
        assert response.status_code == 404

    def test_no_source_video_means_no_player_and_plain_timestamps(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text
        assert PLAYER_SCRIPT not in page
        assert 'id="source-player"' not in page
        assert 'class="play-span"' not in page
        # The control to add one is still there.
        assert f'action="/transcripts/{transcript_id}/video"' in page

    def test_source_video_adds_the_player_and_playable_timestamps(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        page = client.get(f"/transcripts/{transcript_id}").text
        assert PLAYER_SCRIPT in page
        assert 'id="source-player"' in page
        assert f'data-video-id="{VIDEO_ID}"' in page
        assert "https://www.youtube-nocookie.com" in page
        # Each timestamp is a fallback watch link carrying its span's bounds.
        assert 'class="play-span"' in page
        assert f"https://www.youtube.com/watch?v={VIDEO_ID}&amp;t=" in page
        assert "data-start=" in page and "data-end=" in page
        # The change/remove controls sit on the page.
        assert f'value="https://www.youtube.com/watch?v={VIDEO_ID}"' in page
        assert f'action="/transcripts/{transcript_id}/video/delete"' in page

    def test_decision_partial_keeps_the_playable_timestamp(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        response = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision", data={"action": "reject"}
        )
        assert 'class="play-span"' in response.text


class _Lookup:
    """A stub oEmbed lookup: ``found`` maps video IDs to their metadata,
    anything else fails. Records each ID it was asked for."""

    def __init__(self, found: dict[str, VideoMetadata]) -> None:
        self.found = found
        self.calls: list[str] = []

    def __call__(self, video_id: str) -> VideoMetadata | None:
        self.calls.append(video_id)
        return self.found.get(video_id)


OTHER_ID = "abcdefghijk"
RAFT = VideoMetadata(title="Raft in 10 minutes", channel="Distributed Dan")
KAFKA = VideoMetadata(title="Kafka internals", channel="Stream & Co")


class TestSourceVideoMetadata:
    """#38: the Source video's title and channel, offered as Priming terms."""

    def test_upload_with_a_link_stores_title_and_channel(self, tmp_path: Path) -> None:
        lookup = _Lookup({VIDEO_ID: RAFT})
        client = _make_client(tmp_path, video_lookup=lookup)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        record = _record(tmp_path, client, transcript_id)
        assert (record.video_title, record.video_channel) == (RAFT.title, RAFT.channel)
        assert lookup.calls == [VIDEO_ID]

    def test_upload_without_a_link_looks_nothing_up(self, tmp_path: Path) -> None:
        lookup = _Lookup({})
        client = _make_client(tmp_path, video_lookup=lookup)
        _upload(client)
        assert lookup.calls == []

    def test_a_bad_link_looks_nothing_up(self, tmp_path: Path) -> None:
        lookup = _Lookup({})
        client = _make_client(tmp_path, video_lookup=lookup)
        transcript_id = _upload(client)
        client.post(f"/transcripts/{transcript_id}/video", data={"video_link": "not a link"})
        assert lookup.calls == []

    def test_a_failed_lookup_saves_the_link_without_metadata(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, video_lookup=_Lookup({}))
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        record = _record(tmp_path, client, transcript_id)
        assert record.video_id == VIDEO_ID
        assert (record.video_title, record.video_channel) == (None, None)

    def test_changing_the_video_looks_it_up_again(self, tmp_path: Path) -> None:
        lookup = _Lookup({VIDEO_ID: RAFT, OTHER_ID: KAFKA})
        client = _make_client(tmp_path, video_lookup=lookup)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        client.post(f"/transcripts/{transcript_id}/video", data={"video_link": OTHER_ID})

        record = _record(tmp_path, client, transcript_id)
        assert (record.video_title, record.video_channel) == (KAFKA.title, KAFKA.channel)
        assert lookup.calls == [VIDEO_ID, OTHER_ID]

    def test_changing_to_a_video_whose_lookup_fails_drops_the_old_metadata(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path, video_lookup=_Lookup({VIDEO_ID: RAFT}))
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        client.post(f"/transcripts/{transcript_id}/video", data={"video_link": OTHER_ID})

        record = _record(tmp_path, client, transcript_id)
        assert record.video_id == OTHER_ID
        assert (record.video_title, record.video_channel) == (None, None)

    def test_removing_the_video_clears_title_and_channel(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, video_lookup=_Lookup({VIDEO_ID: RAFT}))
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        client.post(f"/transcripts/{transcript_id}/video/delete")

        record = _record(tmp_path, client, transcript_id)
        assert (record.video_title, record.video_channel) == (None, None)

    def test_priming_terms_are_prefilled_with_title_and_channel(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, video_lookup=_Lookup({OTHER_ID: KAFKA}))
        transcript_id = _upload(client, data={"video_link": OTHER_ID})

        page = client.get(f"/transcripts/{transcript_id}").text
        assert 'value="Kafka internals, Stream &amp; Co"' in page

    def test_priming_terms_prefill_with_only_a_title(self, tmp_path: Path) -> None:
        lookup = _Lookup({VIDEO_ID: VideoMetadata(title="Raft in 10 minutes", channel=None)})
        client = _make_client(tmp_path, video_lookup=lookup)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        page = client.get(f"/transcripts/{transcript_id}").text
        assert 'value="Raft in 10 minutes"' in page

    @pytest.mark.parametrize("video_link", [None, VIDEO_ID], ids=["no-video", "no-metadata"])
    def test_no_prefill_without_metadata(self, tmp_path: Path, video_link: str | None) -> None:
        client = _make_client(tmp_path)
        data = {"video_link": video_link} if video_link else None
        transcript_id = _upload(client, data=data)

        page = client.get(f"/transcripts/{transcript_id}").text
        assert 'name="priming_terms" value=' not in page

    def test_the_prefill_is_not_sent_unless_submitted(self, tmp_path: Path) -> None:
        stub = StubReader()
        client = _make_client(tmp_path, reader=stub, video_lookup=_Lookup({VIDEO_ID: RAFT}))
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-test"})

        assert stub.requests[0].priming_terms == []

    @pytest.mark.parametrize(
        "reader", [None, StubReader(garbage=True)], ids=["no-key", "all-chunks-failed"]
    )
    def test_a_failed_run_keeps_the_submitted_terms_not_the_suggestion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: Reader | None
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = _make_client(
            tmp_path, reader=reader or StubReader(), video_lookup=_Lookup({VIDEO_ID: RAFT})
        )
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        page = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "" if reader is None else "sk-or-test", "priming_terms": "Raft"},
        ).text

        assert 'class="error"' in page
        assert 'name="priming_terms" value="Raft"' in page
        assert RAFT.title not in page.split('name="priming_terms"')[1].split(">")[0]

    def test_a_failed_run_with_no_terms_offers_the_suggestion_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = _make_client(tmp_path, video_lookup=_Lookup({VIDEO_ID: RAFT}))
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        page = client.post(
            f"/transcripts/{transcript_id}/correct", data={"priming_terms": "  "}
        ).text

        assert 'value="Raft in 10 minutes, Distributed Dan"' in page

    def test_submitted_terms_are_kept_without_a_source_video(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(f"/transcripts/{transcript_id}/correct", data={"priming_terms": "Kafka"})

        page = client.get(f"/transcripts/{transcript_id}").text
        assert 'name="priming_terms" value="Kafka"' in page


def _cue_row(page: str, index: int) -> str:
    """The HTML of Cue ``index``'s row in the All Cues view."""
    start = page.index(f'id="cue-{index}"')
    end = page.find('class="cue-row"', start)
    return page[start : end if end != -1 else None]


class TestAllCuesView:
    def test_flags_is_the_default_view_and_all_cues_a_second_tab(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert 'id="flags-view" role="tabpanel"' in page
        assert 'id="cues-view" role="tabpanel" aria-labelledby="tab-cues" hidden' in page
        assert 'aria-selected="true">Flags</button>' in page
        assert 'aria-selected="false">All Cues</button>' in page
        assert "Each Cue is one timed caption line from your file" in page

    def test_every_cue_is_listed_in_order(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        positions = [page.index(f'id="cue-{i}"') for i in (1, 2, 3)]
        assert positions == sorted(positions)
        assert "Welcome back to the lecture on distributed systems." in _cue_row(page, 1)

    def test_flagged_spans_link_to_their_flag_card_with_their_status(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        record = _record(tmp_path, client, transcript_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.span == "con sensus")

        page = client.get(f"/transcripts/{transcript_id}").text

        assert (
            f'<a class="cue-flag pending" href="#flag-{flag_id}"' in _cue_row(page, 2)
        )

    def test_a_decision_refreshes_the_cues_it_touches(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        record = _record(tmp_path, client, transcript_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.span == "con sensus")

        response = client.post(
            f"/transcripts/{transcript_id}/flags/{flag_id}/decision",
            data={"action": "accept", "text": "consensus"},
        ).text

        assert f'id="flag-{flag_id}"' in response
        assert '<div class="cue-row" id="cue-2" hx-swap-oob="true">' in response
        assert f'<a class="cue-flag accepted" href="#flag-{flag_id}"' in response
        assert 'about <a class="cue-flag accepted"' in response
        assert ">consensus</a> algorithms." in response
        assert 'id="cue-1"' not in response

    def test_cue_timestamps_play_the_cue_with_a_source_video(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, data={"video_link": VIDEO_ID})

        row = _cue_row(client.get(f"/transcripts/{transcript_id}").text, 2)

        assert 'class="play-span"' in row
        assert 'data-start="3.5" data-end="7.2"' in row
        # the fallback link starts a second early, as a Flag's does
        assert f"https://www.youtube.com/watch?v={VIDEO_ID}&amp;t=2s" in row

    def test_cue_timestamps_are_plain_text_without_a_source_video(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        row = _cue_row(client.get(f"/transcripts/{transcript_id}").text, 2)

        assert "[00:00:03.500 → 00:00:07.200]" in row
        assert "play-span" not in row

    def test_a_cue_emptied_by_a_cross_cue_fix_shows_where_it_went(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cross.srt"
        path.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nwe reached con\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\nsensus\n\n"
            "3\n00:00:04,000 --> 00:00:06,000\nquickly.\n",
            encoding="utf-8",
        )
        client = _make_client(tmp_path, reader=StubReader(extra={"con sensus": "consensus"}))
        transcript_id = _upload(client, "cross.srt", path)
        client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-test"})
        client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "accept", "text": ""},
        )

        page = client.get(f"/transcripts/{transcript_id}").text

        assert "merged into Cue 1 by an accepted fix" in _cue_row(page, 2)
        assert "consensus" in _cue_row(page, 1)


class TestEditCueRoute:
    def _edit(self, client: TestClient, transcript_id: str, cue: int, text: str):
        return client.post(
            f"/transcripts/{transcript_id}/cues/{cue}/edit", data={"text": text}
        )

    def test_each_cue_has_an_edit_box_starting_from_its_exported_text(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        row = _cue_row(client.get(f"/transcripts/{transcript_id}").text, 1)

        assert f'hx-post="/transcripts/{transcript_id}/cues/1/edit"' in row
        assert ">Welcome back to the lecture on distributed systems.</textarea>" in row

    def test_saving_shows_the_new_span_and_adds_a_flag_card_in_order(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        n = len(_record(tmp_path, client, transcript_id).flags)

        response = self._edit(
            client, transcript_id, 1, "Welcome back to the lectures on distributed systems."
        )

        assert response.status_code == 200
        record = _record(tmp_path, client, transcript_id)
        assert len(record.flags) == n + 1
        flag_id = n
        body = response.text
        assert f'<a class="cue-flag accepted" href="#flag-{flag_id}"' in _cue_row(body, 1)
        assert 'id="flags-list" hx-swap-oob="true"' in body
        assert "Added by you" in body
        assert "1 added by you" in body
        # the new card sits in transcript order: before the Flags of later Cues
        assert body.index(f'id="flag-{flag_id}"') < body.index('id="flag-0"')

    def test_the_page_shows_the_added_by_you_badge_and_count(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        self._edit(client, transcript_id, 1, "Welcome back to the lectures on distributed systems.")

        page = client.get(f"/transcripts/{transcript_id}").text

        assert 'class="badge">Added by you' in page
        assert "1 added by you" in page

    def test_export_writes_the_edit_and_rejecting_undoes_it(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        before = client.get(f"/transcripts/{transcript_id}/export").text
        n = len(_record(tmp_path, client, transcript_id).flags)
        self._edit(client, transcript_id, 1, "Welcome back to the lectures on distributed systems.")

        assert client.get(f"/transcripts/{transcript_id}/export").text == before.replace(
            "lecture ", "lectures "
        )
        response = client.post(
            f"/transcripts/{transcript_id}/flags/{n}/decision", data={"action": "reject"}
        )

        assert "Welcome back to the lecture on distributed systems." in response.text
        assert client.get(f"/transcripts/{transcript_id}/export").text == before

    def test_an_unsaved_change_is_reported_and_keeps_the_text_in_the_box(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        n = len(_record(tmp_path, client, transcript_id).flags)

        body = self._edit(
            client, transcript_id, 1, "Welcome back to the lecture on distributed systems"
        ).text

        assert "Not saved" in body and "punctuation" in body
        assert ">Welcome back to the lecture on distributed systems</textarea>" in body
        assert len(_record(tmp_path, client, transcript_id).flags) == n

    def test_an_unchanged_text_says_there_is_nothing_to_save(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        body = self._edit(
            client, transcript_id, 1, "Welcome back to the lecture on distributed systems."
        ).text

        assert "No changes to save." in body

    def test_an_empty_cue_is_refused_with_the_text_kept(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        body = self._edit(client, transcript_id, 1, "   ").text

        assert "A Cue can&#39;t be left empty." in body or "A Cue can't be left empty." in body

    def test_unknown_cue_is_404(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        assert self._edit(client, transcript_id, 99, "text").status_code == 404


class TestGlossaryHints:
    """First-time reviewers meet glossary terms; the page explains them (#43)."""

    def test_flags_tab_explains_what_a_flag_is(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert "A Flag is a span of the captions that may be a mistake" in page
        assert "edit the Cue on the All Cues tab" in page

    def test_page_explains_status_and_detector_on_every_flag_card(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert 'class="status-pill" title="Your Review Decision' in page
        assert "Dismissed: the LLM judged it not an error" in page
        assert 'title="What raised this Flag' in page

    def test_rerendered_flag_card_keeps_its_tooltips(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        card = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "reject"},
        ).text

        assert 'class="status-pill" title="Your Review Decision' in card
        assert 'title="What raised this Flag' in card

    def test_llm_verdict_line_is_explained_on_page_and_card(self, tmp_path: Path) -> None:
        stub = StubReader(replacement_for={"cubernetes": "Kubernetes"})
        client = _make_client(tmp_path, reader=stub)
        transcript_id = _upload(client)
        page = client.post(
            f"/transcripts/{transcript_id}/correct",
            data={"api_key": "sk-or-test"},
            follow_redirects=True,
        ).text
        card = client.post(
            f"/transcripts/{transcript_id}/flags/0/decision",
            data={"action": "reject"},
        ).text

        for body in (page, card):
            assert "Confirmed: the model agrees it is an error and proposes the replacement" in body

    def test_added_by_you_badge_is_explained(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        body = client.post(
            f"/transcripts/{transcript_id}/cues/1/edit",
            data={"text": "Welcome back to the lectures on distributed systems."},
        ).text

        assert 'title="You raised this Flag by editing a Cue' in body

    def test_added_by_you_tooltip_is_on_the_page_and_the_rerendered_card(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        n = len(_record(tmp_path, client, transcript_id).flags)
        client.post(
            f"/transcripts/{transcript_id}/cues/1/edit",
            data={"text": "Welcome back to the lectures on distributed systems."},
        )

        page = client.get(f"/transcripts/{transcript_id}").text
        card = client.post(
            f"/transcripts/{transcript_id}/flags/{n}/decision", data={"action": "reject"}
        ).text

        for body in (page, card):
            assert 'title="You raised this Flag by editing a Cue' in body
