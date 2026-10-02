from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from caption_checker.readthrough import Reader, StubReader
from caption_checker.web.app import create_app
from caption_checker.web.free_tier import LEDGER_FILENAME, Limits
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
    **kwargs: Any,
) -> TestClient:
    app = create_app(
        _storage_for(tmp_path), reader=reader, video_lookup=video_lookup, **kwargs
    )
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

    def test_correct_with_a_key_populates_corrections(self, tmp_path: Path) -> None:
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


class TestVisitorKey:
    """#45 / ADR 0010: a visitor's own key lives in their browser and is
    sent with each Correct; the server uses it for that run only."""

    @pytest.fixture
    def keys_used(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """The key each run built its reader with. Every run fails, so the
        Transcript stays retriable."""
        from caption_checker.web import service

        used: list[str] = []

        def reader(config, *, api_key):
            used.append(api_key)
            return StubReader(garbage=True)

        monkeypatch.setattr(service, "OpenRouterReader", reader)
        monkeypatch.setenv("OPENROUTER_CONFIG", "flash-v4")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        return used

    def test_a_correct_with_a_key_leaves_no_key_on_disk(
        self, tmp_path: Path, keys_used: list[str]
    ) -> None:
        # The run fails, so its correct_error is written too.
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-secret"})

        assert keys_used == ["sk-or-secret"]
        assert _record(tmp_path, client, transcript_id).correct_error

        stored = [p for p in (tmp_path / "data").rglob("*") if p.is_file()]
        assert stored
        assert not [p for p in stored if b"sk-or-secret" in p.read_bytes()]

    def test_a_correct_without_a_key_uses_the_server_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys_used: list[str]
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "server-key")
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "  "})

        assert keys_used == ["server-key"]

    def test_a_submitted_key_is_used_for_that_run_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys_used: list[str]
    ) -> None:
        # The failing run with the visitor's key never retries on the
        # server's, and the next run without one doesn't remember it.
        monkeypatch.setenv("OPENROUTER_API_KEY", "server-key")
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-mine"})
        client.post(f"/transcripts/{transcript_id}/correct", data={})

        assert keys_used == ["sk-or-mine", "server-key"]

    def test_a_key_left_in_an_old_session_file_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys_used: list[str]
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = _make_client(tmp_path)
        transcript_id = _upload(client)
        session_dir = tmp_path / "data" / "sessions" / client.cookies["cc_session"]
        (session_dir / "session.json").write_text('{"api_key": "sk-or-old"}')

        page = client.post(f"/transcripts/{transcript_id}/correct", data={}).text

        assert keys_used == []
        assert "api key" in page.lower()

    def test_the_page_says_the_key_stays_in_the_browser(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert 'name="api_key"' in page
        assert "never stored on the server" in page
        assert "Forget my key" in page


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


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


class TestRetention:
    """#46: a Transcript is deleted 24 hours after its last activity."""

    def _client(self, tmp_path: Path, clock: _Clock) -> tuple[TestClient, Storage]:
        storage = Storage(tmp_path / "data", clock=clock)
        return TestClient(create_app(storage, video_lookup=_no_metadata)), storage

    @pytest.mark.parametrize(
        ("method", "path", "data"),
        [
            ("post", "/flags/0/decision", {"action": "reject"}),
            ("post", "/cues/1/edit", {"text": "Edited by the reviewer"}),
            ("get", "/export", None),
            ("post", "/correct", {"api_key": "sk-visitor"}),
        ],
    )
    def test_activity_resets_the_clock(
        self, tmp_path: Path, method: str, path: str, data: dict[str, str] | None
    ) -> None:
        clock = _Clock()
        storage = Storage(tmp_path / "data", clock=clock)
        app = create_app(storage, reader=StubReader(), video_lookup=_no_metadata)
        client = TestClient(app)
        transcript_id = _upload(client)
        session_id = client.cookies["cc_session"]

        clock.now += timedelta(hours=20)
        kwargs = {"data": data} if data is not None else {}
        assert getattr(client, method)(f"/transcripts/{transcript_id}{path}", **kwargs).is_success
        clock.now += timedelta(hours=10)
        storage.sweep(timedelta(hours=24), keep_empty_sessions_for=timedelta(hours=24))

        assert storage.load_transcript(session_id, transcript_id) is not None

    def test_viewing_is_not_activity(self, tmp_path: Path) -> None:
        clock = _Clock()
        client, storage = self._client(tmp_path, clock)
        transcript_id = _upload(client)

        clock.now += timedelta(hours=20)
        client.get(f"/transcripts/{transcript_id}")
        clock.now += timedelta(hours=4)
        storage.sweep(timedelta(hours=24), keep_empty_sessions_for=timedelta(hours=24))

        assert client.get(f"/transcripts/{transcript_id}").status_code == 404

    def test_expired_transcripts_are_swept_at_startup(self, tmp_path: Path) -> None:
        clock = _Clock()
        client, _ = self._client(tmp_path, clock)
        transcript_id = _upload(client)

        clock.now += timedelta(hours=24)
        storage = Storage(tmp_path / "data", clock=clock)
        app = create_app(storage, video_lookup=_no_metadata)
        with TestClient(app, cookies=dict(client.cookies)) as restarted:
            assert restarted.get(f"/transcripts/{transcript_id}").status_code == 404

    def test_the_retention_period_is_configurable(self, tmp_path: Path) -> None:
        clock = _Clock()
        client, _ = self._client(tmp_path, clock)
        transcript_id = _upload(client)

        clock.now += timedelta(hours=2)
        storage = Storage(tmp_path / "data", clock=clock)
        app = create_app(storage, video_lookup=_no_metadata, retention=timedelta(hours=1))
        with TestClient(app, cookies=dict(client.cookies)) as restarted:
            assert restarted.get(f"/transcripts/{transcript_id}").status_code == 404

    def test_a_fresh_transcript_survives_startup(self, tmp_path: Path) -> None:
        clock = _Clock()
        client, _ = self._client(tmp_path, clock)
        transcript_id = _upload(client)

        clock.now += timedelta(hours=23)
        storage = Storage(tmp_path / "data", clock=clock)
        app = create_app(storage, video_lookup=_no_metadata)
        with TestClient(app, cookies=dict(client.cookies)) as restarted:
            assert restarted.get(f"/transcripts/{transcript_id}").status_code == 200

    def test_upload_page_says_how_long_a_file_is_kept(self, tmp_path: Path) -> None:
        page = _make_client(tmp_path).get("/").text
        assert "transcript is kept for 24 h after you last use it, or until you delete it" in page
        assert "sends its text to OpenRouter" in page

    def test_review_page_offers_delete_beside_export(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        header = page[page.index("Export corrected") :]
        assert f'action="/transcripts/{transcript_id}/delete"' in header
        assert 'confirm("Delete sample_lecture.srt and its review state?")' in header

    def test_delete_prompt_survives_a_quote_in_the_filename(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        transcript_id = _upload(client, filename="it's.srt")

        page = client.get(f"/transcripts/{transcript_id}").text

        assert "confirm(\"Delete it\\u0027s.srt and its review state?\")" in page


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


class TestFreeTier:
    """#36 / ADR 0008: a Correct with no key of the visitor's own runs on the
    server's key, metered by the Session's Allowance and the Daily budget."""

    @pytest.fixture(autouse=True)
    def server_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "server-key")
        monkeypatch.delenv("OPENROUTER_CONFIG", raising=False)
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    def _words(self, tmp_path: Path, client: TestClient, transcript_id: str) -> int:
        from caption_checker.web import service

        record = _record(tmp_path, client, transcript_id)
        return service.transcript_word_count(_storage_for(tmp_path), record)

    def _sample_words(self, tmp_path: Path) -> int:
        client = _make_client(tmp_path / "count", limits=None)
        return self._words(tmp_path / "count", client, _upload(client))

    def _ledger(self, tmp_path: Path) -> list[str]:
        path = tmp_path / "data" / LEDGER_FILENAME
        return path.read_text().splitlines() if path.exists() else []

    def test_limits_are_on_by_default(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, reader=StubReader())
        transcript_id = _upload(client)
        words = self._words(tmp_path, client, transcript_id)

        client.post(f"/transcripts/{transcript_id}/correct", data={})

        assert _record(tmp_path, client, transcript_id).corrected
        assert len(self._ledger(tmp_path)) == 1
        # The count shows where the next Correct would be sent.
        page = client.get(f"/transcripts/{_upload(client)}").text
        assert f"{10_000 - words:,} of 10,000 words left in your rolling 24 hours" in page
        assert f"{words:,} come back in 24 h" in page

    def test_the_page_shows_words_left_and_this_transcripts_words(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(tmp_path, limits=Limits(allowance_words=5_000))
        transcript_id = _upload(client)
        words = self._words(tmp_path, client, transcript_id)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert "5,000 of 5,000 words left in your rolling 24 hours" in page
        assert f"This Transcript is {words:,} words" in page
        assert "local/dev fallback" not in page

    def test_a_run_over_the_allowance_is_refused(self, tmp_path: Path) -> None:
        client = _make_client(
            tmp_path, reader=_NeverRead(), limits=Limits(allowance_words=10)
        )
        transcript_id = _upload(client)

        response = client.post(f"/transcripts/{transcript_id}/correct", data={})

        assert response.status_code == 429
        assert "Free tier Allowance is used up" in response.text
        assert "longer than the Free tier allows" in response.text
        assert "Enter your own OpenRouter key" in response.text
        assert "Donate" not in response.text
        assert not _record(tmp_path, client, transcript_id).corrected
        assert self._ledger(tmp_path) == []

    def test_an_allowance_refusal_says_when_this_transcript_fits(
        self, tmp_path: Path
    ) -> None:
        # An Allowance of exactly one sample: the first run spends it, and the
        # second fits once that run ages out of the window.
        words = self._sample_words(tmp_path)
        client = _make_client(
            tmp_path, reader=StubReader(), limits=Limits(allowance_words=words)
        )
        client.post(f"/transcripts/{_upload(client)}/correct", data={})

        response = client.post(f"/transcripts/{_upload(client)}/correct", data={})

        assert response.status_code == 429
        assert "You'll have enough in 24 h" in response.text

    def test_a_run_over_the_daily_budget_is_refused(self, tmp_path: Path) -> None:
        client = _make_client(
            tmp_path,
            reader=_NeverRead(),
            limits=Limits(daily_budget_usd=0.0, donate_url="https://example.org/give"),
        )
        transcript_id = _upload(client)

        response = client.post(f"/transcripts/{transcript_id}/correct", data={})

        assert response.status_code == 429
        assert "Daily budget" in response.text
        assert "is used up" in response.text
        assert 'href="https://example.org/give"' in response.text

    def test_a_run_on_the_visitors_own_key_is_never_metered(
        self, tmp_path: Path
    ) -> None:
        client = _make_client(
            tmp_path,
            reader=StubReader(),
            limits=Limits(allowance_words=10, daily_budget_usd=0.0),
        )
        transcript_id = _upload(client)

        response = client.post(
            f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-mine"}
        )

        assert response.status_code == 200
        assert _record(tmp_path, client, transcript_id).corrected
        assert self._ledger(tmp_path) == []

    def test_limits_off_means_no_ledger_and_no_refusal(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path, reader=StubReader(), limits=None)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text
        client.post(f"/transcripts/{transcript_id}/correct", data={})

        assert "words left in your rolling 24 hours" not in page
        assert _record(tmp_path, client, transcript_id).corrected
        assert self._ledger(tmp_path) == []

    def test_an_unpriced_model_fails_at_startup_with_limits_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from caption_checker.readthrough import ConfigError

        monkeypatch.setenv("OPENROUTER_MODEL", "some/slug")
        with pytest.raises(ConfigError, match="no known price"):
            create_app(_storage_for(tmp_path))
        create_app(_storage_for(tmp_path), limits=None)
        # No server key, no Free tier to price.
        monkeypatch.delenv("OPENROUTER_API_KEY")
        create_app(_storage_for(tmp_path))

    def test_with_no_server_key_the_page_asks_for_the_visitors_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
        client = _make_client(tmp_path)
        transcript_id = _upload(client)

        page = client.get(f"/transcripts/{transcript_id}").text

        assert "words left in your rolling 24 hours" not in page
        assert "no Free tier" in page


class _NeverRead(StubReader):
    def read(self, request):  # type: ignore[override]
        raise AssertionError("a refused run must not reach the model")


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


class TestServeLimits:
    """#36: `serve` meters the server key by default; local use turns the
    limits off explicitly."""

    @pytest.fixture
    def limits_used(self, monkeypatch: pytest.MonkeyPatch) -> list[Limits | None]:
        import uvicorn

        from caption_checker.web import app as web_app

        used: list[Limits | None] = []

        def fake_create_app(
            storage: Storage, *, limits: Limits | None, retention: timedelta
        ) -> object:
            used.append(limits)
            self.retention = retention
            return object()

        monkeypatch.setattr(web_app, "create_app", fake_create_app)
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
        for name in (
            "NO_LIMITS", "ALLOWANCE_WORDS", "DAILY_BUDGET_USD", "DONATE_URL", "RETENTION_HOURS"
        ):
            monkeypatch.delenv(f"CAPTION_CHECKER_{name}", raising=False)
        return used

    def _serve(self, tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> None:
        from click.testing import CliRunner

        from caption_checker.cli import main

        result = CliRunner().invoke(
            main, ["serve", "--data-dir", str(tmp_path), *args], env=env
        )
        assert result.exit_code == 0, result.output

    def test_limits_are_on_with_launch_figures_by_default(
        self, tmp_path: Path, limits_used: list[Limits | None]
    ) -> None:
        self._serve(tmp_path)
        assert limits_used == [Limits(allowance_words=10_000, daily_budget_usd=0.25)]

    def test_no_limits_turns_them_off(
        self, tmp_path: Path, limits_used: list[Limits | None]
    ) -> None:
        self._serve(tmp_path, "--no-limits")
        assert limits_used == [None]

    def test_figures_and_donate_link_are_configurable(
        self, tmp_path: Path, limits_used: list[Limits | None]
    ) -> None:
        self._serve(
            tmp_path,
            "--allowance-words",
            "5000",
            env={
                "CAPTION_CHECKER_DAILY_BUDGET_USD": "1.5",
                "CAPTION_CHECKER_DONATE_URL": "https://example.org/give",
            },
        )
        assert limits_used == [
            Limits(
                allowance_words=5_000,
                daily_budget_usd=1.5,
                donate_url="https://example.org/give",
            )
        ]

    def test_transcripts_are_kept_24_hours_by_default(
        self, tmp_path: Path, limits_used: list[Limits | None]
    ) -> None:
        self._serve(tmp_path)
        assert self.retention == timedelta(hours=24)

    def test_the_retention_period_is_configurable(
        self, tmp_path: Path, limits_used: list[Limits | None]
    ) -> None:
        self._serve(tmp_path, env={"CAPTION_CHECKER_RETENTION_HOURS": "1.5"})
        assert self.retention == timedelta(hours=1.5)


@pytest.mark.parametrize(
    ("span", "shown"),
    [
        (timedelta(hours=5, minutes=12), "5 h 12 min"),
        (timedelta(hours=19), "19 h"),
        (timedelta(minutes=40), "40 min"),
        (timedelta(hours=2, seconds=1), "2 h 1 min"),  # rounded up, never early
        (timedelta(seconds=5), "1 min"),
    ],
)
def test_waits_are_shown_rounded_up_to_the_minute(span: timedelta, shown: str) -> None:
    from caption_checker.web.app import _duration

    assert _duration(span) == shown
