from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from caption_checker.readthrough import Reader, StubReader
from caption_checker.web.app import create_app
from caption_checker.web.storage import Storage

DATA_DIR = Path(__file__).parent / "data"
SAMPLE = DATA_DIR / "sample_lecture.srt"
SAMPLE_VTT = DATA_DIR / "sample_lecture.vtt"


def _storage_for(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "data")


def _make_client(tmp_path: Path, reader: Reader | None = None) -> TestClient:
    app = create_app(_storage_for(tmp_path), reader=reader)
    return TestClient(app)


def _upload(client: TestClient, filename: str = "sample_lecture.srt", path: Path = SAMPLE) -> str:
    with path.open("rb") as f:
        response = client.post(
            "/transcripts", files={"file": (filename, f, "text/plain")}, follow_redirects=False
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
