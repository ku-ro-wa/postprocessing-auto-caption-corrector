"""#65: the Example -- a Transcript run through Correct once, offline, saved
with the app, and copied into a visitor's Session on request."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from caption_checker.readthrough import StubReader
from caption_checker.web import example
from caption_checker.web.app import create_app
from caption_checker.web.models import TranscriptRecord
from caption_checker.web.source_video import VideoMetadata
from caption_checker.web.storage import Storage
from caption_checker.web.usage import USAGE_FILENAME

from fake_clock import FakeClock

SAMPLE = Path(__file__).parent / "data" / "sample_lecture.srt"
START = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def _metadata(video_id: str) -> VideoMetadata | None:
    return VideoMetadata(title="A lecture", channel="A channel")


def _client(storage: Storage, **kwargs: Any) -> TestClient:
    return TestClient(create_app(storage, reader=StubReader(), video_lookup=_metadata, **kwargs))


def _upload(client: TestClient) -> str:
    with SAMPLE.open("rb") as f:
        response = client.post(
            "/transcripts",
            files={"file": ("sample_lecture.srt", f, "text/plain")},
            data={"video_link": "https://youtu.be/dQw4w9WgXcQ"},
            follow_redirects=False,
        )
    assert response.status_code == 303
    return response.headers["location"].rsplit("/", 1)[-1]


def _corrected_and_reviewed(storage: Storage) -> tuple[str, TranscriptRecord]:
    """A Transcript through Correct, with one Flag rejected and one Cue
    edited by the reviewer -- as the operator leaves it after checking the
    run."""
    client = _client(storage, example_dir=None)
    transcript_id = _upload(client)
    client.post(f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-test"})
    client.post(f"/transcripts/{transcript_id}/flags/0/decision", data={"action": "reject"})
    client.post(f"/transcripts/{transcript_id}/cues/1/edit", data={"text": "Edited by the reviewer"})
    record = storage.load_transcript(client.cookies["cc_session"], transcript_id)
    assert record is not None and record.corrected and record.added_count == 1
    return transcript_id, record


@pytest.fixture
def example_dir(tmp_path: Path) -> Path:
    source = Storage(tmp_path / "operator")
    transcript_id, _ = _corrected_and_reviewed(source)
    dest = tmp_path / "example"
    example.capture(source, transcript_id, dest)
    return dest


class TestCapture:
    def test_saves_the_original_and_the_run_with_review_reset(self, tmp_path: Path) -> None:
        storage = Storage(tmp_path / "operator")
        transcript_id, before = _corrected_and_reviewed(storage)
        dest = tmp_path / "example"

        saved = example.capture(storage, transcript_id, dest)

        assert (dest / "original.srt").read_bytes() == SAMPLE.read_bytes()
        assert saved.added_count == 0  # the reviewer's own Flags are dropped
        reviewer_raised = len(before.flags) - 1
        assert saved.flags == before.flags[:reviewer_raised]
        assert saved.corrections == before.corrections[:reviewer_raised]
        assert all(d.status == "pending" for d in saved.decisions)
        assert len(saved.decisions) == len(saved.flags)
        assert saved.corrected_at == before.corrected_at
        assert saved.video_id == before.video_id
        assert saved.session_id == ""
        assert example.load(dest) == saved

    def test_replaces_an_earlier_capture(self, tmp_path: Path) -> None:
        storage = Storage(tmp_path / "operator")
        transcript_id, _ = _corrected_and_reviewed(storage)
        dest = tmp_path / "example"
        dest.mkdir()
        (dest / "original.vtt").write_text("WEBVTT\n")

        example.capture(storage, transcript_id, dest)

        assert sorted(p.name for p in dest.iterdir()) == ["original.srt", "state.json"]

    def test_refuses_a_transcript_correct_has_not_run_on(self, tmp_path: Path) -> None:
        storage = Storage(tmp_path / "operator")
        transcript_id = _upload(_client(storage, example_dir=None))

        with pytest.raises(example.ExampleError, match="Correct"):
            example.capture(storage, transcript_id, tmp_path / "example")

    def test_refuses_an_unknown_transcript(self, tmp_path: Path) -> None:
        with pytest.raises(example.ExampleError, match="No transcript"):
            example.capture(Storage(tmp_path / "operator"), "nope", tmp_path / "example")


class TestCaptureCommand:
    def test_captures_from_the_data_dir_into_the_given_dir(self, tmp_path: Path) -> None:
        from caption_checker.cli import main

        storage = Storage(tmp_path / "operator")
        transcript_id, _ = _corrected_and_reviewed(storage)
        dest = tmp_path / "example"

        result = CliRunner().invoke(
            main,
            ["capture-example", transcript_id, "--data-dir", str(storage.root), "--to", str(dest)],
        )

        assert result.exit_code == 0, result.output
        assert example.load(dest) is not None
        assert str(dest) in result.output

    def test_an_uncorrected_transcript_is_an_error(self, tmp_path: Path) -> None:
        from caption_checker.cli import main

        storage = Storage(tmp_path / "operator")
        transcript_id = _upload(_client(storage, example_dir=None))

        result = CliRunner().invoke(
            main,
            ["capture-example", transcript_id, "--data-dir", str(storage.root),
             "--to", str(tmp_path / "example")],
        )

        assert result.exit_code != 0
        assert "Correct" in result.output


class TestCopyInto:
    def test_copies_the_example_into_the_session_as_new_activity(
        self, tmp_path: Path, example_dir: Path
    ) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path / "data", clock=clock)
        session_id = storage.create_session()

        record = example.copy_into(storage, session_id, example_dir)

        assert record.is_example
        assert record.session_id == session_id
        assert record.created_at == record.last_activity == START.isoformat()
        assert storage.load_transcript(session_id, record.id) == record
        assert storage.load_cues(session_id, record.id)

    def test_every_copy_is_its_own_transcript(self, tmp_path: Path, example_dir: Path) -> None:
        storage = Storage(tmp_path / "data")
        session_id = storage.create_session()

        first = example.copy_into(storage, session_id, example_dir)
        second = example.copy_into(storage, session_id, example_dir)

        assert first.id != second.id

    def test_is_available_only_with_a_saved_example(
        self, tmp_path: Path, example_dir: Path
    ) -> None:
        assert example.is_available(example_dir)
        assert not example.is_available(tmp_path / "missing")
        assert not example.is_available(None)


class TestExampleRoutes:
    def _client(self, tmp_path: Path, example_dir: Path | None) -> TestClient:
        return _client(Storage(tmp_path / "data"), example_dir=example_dir)

    def _open(self, client: TestClient) -> str:
        response = client.post("/example", follow_redirects=False)
        assert response.status_code == 303
        return response.headers["location"].rsplit("/", 1)[-1]

    def test_the_landing_page_offers_it_only_when_one_is_saved(
        self, tmp_path: Path, example_dir: Path
    ) -> None:
        assert 'action="/example"' in self._client(tmp_path, example_dir).get("/").text
        assert 'action="/example"' not in self._client(tmp_path, None).get("/").text
        assert 'action="/example"' not in self._client(tmp_path, tmp_path / "none").get("/").text

    def test_without_one_the_route_is_not_found(self, tmp_path: Path) -> None:
        assert self._client(tmp_path, None).post("/example").status_code == 404

    def test_opens_a_corrected_copy_marked_as_an_example(
        self, tmp_path: Path, example_dir: Path
    ) -> None:
        client = self._client(tmp_path, example_dir)
        transcript_id = self._open(client)

        page = client.get(f"/transcripts/{transcript_id}").text
        assert "This is a saved example" in page
        assert "AI read-through done" in page
        assert 'id="correct-form"' not in page
        assert "(example)" in client.get("/").text.split("Your transcripts")[-1]

    def test_an_uploaded_transcript_is_not_marked(self, tmp_path: Path, example_dir: Path) -> None:
        client = self._client(tmp_path, example_dir)
        page = client.get(f"/transcripts/{_upload(client)}").text
        assert "This is a saved example" not in page

    def test_each_session_reviews_its_own_copy(self, tmp_path: Path, example_dir: Path) -> None:
        storage = Storage(tmp_path / "data")
        alice = _client(storage, example_dir=example_dir)
        bob = _client(storage, example_dir=example_dir)
        alices = self._open(alice)
        bobs = self._open(bob)
        assert alices != bobs

        alice.post(f"/transcripts/{alices}/flags/0/decision", data={"action": "reject"})

        bob_record = storage.load_transcript(bob.cookies["cc_session"], bobs)
        assert bob_record is not None
        assert bob_record.decisions[0].status == "pending"
        assert bob.get(f"/transcripts/{alices}").status_code == 404

    def test_correct_is_refused_on_a_copy(self, tmp_path: Path, example_dir: Path) -> None:
        client = self._client(tmp_path, example_dir)
        transcript_id = self._open(client)

        response = client.post(
            f"/transcripts/{transcript_id}/correct", data={"api_key": "sk-or-test"}
        )

        assert response.status_code == 409
        events = (tmp_path / "data" / USAGE_FILENAME).read_text()
        assert "correct" not in events

    def test_opening_one_is_tallied_as_an_example(
        self, tmp_path: Path, example_dir: Path
    ) -> None:
        client = self._client(tmp_path, example_dir)
        self._open(client)
        lines = (tmp_path / "data" / USAGE_FILENAME).read_text().splitlines()
        assert [line.split(" ", 1)[1] for line in lines] == ["example"]

    def test_the_retention_sweep_deletes_copies(self, tmp_path: Path, example_dir: Path) -> None:
        clock = FakeClock(START)
        storage = Storage(tmp_path / "data", clock=clock)
        client = _client(storage, example_dir=example_dir)
        transcript_id = self._open(client)

        clock.now += timedelta(hours=25)
        storage.sweep(timedelta(hours=24), keep_empty_sessions_for=timedelta(hours=24))

        assert storage.load_transcript(client.cookies["cc_session"], transcript_id) is None
        assert example.is_available(example_dir)
