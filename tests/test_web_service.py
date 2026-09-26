from __future__ import annotations

from pathlib import Path

import pytest

from caption_checker.corrector import Correction, CorrectorError
from caption_checker.readthrough import ChunkRequest, ChunkVerdict, StubReader
from caption_checker.web import service
from caption_checker.web.storage import Storage

DATA_DIR = Path(__file__).parent / "data"


class _AssertNotCalledReader(StubReader):
    """A ``Reader`` that fails the test if it's ever invoked."""

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
        raise AssertionError("should not call the reader again once already corrected")


class _FailingOn(StubReader):
    """A ``StubReader`` whose every request for a chunk containing ``word``
    fails, as a malformed reply would."""

    def __init__(self, word: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.word = word

    def read(self, request: ChunkRequest) -> list[ChunkVerdict]:
        if any(text.strip(".,") == self.word for _, text in request.words):
            raise CorrectorError("stub: garbage reply")
        return super().read(request)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "data")


@pytest.fixture
def session_id(storage: Storage) -> str:
    return storage.create_session()


def _upload_sample(storage: Storage, session_id: str, filename: str = "sample_lecture.srt"):
    content = (DATA_DIR / filename).read_bytes()
    return service.upload_transcript(storage, session_id, filename, content)


class TestUploadTranscript:
    def test_scans_synchronously_with_cli_equivalent_defaults(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)

        assert record.flags, "expected the sample lecture to produce local flags"
        assert any("cubernetes" in f.span.lower() for f in record.flags)
        assert record.corrections == [None] * len(record.flags)
        assert all(d.status == "pending" for d in record.decisions)

    def test_persists_and_reloads(self, storage: Storage, session_id: str) -> None:
        record = _upload_sample(storage, session_id)

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert [f.span for f in reloaded.flags] == [f.span for f in record.flags]

    def test_rejects_unsupported_extension(self, storage: Storage, session_id: str) -> None:
        with pytest.raises(service.InvalidTranscriptError):
            service.upload_transcript(storage, session_id, "notes.txt", b"hello")

        assert storage.list_transcripts(session_id) == []

    def test_rejects_unparseable_srt_and_leaves_no_record(
        self, storage: Storage, session_id: str
    ) -> None:
        with pytest.raises(service.InvalidTranscriptError):
            service.upload_transcript(storage, session_id, "broken.srt", b"not an srt file at all")

        assert storage.list_transcripts(session_id) == []


def _long_transcript(sentences: int, *, marker_at: int) -> bytes:
    """An SRT long enough for several Read-through chunks (40-word
    sentences, one per Cue), with the non-word "zzqx" in Cue ``marker_at``."""
    blocks = []
    for i in range(sentences):
        words = ["the", "cat", "sat", "on", "the", "mat"] * 6 + ["and", "then", "it", "slept."]
        if i == marker_at:
            words[0] = "zzqx"
        blocks.append(
            f"{i + 1}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\n{' '.join(words)}\n"
        )
    return "\n".join(blocks).encode()


class TestRunCorrection:
    """The web `correct` pass is the Read-through (ADR 0006), over the
    ``Reader`` seam."""

    def test_hint_verdicts_align_with_their_flags(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        local = list(record.flags)
        stub = StubReader(replacement_for={"cubernetes": "Kubernetes"})

        result = service.run_correction(storage, record, api_key="test-key", reader=stub)

        assert result.flags == local  # no finds scripted -> nothing appended
        flag_id = next(i for i, f in enumerate(local) if f.span == "cubernetes")
        assert result.corrections[flag_id].replacement == "Kubernetes"
        assert all(c is not None for c in result.corrections)
        assert result.correct_error is None
        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None and reloaded.corrected

    def test_new_finds_are_appended_as_flags_with_pending_decisions(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        n_local = len(record.flags)
        stub = StubReader(extra={"leader election": "leader elections"})

        service.run_correction(storage, record, api_key="test-key", reader=stub)

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert len(reloaded.flags) == len(reloaded.corrections) == len(reloaded.decisions)
        assert len(reloaded.flags) == n_local + 1
        found = reloaded.flags[n_local]
        assert (found.span, found.detector) == ("leader election", "read_through")
        assert reloaded.corrections[n_local].replacement == "leader elections"
        assert reloaded.decisions[n_local].status == "pending"

    def test_a_new_find_exports_like_a_local_flag(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubReader(extra={"leader election": "leader elections"})
        service.run_correction(storage, record, api_key="test-key", reader=stub)

        service.set_decision(record, len(record.flags) - 1, action="accept", text=None)

        assert "leader elections using" in service.export_transcript(storage, record)

    def test_rows_list_new_finds_in_transcript_order(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubReader(extra={"leader election": "leader elections"})
        service.run_correction(storage, record, api_key="test-key", reader=stub)

        rows = service.transcript_rows(record)

        starts = [min(r.flag.global_indices) for r in rows]
        assert starts == sorted(starts)
        assert rows[[r.flag.span for r in rows].index("leader election")].id == len(record.flags) - 1

    def test_dismissed_hints_are_kept_and_marked_dismissed(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubReader(null_spans={"con sensus"})

        service.run_correction(storage, record, api_key="test-key", reader=stub)

        row = next(r for r in service.transcript_rows(record) if r.flag.span == "con sensus")
        assert row.dismissed
        assert row.decision.status == "pending"  # skipped on export unless overridden

    def test_widened_hint_replaces_its_flag_in_place(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.span == "cubernetes")
        service.set_decision(record, flag_id, action="accept", text="Kubernetes")
        stub = StubReader(
            widen={"cubernetes": "look at cubernetes"},
            replacement_for={"look at cubernetes": "look at Kubernetes"},
        )

        service.run_correction(storage, record, api_key="test-key", reader=stub)

        assert record.flags[flag_id].span == "look at cubernetes"
        assert record.corrections[flag_id].replacement == "look at Kubernetes"
        # the accepted text was for the narrower span: back to pending
        assert record.decisions[flag_id].status == "pending"

    def test_failed_chunks_are_counted_and_leave_their_flags_unjudged(
        self, storage: Storage, session_id: str
    ) -> None:
        content = _long_transcript(30, marker_at=25)
        record = service.upload_transcript(storage, session_id, "long.srt", content)
        marker = next(i for i, f in enumerate(record.flags) if f.span == "zzqx")

        service.run_correction(
            storage, record, api_key="test-key", reader=_FailingOn("zzqx")
        )

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert reloaded.corrected
        assert reloaded.chunk_count > 1
        assert reloaded.failed_chunks == 1
        assert reloaded.corrections[marker] is None
        assert reloaded.correct_error is None

    def test_priming_terms_reach_the_reader(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubReader()

        service.run_correction(
            storage, record, api_key="test-key", reader=stub, priming_terms=["Kafka", "Raft"]
        )

        assert stub.requests[0].priming_terms == ["Kafka", "Raft"]

    def test_runs_on_a_transcript_with_no_local_flags(
        self, storage: Storage, session_id: str
    ) -> None:
        content = b"1\n00:00:00,000 --> 00:00:02,000\nThe whether is fine today.\n"
        record = service.upload_transcript(storage, session_id, "clean.srt", content)
        assert record.flags == []
        stub = StubReader(extra={"whether": "weather"})

        service.run_correction(storage, record, api_key="test-key", reader=stub)

        assert [f.span for f in record.flags] == ["whether"]
        assert record.corrected

    def test_noop_when_already_corrected(self, storage: Storage, session_id: str) -> None:
        """Re-running `correct` on an already-corrected Transcript is out of
        scope — only a failed run is retriable."""
        record = _upload_sample(storage, session_id)
        service.run_correction(storage, record, api_key="test-key", reader=StubReader())
        first_corrections = list(record.corrections)

        result = service.run_correction(
            storage, record, api_key="test-key", reader=_AssertNotCalledReader()
        )

        assert result.corrections == first_corrections

    def test_every_chunk_failing_sets_retriable_error_and_reraises(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)

        with pytest.raises(CorrectorError):
            service.run_correction(
                storage, record, api_key="test-key", reader=StubReader(garbage=True)
            )

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert reloaded.correct_error is not None
        assert not reloaded.corrected

    def test_noop_correction_downgraded_to_dismissed(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.span == "cubernetes")
        # The stub echoes the flag's own span back as its "correction" --
        # exactly the same-text no-op the web layer needs to downgrade.
        stub = StubReader(replacement_for={"cubernetes": "cubernetes"})

        result = service.run_correction(storage, record, api_key="test-key", reader=stub)

        assert result.corrections[flag_id] is not None
        assert result.corrections[flag_id].replacement is None


class TestSetDecision:
    def test_accept_defaults_to_llm_correction_text(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = 0
        record.corrections[flag_id] = Correction(
            id=str(flag_id), replacement="Kubernetes", confidence=0.9
        )

        service.set_decision(record, flag_id, action="accept", text=None)

        assert record.decisions[flag_id].status == "accepted"
        assert record.decisions[flag_id].text == "Kubernetes"

    def test_accept_with_edit_overrides_correction_text(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = 0
        record.corrections[flag_id] = Correction(
            id=str(flag_id), replacement="Kubernetes", confidence=0.9
        )

        service.set_decision(record, flag_id, action="accept", text="Kubernetes cluster")

        assert record.decisions[flag_id].text == "Kubernetes cluster"

    def test_accept_without_correction_falls_back_to_top_local_candidate(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.candidates)

        service.set_decision(record, flag_id, action="accept", text=None)

        assert record.decisions[flag_id].text == record.flags[flag_id].candidates[0]

    def test_accept_without_correction_or_candidate_falls_back_to_span(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = 0
        record.flags[flag_id].candidates = []

        service.set_decision(record, flag_id, action="accept", text=None)

        assert record.decisions[flag_id].text == record.flags[flag_id].span

    def test_reject_clears_text(self, storage: Storage, session_id: str) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = 0

        service.set_decision(record, flag_id, action="accept", text="whatever")
        service.set_decision(record, flag_id, action="reject", text=None)

        assert record.decisions[flag_id].status == "rejected"
        assert record.decisions[flag_id].text is None

    def test_unknown_flag_id_raises(self, storage: Storage, session_id: str) -> None:
        record = _upload_sample(storage, session_id)
        with pytest.raises(IndexError):
            service.set_decision(record, len(record.flags) + 5, action="accept", text=None)

    def test_unknown_action_raises(self, storage: Storage, session_id: str) -> None:
        record = _upload_sample(storage, session_id)
        with pytest.raises(ValueError):
            service.set_decision(record, 0, action="frobnicate", text=None)


class TestTranscriptRows:
    def test_default_text_uses_top_local_candidate_when_no_correction(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.candidates)

        rows = service.transcript_rows(record)

        assert rows[flag_id].default_text == record.flags[flag_id].candidates[0]

    def test_default_text_prefers_llm_correction_over_local_candidate(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(i for i, f in enumerate(record.flags) if f.candidates)
        record.corrections[flag_id] = Correction(
            id=str(flag_id), replacement="LLM Replacement", confidence=0.9
        )

        rows = service.transcript_rows(record)

        assert rows[flag_id].default_text == "LLM Replacement"

    def test_default_text_falls_back_to_span_with_no_candidate_or_correction(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = 0
        record.flags[flag_id].candidates = []

        rows = service.transcript_rows(record)

        assert rows[flag_id].default_text == record.flags[flag_id].span


class TestExportTranscript:
    def test_only_accepted_decisions_change_text(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag_id = next(
            i for i, f in enumerate(record.flags) if "cubernetes" in f.span.lower()
        )
        service.set_decision(record, flag_id, action="accept", text="Kubernetes")

        exported = service.export_transcript(storage, record)

        assert "Kubernetes" in exported
        assert "cubernetes" not in exported.lower().replace("kubernetes", "")

    def test_pending_and_rejected_flags_keep_original_text(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        original = storage.load_cues(session_id, record.id)

        if record.flags:
            service.set_decision(record, 0, action="reject", text=None)

        exported = service.export_transcript(storage, record)

        # Nothing accepted -> export is byte-identical to a straight reparse+reserialize.
        from caption_checker.parser import serialize

        assert exported == serialize(original, format=record.format)
