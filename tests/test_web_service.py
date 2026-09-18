from __future__ import annotations

from pathlib import Path

import pytest

from caption_checker.corrector import Correction, CorrectorError, FlagContext, StubCorrector
from caption_checker.web import service
from caption_checker.web.storage import Storage

DATA_DIR = Path(__file__).parent / "data"


class _AssertNotCalledCorrector:
    """A ``Corrector`` that fails the test if it's ever invoked."""

    def correct(self, batch: list[FlagContext]) -> list[Correction]:
        raise AssertionError("should not call the corrector again once already corrected")


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


class TestRunCorrection:
    def test_populates_corrections_aligned_with_flags(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubCorrector(replacement_for={f.span: "fixed" for f in record.flags})

        result = service.run_correction(storage, record, api_key="test-key", corrector=stub)

        assert len(result.corrections) == len(result.flags)
        assert all(c is not None and c.replacement == "fixed" for c in result.corrections)
        assert result.correct_error is None

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert reloaded.has_corrections

    def test_noop_when_already_corrected(self, storage: Storage, session_id: str) -> None:
        """Re-running `correct` on an already-corrected Transcript is out of
        scope — only a failed run is retriable."""
        record = _upload_sample(storage, session_id)
        stub = StubCorrector(replacement_for={f.span: "fixed" for f in record.flags})
        service.run_correction(storage, record, api_key="test-key", corrector=stub)
        first_corrections = list(record.corrections)

        result = service.run_correction(
            storage, record, api_key="test-key", corrector=_AssertNotCalledCorrector()
        )

        assert result.corrections == first_corrections

    def test_failure_sets_retriable_error_and_reraises(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        stub = StubCorrector(garbage_spans={f.span for f in record.flags})

        with pytest.raises(CorrectorError):
            service.run_correction(storage, record, api_key="test-key", corrector=stub)

        reloaded = storage.load_transcript(session_id, record.id)
        assert reloaded is not None
        assert reloaded.correct_error is not None
        assert not reloaded.has_corrections

    def test_noop_correction_downgraded_to_dismissed(
        self, storage: Storage, session_id: str
    ) -> None:
        record = _upload_sample(storage, session_id)
        flag = next(f for f in record.flags if "cubernetes" in f.span.lower())
        flag_id = record.flags.index(flag)
        # The stub echoes the flag's own span back as its "correction" --
        # exactly the same-text no-op the web layer needs to downgrade.
        stub = StubCorrector(replacement_for={flag.span: flag.span})

        result = service.run_correction(storage, record, api_key="test-key", corrector=stub)

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
