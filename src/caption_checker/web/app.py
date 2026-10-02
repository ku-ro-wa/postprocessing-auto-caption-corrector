"""FastAPI app: the `caption-checker serve` web review UI.

Routes are a thin HTTP layer over ``caption_checker.web.service`` — no
business logic lives here beyond request/response plumbing and template
rendering. Session-cookie handling lives in ``_SessionCookieMiddleware``
below rather than being repeated per route.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from caption_checker.corrector import CorrectorError, MissingAPIKeyError
from caption_checker.readthrough import Reader
from caption_checker.web import service
from caption_checker.web import source_video
from caption_checker.web.models import TranscriptRecord
from caption_checker.web.storage import Storage

COOKIE_NAME = "cc_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 180  # 180 days

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _fmt_ts(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _highlight(context: str, span: str) -> Markup:
    """Escape a Flag's sentence context and span, then wrap the span in
    <mark> — escaping first so caption text can't inject markup."""
    escaped_ctx = str(escape(context or span))
    escaped_span = str(escape(span))
    if escaped_span and escaped_span in escaped_ctx:
        escaped_ctx = escaped_ctx.replace(escaped_span, f"<mark>{escaped_span}</mark>", 1)
    return Markup(escaped_ctx)


class _SessionCookieMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a Session: reads ``cc_session`` off the
    incoming cookie, creating a new anonymous Session when it's missing or
    stale, and stashes the id on ``request.state.session_id`` for routes to
    read. Issues the cookie only when a Session was just created, so
    routes never touch cookie plumbing themselves."""

    def __init__(self, app: FastAPI, *, storage: Storage) -> None:
        super().__init__(app)
        self.storage = storage

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        session_id = request.cookies.get(COOKIE_NAME)
        is_new = not (session_id and self.storage.session_exists(session_id))
        if is_new:
            session_id = self.storage.create_session()
        request.state.session_id = session_id

        response = await call_next(request)

        if is_new:
            response.set_cookie(
                COOKIE_NAME,
                session_id,
                max_age=COOKIE_MAX_AGE,
                httponly=True,
                samesite="lax",
            )
        return response


def _priming_terms(raw: str) -> list[str]:
    """The Priming terms field: one term per line or comma-separated."""
    return [t.strip() for t in re.split(r"[,\n]", raw) if t.strip()]


def create_app(
    storage: Storage,
    *,
    reader: Reader | None = None,
    video_lookup: source_video.MetadataLookup = source_video.lookup_metadata,
) -> FastAPI:
    """``reader`` lets tests inject a ``StubReader`` (or any other
    ``Reader``) at the same seam the CLI's Read-through tests use — no route
    in this app talks to OpenRouter directly. ``video_lookup`` likewise
    stands in for the Source video's YouTube oEmbed lookup.

    Without one, the Read-through configuration comes from the environment
    (``service.config_from_env``), resolved here so a bad one stops startup
    instead of failing the first correct."""
    config = None if reader is not None else service.config_from_env()
    app = FastAPI(title="caption-checker")
    app.add_middleware(_SessionCookieMiddleware, storage=storage)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["ts"] = lambda td: _fmt_ts(td.total_seconds())
    templates.env.filters["highlight"] = _highlight
    templates.env.globals["source_video"] = source_video

    def load_or_404(session_id: str, transcript_id: str) -> TranscriptRecord:
        record = storage.load_transcript(session_id, transcript_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Transcript not found")
        return record

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        transcripts = storage.list_transcripts(request.state.session_id)
        return templates.TemplateResponse(request, "index.html", {"transcripts": transcripts})

    @app.post("/transcripts")
    async def upload(
        request: Request, file: UploadFile, video_link: str = Form("")
    ) -> Response:
        session_id = request.state.session_id
        content = await file.read()
        try:
            record = service.upload_transcript(
                storage,
                session_id,
                file.filename or "upload",
                content,
                video_link=video_link,
                video_lookup=video_lookup,
            )
        except (service.InvalidTranscriptError, service.InvalidVideoLinkError) as exc:
            transcripts = storage.list_transcripts(session_id)
            return templates.TemplateResponse(
                request,
                "index.html",
                {"transcripts": transcripts, "upload_error": str(exc), "video_link": video_link},
                status_code=400,
            )

        return RedirectResponse(f"/transcripts/{record.id}", status_code=303)

    def review_page(
        request: Request, record: TranscriptRecord, *, status_code: int = 200, **extra: object
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "transcript": record,
                "rows": service.transcript_rows(storage, record),
                "cues": service.cue_rows(storage, record),
                "summary": service.correction_summary(record),
                "has_server_key": bool(os.environ.get("OPENROUTER_API_KEY")),
                "suggested_priming_terms": service.suggested_priming_terms(record),
                **extra,
            },
            status_code=status_code,
        )

    @app.get("/transcripts/{transcript_id}", response_class=HTMLResponse)
    def show(request: Request, transcript_id: str) -> Response:
        return review_page(request, load_or_404(request.state.session_id, transcript_id))

    @app.post("/transcripts/{transcript_id}/video")
    def link_source_video(
        request: Request, transcript_id: str, video_link: str = Form("")
    ) -> Response:
        record = load_or_404(request.state.session_id, transcript_id)
        try:
            service.set_source_video(storage, record, video_link, video_lookup=video_lookup)
        except service.InvalidVideoLinkError as exc:
            return review_page(
                request, record, status_code=400, video_error=str(exc), video_link=video_link
            )
        return RedirectResponse(f"/transcripts/{transcript_id}", status_code=303)

    @app.post("/transcripts/{transcript_id}/video/delete")
    def remove_source_video(request: Request, transcript_id: str) -> Response:
        record = load_or_404(request.state.session_id, transcript_id)
        service.clear_source_video(storage, record)
        return RedirectResponse(f"/transcripts/{transcript_id}", status_code=303)

    @app.post("/transcripts/{transcript_id}/correct")
    def correct(
        request: Request,
        transcript_id: str,
        api_key: str = Form(""),
        priming_terms: str = Form(""),
    ) -> Response:
        session_id = request.state.session_id
        record = load_or_404(session_id, transcript_id)
        # Saved with the run's outcome below, so a failed run shows them again.
        record.priming_terms = priming_terms.strip()

        # The visitor's own key is used for this run and never stored. Without
        # one the run falls to the server's key; a run on the visitor's key
        # that fails never retries on the server's (ADR 0010).
        resolved_key = api_key.strip() or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            record.correct_error = (
                "No OpenRouter API key available. Enter one below, or set "
                "OPENROUTER_API_KEY in the server's .env to use it for local testing."
            )
            storage.save_transcript(record)
        else:
            try:
                service.run_correction(
                    storage,
                    record,
                    api_key=resolved_key,
                    config=config,
                    reader=reader,
                    priming_terms=_priming_terms(priming_terms),
                )
            except (MissingAPIKeyError, CorrectorError):
                pass  # record.correct_error already set by run_correction

        return RedirectResponse(f"/transcripts/{transcript_id}", status_code=303)

    @app.post("/transcripts/{transcript_id}/flags/{flag_id}/decision", response_class=HTMLResponse)
    def decide(
        request: Request,
        transcript_id: str,
        flag_id: int,
        action: str = Form(...),
        text: str = Form(""),
    ) -> Response:
        session_id = request.state.session_id
        record = load_or_404(session_id, transcript_id)

        try:
            service.set_decision(record, flag_id, action=action, text=text or None)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        storage.save_transcript(record)
        row = next(r for r in service.transcript_rows(storage, record) if r.id == flag_id)
        # The All Cues view's rows for the Cues this decision changes ride
        # along as out-of-band swaps.
        cues = service.cue_rows(
            storage, record, cue_indices=service.cues_affected(storage, record, flag_id)
        )
        return templates.TemplateResponse(
            request,
            "partials/decision.html",
            {"transcript": record, "row": row, "cues": cues, "oob": True},
        )

    @app.post("/transcripts/{transcript_id}/cues/{cue_index}/edit", response_class=HTMLResponse)
    def edit_cue(
        request: Request, transcript_id: str, cue_index: int, text: str = Form("")
    ) -> Response:
        record = load_or_404(request.state.session_id, transcript_id)

        extra: dict[str, object] = {}
        try:
            result = service.edit_cue(storage, record, cue_index, text)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except service.CueEditError as exc:
            # Nothing was recorded: the reviewer's text stays in the box.
            extra = {"edit_error": str(exc), "edit_text": text, "edit_open": True}
        else:
            storage.save_transcript(record)
            extra = {"edit_result": result, "edit_open": bool(result.unsaved)}
            if result.unsaved:
                extra["edit_text"] = text

        # The edited Cue swaps in place; the Flags list and counts, where a
        # new Flag belongs in transcript order, ride along out of band.
        cue = service.cue_rows(storage, record, cue_indices=[cue_index])[0]
        return templates.TemplateResponse(
            request,
            "partials/cue_edit.html",
            {
                "transcript": record,
                "cue": cue,
                "rows": service.transcript_rows(storage, record),
                "summary": service.correction_summary(record),
                "oob": True,
                **extra,
            },
        )

    @app.get("/transcripts/{transcript_id}/export")
    def export(request: Request, transcript_id: str) -> Response:
        record = load_or_404(request.state.session_id, transcript_id)
        content = service.export_transcript(storage, record)
        stem = Path(record.filename).stem
        filename = f"{stem}.corrected.{record.format}"
        return Response(
            content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/transcripts/{transcript_id}/delete")
    def delete(request: Request, transcript_id: str) -> Response:
        session_id = request.state.session_id
        load_or_404(session_id, transcript_id)
        storage.delete_transcript(session_id, transcript_id)
        return RedirectResponse("/", status_code=303)

    return app
