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


def create_app(storage: Storage, *, reader: Reader | None = None) -> FastAPI:
    """``reader`` lets tests inject a ``StubReader`` (or any other
    ``Reader``) at the same seam the CLI's Read-through tests use — no route
    in this app talks to OpenRouter directly."""
    app = FastAPI(title="caption-checker")
    app.add_middleware(_SessionCookieMiddleware, storage=storage)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["ts"] = lambda td: _fmt_ts(td.total_seconds())
    templates.env.filters["highlight"] = _highlight

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
    async def upload(request: Request, file: UploadFile) -> Response:
        session_id = request.state.session_id
        content = await file.read()
        try:
            record = service.upload_transcript(
                storage, session_id, file.filename or "upload", content
            )
        except service.InvalidTranscriptError as exc:
            transcripts = storage.list_transcripts(session_id)
            return templates.TemplateResponse(
                request,
                "index.html",
                {"transcripts": transcripts, "upload_error": str(exc)},
                status_code=400,
            )

        return RedirectResponse(f"/transcripts/{record.id}", status_code=303)

    @app.get("/transcripts/{transcript_id}", response_class=HTMLResponse)
    def show(request: Request, transcript_id: str) -> Response:
        session_id = request.state.session_id
        record = load_or_404(session_id, transcript_id)
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "transcript": record,
                "rows": service.transcript_rows(record),
                "summary": service.correction_summary(record),
                "has_session_key": bool(storage.get_session_api_key(session_id)),
                "has_server_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            },
        )

    @app.post("/transcripts/{transcript_id}/correct")
    def correct(
        request: Request,
        transcript_id: str,
        api_key: str = Form(""),
        priming_terms: str = Form(""),
    ) -> Response:
        session_id = request.state.session_id
        record = load_or_404(session_id, transcript_id)

        api_key = api_key.strip()
        if api_key:
            storage.set_session_api_key(session_id, api_key)

        resolved_key = (
            storage.get_session_api_key(session_id) or os.environ.get("OPENROUTER_API_KEY")
        )
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
        row = next(r for r in service.transcript_rows(record) if r.id == flag_id)
        return templates.TemplateResponse(
            request, "partials/flag_row.html", {"transcript": record, "row": row}
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
