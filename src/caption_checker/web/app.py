"""FastAPI app: the `caption-checker serve` web review UI.

Routes are a thin HTTP layer over ``caption_checker.web.service`` — no
business logic lives here beyond request/response plumbing and template
rendering. Session-cookie handling lives in ``_SessionCookieMiddleware``
below rather than being repeated per route.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from caption_checker.corrector import CorrectorError, MissingAPIKeyError
from caption_checker.readthrough import Reader, select_config
from caption_checker.web import service
from caption_checker.web import source_video
from caption_checker.web.free_tier import WINDOW, FreeTier, LimitReached, Limits
from caption_checker.web.models import TranscriptRecord
from caption_checker.web.storage import Storage
from caption_checker.web.usage import UsageLog

#: The product's public name (ADR 0010); a setting, not a constant of the pages.
DEFAULT_APP_NAME = "Misheard"

COOKIE_NAME = "cc_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 180  # 180 days

#: Largest upload accepted (ADR 0010). A three-hour SRT is ~200 KB.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

TEMPLATES_DIR = Path(__file__).parent / "templates"

#: How long a Transcript is kept after its last activity (ADR 0010).
RETENTION = timedelta(hours=24)
SWEEP_INTERVAL = timedelta(hours=1)

logger = logging.getLogger(__name__)


def _fmt_ts(seconds: float, round_up: bool = False) -> str:
    """A moment in the video as a player shows it: "0:03", "1:02:03".
    A start is rounded down and an end (``round_up``) up, so the shown range
    always covers the span and a sub-second one never reads "0:03 → 0:03"."""
    whole = math.ceil(seconds) if round_up else int(seconds)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_uploaded(iso: str) -> str:
    """A stored UTC timestamp as the page states it: "4 Oct 2026, 09:05 UTC"."""
    at = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return f"{at.day} {at:%b %Y, %H:%M} UTC"


def _duration(span: timedelta) -> str:
    """A wait as the page states it, rounded up to the minute so it's never
    early: "5 h 12 min", "19 h", "40 min"."""
    minutes = max(1, math.ceil(span.total_seconds() / 60))
    h, m = divmod(minutes, 60)
    if not h:
        return f"{m} min"
    return f"{h} h {m} min" if m else f"{h} h"


def _highlight(context: str, span: str) -> Markup:
    """Escape a Flag's sentence context and span, then wrap the span in
    <mark> — escaping first so caption text can't inject markup."""
    escaped_ctx = str(escape(context or span))
    escaped_span = str(escape(span))
    if escaped_span and escaped_span in escaped_ctx:
        escaped_ctx = escaped_ctx.replace(escaped_span, f"<mark>{escaped_span}</mark>", 1)
    return Markup(escaped_ctx)


HEALTH_PATH = "/healthz"


class _SessionCookieMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a Session: reads ``cc_session`` off the
    incoming cookie, creating a new anonymous Session when it's missing or
    stale, and stashes the id on ``request.state.session_id`` for routes to
    read. Issues the cookie only when a Session was just created, so
    routes never touch cookie plumbing themselves."""

    def __init__(self, app: FastAPI, *, storage: Storage, secure: bool = False) -> None:
        super().__init__(app)
        self.storage = storage
        self.secure = secure

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path == HEALTH_PATH:
            # A monitor polls this; it must not mint a Session per check.
            return await call_next(request)
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
                secure=self.secure,
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
    limits: Limits | None = Limits(),
    retention: timedelta = RETENTION,
    secure_cookie: bool = False,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    app_name: str = DEFAULT_APP_NAME,
    feedback_email: str | None = None,
) -> FastAPI:
    """``reader`` lets tests inject a ``StubReader`` (or any other
    ``Reader``) at the same seam the CLI's Read-through tests use — no route
    in this app talks to OpenRouter directly. ``video_lookup`` likewise
    stands in for the Source video's YouTube oEmbed lookup.

    Without one, the Read-through configuration comes from the environment
    (``service.config_from_env``), resolved here so a bad one stops startup
    instead of failing the first correct.

    ``limits`` meters a Correct sent without the visitor's own key, which
    runs on the server's key, as the Free tier (ADR 0008): on unless
    explicitly turned off with None, for local use. Its ledger lives at
    ``storage.root``. A Free tier run needs a priced model, also checked
    here when the server has a key to run one on.

    ``secure_cookie`` marks the Session cookie ``Secure`` (for a server
    reached over HTTPS; plain-HTTP local use leaves it off), and uploads
    over ``max_upload_bytes`` are refused before parsing (ADR 0010).

    ``app_name`` heads every page; the footer links ``feedback_email`` as a
    mailto, and has no feedback link while it is None. Each upload, Correct
    run and Export is tallied to ``usage.log`` at ``storage.root`` (ADR 0010).

    While the app runs, Transcripts idle for ``retention`` are swept at
    startup and every ``SWEEP_INTERVAL`` after (ADR 0010). An emptied
    Session is kept for at least the Allowance's window, so its visitor
    can't get a fresh Allowance by waiting out a deletion."""
    config = None if reader is not None else service.config_from_env()
    free_tier = FreeTier(storage.root, limits) if limits is not None else None
    # What a Free tier run is estimated (and, without a ``reader``, run) with.
    free_tier_config = config or select_config()
    if free_tier is not None and os.environ.get("OPENROUTER_API_KEY"):
        service.check_free_tier_config(free_tier_config)

    def sweep() -> None:
        swept = storage.sweep(retention, keep_empty_sessions_for=max(retention, WINDOW))
        if swept:
            logger.info("Deleted %d expired transcript(s)", swept)

    async def sweep_logging_failure() -> None:
        try:
            await run_in_threadpool(sweep)
        except Exception:
            logger.exception("Transcript sweep failed")

    async def sweep_periodically() -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL.total_seconds())
            await sweep_logging_failure()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await sweep_logging_failure()
        task = asyncio.create_task(sweep_periodically())
        try:
            yield
        finally:
            task.cancel()

    usage = UsageLog(storage.root)

    app = FastAPI(title=app_name, lifespan=lifespan)
    app.add_middleware(_SessionCookieMiddleware, storage=storage, secure=secure_cookie)

    @app.api_route(HEALTH_PATH, methods=["GET", "HEAD"], include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["ts"] = lambda td, round_up=False: _fmt_ts(td.total_seconds(), round_up)
    templates.env.filters["highlight"] = _highlight
    templates.env.filters["thousands"] = lambda n: f"{n:,}"
    templates.env.filters["duration"] = _duration
    templates.env.filters["uploaded"] = _fmt_uploaded
    templates.env.globals["source_video"] = source_video
    templates.env.globals["retention"] = retention
    templates.env.globals["app_name"] = app_name
    templates.env.globals["feedback_email"] = feedback_email or None

    # Transcripts with a Correct running, so a refresh or a second click
    # can't start (and charge for) another. One server process (ADR 0010).
    running: set[tuple[str, str]] = set()
    running_lock = threading.Lock()
    # Exports already tallied, by Transcript and file content. Memory only: a
    # restart forgets them and at worst counts one download twice.
    exported: set[tuple[str, int]] = set()

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
        content = await file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "transcripts": storage.list_transcripts(session_id),
                    "upload_error": "That file is larger than the "
                    f"{max_upload_bytes / 1024 / 1024:g} MB limit.",
                    "video_link": video_link,
                },
                status_code=413,
            )
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

        usage.record("upload")
        return RedirectResponse(f"/transcripts/{record.id}", status_code=303)

    def review_page(
        request: Request, record: TranscriptRecord, *, status_code: int = 200, **extra: object
    ) -> Response:
        has_server_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        words_left = next_return = None
        if free_tier is not None and has_server_key:
            words_left = free_tier.words_left(record.session_id)
            next_return = free_tier.next_return(record.session_id)
        return templates.TemplateResponse(
            request,
            "transcript.html",
            {
                "transcript": record,
                "rows": service.transcript_rows(storage, record),
                "cues": service.cue_rows(storage, record),
                "summary": service.correction_summary(record),
                "has_server_key": has_server_key,
                "word_count": service.transcript_word_count(storage, record),
                "words_left": words_left,
                "next_return": next_return,
                "limits": limits,
                "offered_priming_terms": service.offered_priming_terms(record),
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
        key = (session_id, transcript_id)
        # The guard comes before the load, so a run that finishes in between
        # can't leave this one working from a stale, uncorrected record.
        with running_lock:
            already_running = key in running
            running.add(key)
        if already_running:
            return review_page(
                request, load_or_404(session_id, transcript_id), status_code=409, already_running=True
            )
        try:
            return run_correct(
                request, load_or_404(session_id, transcript_id), api_key, priming_terms
            )
        finally:
            with running_lock:
                running.discard(key)

    def run_correct(
        request: Request, record: TranscriptRecord, api_key: str, priming_terms: str
    ) -> Response:
        transcript_id = record.id
        # Saved with the run's outcome below, so a failed run shows them again.
        record.priming_terms = priming_terms.strip()

        # The visitor's own key is used for this run and never stored, nor
        # metered. Without one the run falls to the server's key -- the Free
        # tier, unless the limits are off -- and a run on the visitor's key
        # that fails never retries on the server's (ADR 0008, ADR 0010).
        own_key = api_key.strip()
        server_key = os.environ.get("OPENROUTER_API_KEY")
        terms = _priming_terms(priming_terms)
        run_key = own_key or server_key
        if not run_key:
            record.correct_error = (
                "No OpenRouter API key available. This server has no Free "
                "tier, so enter your own key below."
            )
            storage.save_transcript(record)
        else:
            try:
                if own_key or free_tier is None:
                    service.run_correction(
                        storage,
                        record,
                        api_key=run_key,
                        config=config,
                        reader=reader,
                        priming_terms=terms,
                    )
                else:
                    service.run_on_free_tier(
                        storage,
                        record,
                        free_tier,
                        api_key=run_key,
                        config=free_tier_config,
                        reader=reader,
                        priming_terms=terms,
                    )
            except LimitReached as exc:
                return review_page(
                    request, record, status_code=429, refusal=exc.limit, refusal_wait=exc.wait
                )
            except (MissingAPIKeyError, CorrectorError):
                pass  # record.correct_error already set by run_correction
            else:
                usage.record("correct")  # only a run that finished

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
        # A repeat download of the same file isn't another Export; one after
        # a change to the Transcript is.
        counted = (transcript_id, hash(content))
        if counted not in exported:
            exported.add(counted)
            usage.record("export")
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
