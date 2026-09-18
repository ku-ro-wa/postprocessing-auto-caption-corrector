# Web review UI: upload-based, session-scoped, persisted, server-rendered

The CLI's `check`/`correct` output only reaches you as text or JSON in a
terminal. We're adding a locally-hosted web UI for reviewing Flags,
confirming/editing Corrections, and producing an Export — usable today as a
single-user local tool, but built so a later "point a public URL at it"
deployment doesn't require an architecture rewrite.

Decided: FastAPI + Jinja2 templates (htmx for interactivity, no JS build
step), matching the project's existing minimal Python-only footprint.
Transcripts arrive via browser upload rather than server-side filesystem
browsing, since a public deployment can't assume the browser and the server
share a filesystem. Review state (Flags, Corrections, Review Decisions)
persists to a sidecar JSON file per Transcript rather than living only in
memory, so restarting the server doesn't lose work. Each browser gets an
anonymous Session cookie scoping its own Transcripts and decisions — no
login, no accounts, just enough isolation that a shared URL doesn't mean
shared/overwritten data.

## Considered Options

- **JS-framework frontend (React/Vite)**: more capable for a rich editor,
  but adds a full separate toolchain to a project that's deliberately
  stayed pure Python. Revisit only if Jinja2/htmx can't support the
  accept/reject/edit interactions.
- **Filesystem-path file selection**: simpler today (no upload handling),
  but assumes the server and the browser share a disk — breaks the moment
  this is hosted anywhere but localhost.
- **In-memory-only review state**: simplest, but loses all review progress
  on every server restart/redeploy, which matters a lot more once this
  isn't just a process on your own laptop.
