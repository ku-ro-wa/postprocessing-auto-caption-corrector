# Public deployment: one Fly.io machine, keys stay in the browser, Transcripts last a day

The web UI goes public, for strangers who find it rather than people we
know. That turns three things ADRs 0003 and 0004 left as local-use defaults
into rules a public server has to keep, and settles where it runs.

**Decision**:

- **A visitor's own OpenRouter key never reaches the server's disk.** The
  browser keeps it (local storage, with a "Forget my key" button) and sends it
  with each Correct; the server uses it for that run and drops it. Server-side
  key storage (`session.json`) is removed outright, local `serve` included, so
  there is one key path and the public site can't quietly fall back to a
  stored key. Partly reverses ADR 0004, which kept the key on the Session.
- **Transcripts are deleted 24 hours after their last activity** -- upload,
  run, Review Decision, Cue edit or Export each reset the clock -- or at once
  from a Delete button offered on the export page. Narrows ADR 0003's
  "persisted": state still survives restarts, but only for a day. The spend
  ledger (ADR 0008) is a separate file and is not touched, so deleting a
  Transcript never resets an Allowance. The upload page says how long a file
  is kept, that Correct sends its text to OpenRouter's model provider, and
  that only English captions are supported.
- **Exactly one server process, on one Fly.io machine with a volume.** Storage
  is plain directories and the spend ledger is guarded by an in-process lock,
  so a second instance would corrupt both. The machine stays running (no
  auto-stop) because Correct runs inside its request. No backups: nothing
  lives longer than a day.
- **Correct can't be double-charged.** It stays synchronous for launch, but the
  button disables on click, the page warns it takes a minute or two, and a
  Transcript with a run already going refuses another. A background job with
  progress is future work.
- **Hardening.** The Session cookie is `Secure` (lifetime unchanged; it only
  ties a browser to its Allowance), uploads are capped at 2 MB.
- **Launch figures for ADR 0008.** Allowance 10,000 words per Session per
  day, Daily budget $0.25, and a dedicated OpenRouter key with a hard $5 a
  month credit limit as the backstop -- inside a $5-10 a month total budget.
- **Counting without trackers.** The server tallies first visits (with the
  link's `?ref=` tag), uploads, Correct runs, Free tier refusals and Exports
  itself; no third-party analytics. Feedback goes to a dedicated email
  address.

## Considered Options

- **Encrypt stored keys with a server secret**: protects the disk and
  backups, not against whoever runs the server, and still asks a stranger to
  trust us with a key. "We never store your key" is simpler to say and true.
- **Free tier only, no own keys**: a visitor whose Allowance runs out would
  have nowhere to go.
- **Delete immediately, or a fixed 24 hours after upload**: review spans
  several requests, and a fixed deadline loses a slow reviewer's Review
  Decisions and the Read-through they paid for.
- **Keep Transcripts 7 or 30 days**: kinder to slow reviewers, but more
  strangers' captions sitting on disk than the tool needs.
- **Railway or a plain VPS**: Railway is a fine fallback; a VPS is cheapest
  but means running the OS, TLS and restarts ourselves.
- **Background-job Correct now**: the better experience, but job state,
  polling and surviving restarts is real work before we know anyone uses it.
