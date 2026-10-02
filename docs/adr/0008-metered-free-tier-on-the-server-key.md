---
status: proposed
---

# Meter the web UI's server key as a Free tier: a per-Session word Allowance under a global Daily budget

Amends ADR 0004, which meant the server's `OPENROUTER_API_KEY` fallback for
local use only -- nothing enforces that, so a public deployment would pay for
every visitor's Read-through with no limit. Parked until the tool looks like it
will get users; recorded now so the shape isn't re-derived.

**Decision**: Read-through runs paid by the server key become a metered
**Free tier**. Runs on a Session's own key are never metered, and never fall
back to the server key when that key fails.

- **Allowance, in words.** Each Session may send N Transcript words through
  the Free tier per rolling 24 hours. A run is charged its Transcript's full
  word count (the Read-through reads every word), and a re-run is charged
  again. Words, not audio hours: the web UI never sees audio, and words are
  known before any call and don't move with model prices.
- **Grace margin.** A run may start if its words fit in the remaining
  Allowance plus 20% of the daily Allowance; the balance then goes to zero,
  never negative, so a second upload that just crosses the line isn't wasted.
- **Daily budget, in dollars.** A global cap on Free tier spend across every
  Session per rolling 24 hours, as OpenRouter reports it -- the real limit on
  the server's bill, and the only defence against a visitor clearing cookies
  to reset their Allowance (no per-IP limit or captcha). When a reply carries
  no cost figure, the pre-run estimate is charged.
- **Check before, never stop midway.** Both limits are checked against the
  pre-run estimate; a run that starts always finishes. The worst overshoot is
  one Transcript, backstopped by a hard credit limit on a dedicated OpenRouter
  key.
- **Failures.** The Daily budget is charged whatever was actually spent. The
  Allowance is charged only when the run produced a result (a run finishing
  with some failed chunks counts).
- **One spend ledger.** An append-only file at the data root, one entry per
  run (time, Session, words, cost), written under an in-process lock; both
  totals are sums over the last 24 hours.
- **On by default.** Local use turns the limits off explicitly, so a
  deployment that forgets a setting fails safe.
- **Transparent.** The Transcript page shows words left and this
  Transcript's word count before Correct. A refusal says the Allowance (or
  Daily budget) is used up and offers "enter your own key", plus a donate link
  when one is configured. Because both limits are rolling windows, the page
  never says "today" (which reads as a midnight reset): it says when the
  oldest run's words come back, and a refusal says how long until this
  Transcript fits -- or that it never will -- as relative waits rounded up to
  the minute. A Daily budget wait is approximate, since every Session shares
  it. Donations fund the server key's credits by hand;
  donors get no extra Allowance, which would need accounts.
- The Free tier runs the default Read-through configuration.

Starting figures -- 10,000 words (about an hour of speech, ~$0.06 at Flash)
and $2 a day -- are placeholders, to be revised once real use is measured.

## Considered Options

- **Count Transcripts ("3 to 5 a day")**: Transcripts run from 1 minute to 3
  hours, so the cost of an allowance would vary more than 100x.
- **A single dollar limit per Session**: exact, but opaque to a visitor and
  shifts whenever the default model changes.
- **Truncate a run at the limit**: leaves a half-corrected Transcript.
- **Limits off unless configured**: forgetting the setting reproduces the
  unlimited-spend bug this ADR exists to close.
- **Per-IP limits or a captcha**: more to build and operate; the Daily budget
  already caps the worst case.
