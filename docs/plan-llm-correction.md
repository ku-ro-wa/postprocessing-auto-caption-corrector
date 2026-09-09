# Plan: LLM correction pass + corrected-file export (Sessions 3–4)

Status as of 2026-09-09: Sessions 1–2 complete (parser, CLI, five local
detectors, merge, sentence context, JSON handoff). This plan covers the
remaining two sessions, agreed through a grilling pass. Terminology follows
`CONTEXT.md`.

## Goal

Add a `correct` command that takes the flags `check` produces, resolves the
cheap ones locally, sends the residue to an LLM for a judged correction, lets
the user review, and writes a corrected SRT/VTT plus a sidecar record.

Session 3 is built and tested against the synthetic `tests/data/sample_lecture`
files. Real auto-captioned transcripts are being gathered in parallel and feed
detector tuning + qualitative eval in Session 4.

---

## Command surface

`caption-checker correct FILE -o OUT [options]` — new verb, `check` unchanged.

- Reuses `detect()` internally; no detection logic is duplicated.
- `-o / --output` is **required**; error `pass -o PATH` if absent. Interactive
  prompts and the report go to stderr, so stdout stays clean.
- Exit 0 on completion (even if some flags were left uncorrected), non-zero
  only on hard failure (bad input, no API key, run aborted).

### Options

| Option | Effect |
|---|---|
| `--model SLUG` | OpenRouter model. Default: a Gemini Flash-class slug, pinned at build time, overridable. |
| `--yes-above CONF` | Non-interactive: apply every correction at ≥ CONF, skip the rest. Required when stdin is not a TTY. |
| `--cache-file PATH` | Decision-cache location. Default `~/.cache/caption-checker/corrections.json` (XDG). |
| `--no-cache` | Bypass cache read and write (use for fresh eval runs). |
| `--eval-out PATH` | Also write a markdown eval table (verdict column blank). Off by default. |
| `--estimate` | Dry run: print flag count → chunk count → approximate cost, then exit without calling. |
| `--max-calls N` | Abort if the run would exceed N LLM requests. |
| `--vocab`, `--no-embeddings`, `--oov-zipf` | Same as `check`, passed through to detection. |

---

## Pipeline

1. **Detect** — `detect(cues, vocab, config)` → merged flags (existing code).
2. **Bypass** — resolve flags that need no LLM call (see below). These become
   pending corrections with source `bypass`.
3. **Cache lookup** — for the remaining flags, key `(cleaned_span, model_id)`
   against the decision cache. Hits become pending corrections with source
   `cache`.
4. **Residue → LLM** — everything still unresolved is chunked (~25 flags per
   request) and sent to the `Corrector`. Responses are parsed into
   corrections; new answers are written back to the cache.
5. **Review** — interactive loop (or `--yes-above` filter) over *all* pending
   corrections, bypassed and cached included.
6. **Apply** — accepted corrections spliced into cue text by character offset
   (see ADR-0001); corrected transcript serialized to `-o`.
7. **Sidecar** — `<OUT>.flags.json` records every flag and its outcome.
8. **Eval table** — if `--eval-out`, dump the rows.

### Internal-match bypass (step 2)

Apply a correction without an LLM call when **all** hold:

- the flag is a *pure* `phonetic_internal` flag (a merged
  `oov+phonetic_internal` still goes to the LLM — another detector saw
  something the phonetic match alone doesn't explain);
- it has exactly one candidate, **or** the top candidate's Jaro-Winkler score
  beats the runner-up by ≥ `bypass_jw_margin` (new `DetectConfig` field);
- the matched known-good word actually appears elsewhere in the transcript
  (already true by construction of `phonetic_internal`).

Bypassed corrections still appear in review, pre-filled `y`.

### Batching + failure handling (step 4)

- Chunk size ~25 flags. A 60-min lecture at ~6% flag rate is ~50–90 flags →
  2–4 requests.
- On a malformed / unparseable / id-mismatched chunk response: retry the
  identical request **once**, then on a second failure skip that chunk. Its
  flags pass through uncorrected and are recorded in the sidecar as
  `skipped-parse-failure`.
- `--max-calls` is checked before dispatching; `--estimate` stops here.

---

## Data shapes

### `FlagContext` (what each flag carries into the LLM)

- `id` — stable back-reference for matching the response
- `span` — the flagged text
- `sentence` — the reconstructed sentence context
- `candidates` — detector candidates, ranked (may be empty)
- `detector` — detector name(s)
- `reason` — the detector's one-line rationale (front-loads the hypothesis to
  test; matters most for no-candidate flags)
- `nearby` — the cue before and the cue after, for wider context
- `related` — up to 3 other transcript spans sharing a phonetic code with this
  one (so the model can see the term used correctly elsewhere)

### `Correction` (LLM response, per flag)

```json
{ "id": "...", "replacement": "consensus" | null, "confidence": 0.0-1.0, "rationale": "one line" }
```

`replacement: null` = not-an-error verdict. Zero-shot is not used — the system
prompt carries 2–4 fixed few-shot examples, one per archetype (phonetic-vocab
hit, split-word, oov-with-no-candidate, not-an-error), ~200 tokens overhead.

### Sidecar `<OUT>.flags.json`

List of `{ flag, outcome, correction? }` where `outcome` ∈ `applied`,
`rejected`, `not-an-error`, `bypassed`, `cached`, `skipped-parse-failure`.

### Eval table (`--eval-out`)

Markdown, columns: `timestamp | span | detector | suggestion | llm_conf |
verdict`. `verdict` left blank for manual `correct / wrong / missed` scoring.

---

## Interactive review

Per pending correction, print: timestamp, the cue with the span marked, the
suggested replacement, its LLM confidence, and the one-line rationale. Keys:

- `y` — accept
- `n` — skip (leave the span as-is)
- `e` — edit the replacement text, then accept
- `a` — accept all remaining at ≥ this correction's confidence
- `q` — stop reviewing and write what's accepted so far

Not-an-error verdicts appear pre-set to `n`, overridable. When stdin is not a
TTY and `--yes-above` was not given, exit with an error naming the flag.

---

## Module layout

- `correct.py` — orchestrator: bypass, cache, chunking, review, apply, sidecar.
- `corrector.py` — `Corrector` protocol; `OpenRouterCorrector` (real,
  `python-dotenv` → `.env`, fallback `OPENROUTER_API_KEY` env, clear error if
  neither); `StubCorrector` (tests / offline).
- `cache.py` — decision-cache load/lookup/store.
- `apply.py` — character-offset splice into cue text; serialize.
- `prompt.py` — system prompt + few-shot examples + response parsing.
- `cli.py` — the `correct` command wiring.

The detection pipeline never imports the OpenRouter SDK.

---

## Testing

- `StubCorrector` drives all `correct` unit tests; no network in the suite.
- Round-trip-with-one-edit test: parse → apply one correction → serialize →
  everything outside the corrected span is byte-identical.
- Bypass test: a `phonetic_internal` flag with a clear winner is applied with
  zero `Corrector` calls; a merged flag is not.
- Cache test: second run of the same file makes no `Corrector` calls;
  `--no-cache` forces them.
- Parse-failure test: `StubCorrector` returns garbage → chunk retried once →
  flags land as `skipped-parse-failure`, run still completes.
- Non-TTY without `--yes-above` errors.

---

## Session split

**Session 3** — `corrector.py`, `cache.py`, `prompt.py`, bypass + chunking +
review in `correct.py`, `--estimate` / `--max-calls` / `--eval-out`, tests
against the synthetic sample.

**Session 4** — `apply.py` + corrected-file export + sidecar, wire the full
`correct` flow end to end, run the first qualitative pass over real gathered
transcripts, retune detector thresholds against what shows up, fill an eval
table.

---

## Deferred

- `compare` subcommand for multi-model A/B — revisit after 3–5 real
  transcripts; until then A/B is `--eval-out` twice and a diff.
- Blending detector and LLM confidence into one score — no data to justify a
  formula yet; both are kept separate on the record.
- Decision-cache keying on span alone could mis-serve a garble that resolves
  two ways across different subjects. `--no-cache` plus review is the current
  mitigation; promote to an ADR if it actually bites.
