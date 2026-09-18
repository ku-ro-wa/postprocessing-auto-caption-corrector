# caption-checker

Post-hoc detector for auto-generated caption (ASR) errors. Give it an existing
SRT/VTT transcript — from any captioning tool — and it flags likely
mistranscriptions (garbled technical terms, split words, domain jargon) and
suggests corrections.

## Install

```bash
uv sync                     # core detectors (lexical + phonetic)
uv sync --extra embeddings  # + local semantic-context detector (downloads torch)
```

## Usage

```bash
# human-readable report (exit status 1 when anything is flagged)
uv run caption-checker check lecture.srt

# structured output for downstream tooling
uv run caption-checker check lecture.srt --format json

# add your own domain terms, one per line
uv run caption-checker check lecture.srt --vocab my_terms.txt

# skip the embedding tier (or if the extra isn't installed)
uv run caption-checker check lecture.srt --no-embeddings

# also surface rare (not just unknown) words
uv run caption-checker check lecture.srt --oov-zipf 2.0
```

### Correcting a transcript

`correct` runs the same detection as `check`, then resolves the cheap cases
locally, sends the rest to an LLM (OpenRouter) for a judged correction, lets you
review, and writes a corrected file plus a `<OUT>.flags.json` sidecar recording
what became of every flag. It writes files and spends money, so it is a
separate verb from `check` and needs an explicit `-o`.

```bash
# interactive review; needs OPENROUTER_API_KEY (env or .env)
uv run caption-checker correct lecture.srt -o lecture.fixed.srt

# unattended: apply every correction at or above a confidence, skip the rest
uv run caption-checker correct lecture.srt -o lecture.fixed.srt --yes-above 0.8

# see the price before committing (no API call)
uv run caption-checker correct lecture.srt -o /dev/null --estimate

# pick a model; cap spend; skip the cross-run decision cache
uv run caption-checker correct lecture.srt -o out.srt \
    --model anthropic/claude-3.5-haiku --max-calls 4 --no-cache
```

Interactive keys: `y` accept, `n` skip, `e` edit then accept, `a` accept all
remaining at or above this confidence, `q` stop and write what's accepted so
far. `check` is unchanged — still read-only, still free.

Example:

```
$ caption-checker check tests/data/sample_lecture.srt --no-embeddings
sample_lecture.srt: 4 likely caption errors

[00:00:03,500 → 00:00:07,200] cue 2
  Today we're going to talk about »con sensus« algorithms.
    » con sensus  [split_word] → "consensus"  0.75

[00:00:24,000 → 00:00:28,500] cue 7
  Next week we'll look at »cubernetes« and container orchestration.
    » cubernetes  [oov+phonetic_vocab] → "Kubernetes"  0.98
```

## How it works

| Detector | Signal |
|---|---|
| `oov` | token is neither common English (`wordfreq`) nor a known domain term |
| `phonetic_vocab` | token sounds exactly like a curated domain term (Double Metaphone) |
| `phonetic_internal` | token sounds like a known-good word used elsewhere in the same transcript |
| `split_word` | 2–3 adjacent tokens joined sound like one term or a common word |
| `context_embedding` | token is a weak semantic fit for its sentence (local MiniLM; optional) |

Overlapping flags from different detectors are merged; agreement raises
confidence. The built-in vocabulary lives at
`src/caption_checker/data/domain_vocab.txt`.

Each transcript also builds its own throwaway vocabulary: terms unknown to
`wordfreq`/the curated list but spelled the same way several times (a
recurring brand, guest, or company name) are trusted as that document's own
correct term, so later occurrences aren't flagged and near-miss misspellings
elsewhere get it as a candidate. Because a deterministic ASR mistake repeats
too, near-duplicate misspellings that each independently clear the
recurrence threshold ("Kashi" and "Caushi" alongside the correct "Kalshi")
are clustered together first — only the most frequent spelling in a cluster
is trusted; the rest stay correctable.

## Web review UI

```bash
uv run caption-checker serve  # http://127.0.0.1:8000
```

Upload an SRT/VTT through the browser to get it scanned automatically by
the local detectors, then review each Flag in context — accept, reject, or
edit a suggestion before accepting — and download a corrected file that
reflects only your accepted Review Decisions. The LLM `correct` pass only
ever runs when you trigger it on a specific transcript; it uses your own
OpenRouter key entered in the browser, falling back to the server's
`OPENROUTER_API_KEY` only for local/dev use. Uploads and review state are
private to your browser session and persist across server restarts. See
`docs/adr/0003-web-ui-upload-session-persisted.md` and
`docs/adr/0004-llm-correction-manual-session-keyed.md` for the reasoning
behind these choices.

## Roadmap

1. ~~SRT/VTT parser + CLI round-trip~~
2. ~~Phonetic / statistical anomaly flagging~~
3. ~~LLM correction pass + corrected-file export (`correct`, OpenRouter, swappable model)~~
4. ~~Web review UI: upload, review, export (`serve`)~~ ← current
5. Qualitative pass over real transcripts + detector-threshold retuning

Sessions 3–4 are planned in detail in [`docs/plan-llm-correction.md`](docs/plan-llm-correction.md).
Domain vocabulary for the codebase itself is in [`CONTEXT.md`](CONTEXT.md); design
decisions in [`docs/adr/`](docs/adr/).
