# caption-checker

Post-hoc detector for auto-generated caption (ASR) errors. Give it an existing
SRT/VTT transcript — from any captioning tool — and it flags likely
mistranscriptions (garbled technical terms, split words, domain jargon) and
suggests corrections.

## Install

```bash
uv sync
```

## Usage

```bash
# human-readable report (exit status 1 when anything is flagged)
uv run caption-checker check lecture.srt

# structured output for downstream tooling
uv run caption-checker check lecture.srt --format json

# add your own domain terms, one per line
uv run caption-checker check lecture.srt --vocab my_terms.txt

# Priming terms for this one transcript (a speaker, product, company)
uv run caption-checker check lecture.srt --priming-term "Jensen Huang" --priming-term Nvidia

# also surface rare (not just unknown) words
uv run caption-checker check lecture.srt --oov-zipf 2.0
```

### Correcting a transcript

`correct` runs the same detection as `check`, then has an LLM (OpenRouter)
read the whole transcript -- the Read-through (ADR 0006) -- lets you review,
and writes a corrected file plus a `<OUT>.flags.json` sidecar recording what
became of every flag. It writes files and spends money, so it is a separate
verb from `check` and needs an explicit `-o`.

```bash
# interactive review; needs OPENROUTER_API_KEY (env or .env)
uv run caption-checker correct lecture.srt -o lecture.fixed.srt

# unattended: apply every correction at or above a confidence, skip the rest
uv run caption-checker correct lecture.srt -o lecture.fixed.srt --yes-above 0.8

# Priming terms: names and terms known to occur in this recording
uv run caption-checker correct lecture.srt -o out.srt --priming-term "Jensen Huang"

# see the price of the pass that would run, before committing (no API call)
uv run caption-checker correct lecture.srt -o /dev/null --estimate

# pick a model; cap spend
uv run caption-checker correct lecture.srt -o out.srt \
    --model anthropic/claude-3.5-haiku --max-calls 4

# the older per-flag pass instead (skip the cross-run decision cache too)
uv run caption-checker correct lecture.srt -o out.srt --per-flag --no-cache
```

The Read-through sends one request per ~400-word chunk, with every flag as a
hint plus the Priming terms, and also reports errors no detector raised;
those carry the `read_through` detector. It has no bypass or decision cache,
and drops any verdict whose span crosses a cue boundary (a correction is
spliced into one cue). `--per-flag` sends only the flagged spans that the
internal-match bypass and the decision cache (`--cache-file`, `--no-cache`)
don't resolve -- cheaper, but it can't find what the detectors missed.

Interactive keys: `y` accept, `n` skip, `e` edit then accept, `a` accept all
remaining at or above this confidence, `q` stop and write what's accepted so
far. `check` is unchanged — still read-only, still free.

Example:

```
$ caption-checker check tests/data/sample_lecture.srt
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
| `phonetic_vocab` | token sounds exactly like a curated domain term (Double Metaphone), or is one with the wrong casing ("Deepseek") |
| `phonetic_internal` | token sounds like a known-good word used elsewhere in the same transcript |
| `split_word` | 2–3 adjacent tokens joined sound like one term or a common word |

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
reflects only your accepted Review Decisions. The LLM pass -- the
Read-through, with any Priming terms you enter next to the key -- only ever
runs when you trigger it on a specific transcript. Errors it finds that the
local scan missed join the transcript as new Flags to review like any other;
Flags it judged not an error are shown as dismissed and left unchanged. If
some chunks of the transcript failed, the page says how many, and the Flags
in them stay unjudged (only a run where every chunk failed can be retried). It uses your own OpenRouter key entered in the
browser, falling back to the server's `OPENROUTER_API_KEY` only for
local/dev use. Uploads and review state are
private to your browser session and persist across server restarts. See
`docs/adr/0003-web-ui-upload-session-persisted.md` and
`docs/adr/0004-llm-correction-manual-session-keyed.md` for the reasoning
behind these choices.

## Evaluation

Tuning detector thresholds (in `oov`, `phonetic_vocab`, `phonetic_internal`,
`split_word`) needs a way to confirm a change didn't
quietly regress previously-fixed behavior, without re-eyeballing the web
review UI or spending on the LLM pass.

**Regression gate** — automatic, part of the test suite:

```bash
uv run pytest tests/test_regression_gate.py -v
```

Runs `detect()` (no network calls) against the Scored corpus
(`tests/data/scored_corpus.json`) and reports flag recall, precision, and
cold-flag rate as three separate numbers, checked against hand-set floors in
`tests/test_regression_gate.py`. Add a should-flag or should-not-flag entry
to the corpus whenever a real transcript reveals a missed error or a false
positive, so the gate permanently guards against it resurfacing. See
`docs/adr/0005-scope-eval-loop-to-local-pipeline.md` for how entries are
curated: compare a video's `.auto` transcript against its Reference caption
for candidate mismatches, then spot-check each one against the actual audio
before adding it — a Reference caption isn't verified ground truth on its
own.

**Eval command** — manual, no floors:

```bash
uv run caption-checker eval --corpus scored --system local
```

Scores a system under test on a named corpus and prints recall (overall and
by kind), case precision, Flag-level precision and cold-flag rate, each as
its own number. Recall counts detection on every corpus: a case is caught
by any flag touching it, and each kind's line also shows how many of those
flags proposed the case's candidate (ADR 0006). The pytest Regression gate
still requires the candidate. Flag-level precision (flags touching any labelled error, out
of all flags emitted) is computed only over transcripts whose errors are
listed exhaustively — the 5 Audited transcripts in the Scored corpus, not the
hand-made fixtures. Corpora and systems are registered in
`caption_checker/evaluation.py` (`CORPORA`, `SYSTEMS`).

```bash
uv run caption-checker eval --system read-through [--model SLUG]
```

scores the Read-through (it calls OpenRouter, so it costs money: about
$0.05–0.07 per audio hour with the default model on the Dev sets). It counts
only the Flags it claims are errors, and adds its spend per audio hour plus
any failed chunks. The default model,
`google/gemini-2.5-flash`, was picked over `google/gemini-2.5-flash-lite`
on the Dev sets (issue #23): Lite is about 3x cheaper but its Flag-level
precision on the Scored corpus fell below the local pipeline's (0.57 vs
0.71) and some of its Earnings-21 chunks failed on oversized replies.

**Audited Held-out set** — `audited-heldout`, final comparison only:

```bash
uv run caption-checker eval --corpus audited-heldout --final
```

Five more Audited transcripts (`tests/data/audited_heldout_corpus.json`,
issue #22): K-pop, makeup, coffee, CEO pay and restaurant desserts, two with
accented main speakers. They were audited after the local pipeline was frozen
at `2de1655`. They list errors only (no should-not-flag cases), so case
precision is vacuous and Flag-level precision is the number to read. Like
`earnings21-heldout`, `eval` refuses it without `--final`, and a video used
to motivate a change moves to the Dev set.

**Earnings-21 Auto-labelled corpora** — built locally, never committed:

```bash
uv run caption-checker build-earnings21            # fetch + build into .cache/earnings21/
uv run caption-checker eval --corpus earnings21-dev [--priming]
```

`build-earnings21` downloads Google's ASR output and Rev's verbatim
references for the Earnings-21 calls
([revdotcom/speech-datasets](https://github.com/revdotcom/speech-datasets),
CC BY-SA 4.0) into the gitignored `.cache/earnings21/raw/`, writes Google's
output as one SRT per call, aligns it against the reference, and turns each
disagreement into a case: `format`, `function-word`, `non-word` or
`real-word`, plus an `entity` tag on names. Deletions (nothing on screen),
filler-only regions and regions over 6 tokens a side (alignment drift) are
dropped and counted. `earnings21-heldout` is the dataset's `eval10` list
(11 calls) and is scored only for ADR 0006's final comparison (`eval`
refuses it without `--final`);
`earnings21-dev` is the first 5 other calls; the rest are never fetched.
`--priming` hands each call's company name to the system as Priming terms.
Headline recall covers `non-word` + `real-word`; the candidate counts are
weak here, since candidates are Rev's verbatim words. Every number here is Google
2021 ASR against verbatim-style, noisy labels: use it to compare systems,
not as an absolute score.

**Smoke corpus** — manual, no dedicated tooling: after a threshold change
passes the regression gate, sanity-check generalization by running `check`
over a larger batch of real, unscored transcripts and eyeballing the result:

```bash
for f in path/to/smoke-batch/*.srt; do
  uv run caption-checker check "$f" --format json
done
```

There's no answer key here — watch for a flag-rate spike or an unfamiliar
class of flag compared to prior runs, which signals the change overfit to
the small Scored corpus rather than generalizing.

## Roadmap

1. ~~SRT/VTT parser + CLI round-trip~~
2. ~~Phonetic / statistical anomaly flagging~~
3. ~~LLM correction pass + corrected-file export (`correct`, OpenRouter, swappable model)~~
4. ~~Web review UI: upload, review, export (`serve`)~~
5. ~~Regression gate + Scored/Smoke corpus for detector tuning~~
6. Ongoing detector-threshold retuning against the regression gate ← current

Sessions 3–4 are planned in detail in [`docs/plan-llm-correction.md`](docs/plan-llm-correction.md).
Domain vocabulary for the codebase itself is in [`CONTEXT.md`](CONTEXT.md); design
decisions in [`docs/adr/`](docs/adr/).
