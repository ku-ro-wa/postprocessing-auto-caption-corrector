# Auto-Generated Caption Error Detection Tool — Project Plan

> **Status (2026-09-22):** Sessions 1–5 below are done and shipped, plus a web
> review UI and a Regression gate that weren't in the original scope. Current
> frontier is ongoing detector-threshold retuning against that gate. This
> file is kept as the historical design record; day-to-day roadmap tracking
> lives in
> [`README.md`](README.md)'s Roadmap section and
> [`docs/plan-llm-correction.md`](docs/plan-llm-correction.md). Domain
> language is in [`CONTEXT.md`](CONTEXT.md); decisions with rationale are in
> [`docs/adr/`](docs/adr/).

## Overview
A post-hoc checker that takes an existing auto-generated transcript (SRT/VTT) — regardless of what tool produced it — and flags likely ASR errors (mistranscribed technical terms, slang, domain jargon), then suggests corrections. Chosen over a pre-processing approach because it's more original, workflow-agnostic, and solves the harder problem of catching *unanticipated* errors.

Framed as a personal build / portfolio project, not for commercialization.

## Core Pipeline
1. **Input parsing** — read SRT/VTT, extract cue text + timestamps, track word-level position across the transcript. ✅ `parser.py` / `models.py` (`Cue`, `Word` with per-cue and global position).
2. **Anomaly detection** (cheap, local, no API calls): ✅ implemented as four independent detectors in `detectors/`, ordered cheapest-first and merged, beyond what was originally sketched:
   - `oov` — token is neither common English (`wordfreq`) nor in the domain vocabulary.
   - `phonetic_vocab` — Double Metaphone match against the curated domain vocabulary.
   - `phonetic_internal` — Double Metaphone match against a known-good word used elsewhere in the same transcript.
   - `split_word` — 2–3 adjacent tokens that sound like one real term or common word (not in the original plan; added after real ASR output showed split-word garbles like "con sensus" → "consensus").
   - (A fifth, `context_embedding` — local MiniLM sentence-embedding fit — was removed 2026-09-24: on the Scored corpus it caught none of the real-word errors it targeted and added false positives. A masked-LM replacement was prototyped and not shipped: about one real catch per four extra flags.)
3. **Context-based correction** (LLM step, only for flagged residue): ✅ `correct` command / `corrector.py`. Each flag is packaged with sentence context and sent to OpenRouter for a judged correction (replacement + confidence, or a not-an-error verdict).
4. **Output** — ✅ two paths now exist:
   - CLI (`correct -o`): interactive accept/skip/edit/accept-all review, writes corrected file + `<OUT>.flags.json` sidecar.
   - Web UI (`serve`, not in the original plan): upload, review each flag in a browser (accept/reject/edit), export a corrected file from accepted Review Decisions only. See ADRs 0003/0004.

## Stack
- **Language**: Python
- **SRT/VTT parsing**: `srt`, `webvtt-py`
- **Phonetic matching**: `jellyfish` (Double Metaphone), `metaphone`
- **Word frequency / OOV**: `wordfreq`
- **LLM calls**: OpenRouter, model swappable via `--model` (default `google/gemini-2.5-flash`)
- **CLI**: `click`
- **Web UI** (not originally planned): `fastapi`, `uvicorn`, `jinja2`, `python-multipart`
- **Interface**: CLI (`check`, `correct`, `roundtrip`) plus a locally-hosted web review UI (`serve`)

## Build Order
1. ✅ SRT/VTT parser + CLI skeleton that round-trips a file unchanged (`roundtrip` command)
2. ✅ Phonetic/statistical anomaly flagging — `check` command, text and JSON output
3. ✅ LLM correction step — `correct` command, structured JSON-in/JSON-out via OpenRouter
4. ✅ Diff output + corrected file export — CLI interactive review + sidecar; **web review UI added as an unplanned 4th phase**, now itself covered by tests (upload, per-flag decisions, dashboard listing/isolation/delete, export with mixed decisions, VTT format)
5. ✅ Regression gate + Scored corpus (5 real-video fixtures) + Smoke corpus workflow — see README's `## Evaluation` section and `docs/adr/0005-scope-eval-loop-to-local-pipeline.md`. Ongoing detector-threshold retuning against the gate is the current next step.

## Cost Control (limiting LLM calls)
- **Batch requests**: ✅ implemented — flags are batched per `correct` run, not sent one call per word.
- **Aggressive pre-filtering**: ✅ five-detector pipeline keeps residue to genuinely suspicious spans before anything reaches the LLM.
- **Caching**: ✅ decision cache (`--cache-file`, default `~/.cache/caption-checker/corrections.json`, `--no-cache` to disable) — keyed on the garbled span + model identity, on the assumption a given garble resolves the same way every time.
- **Skip LLM for high-confidence cases**: ✅ "internal-match bypass" (see `CONTEXT.md`) — a flagged span that sounds exactly like a known-good word used correctly elsewhere, with one clear-winning candidate, is corrected without an LLM call. Bypassed corrections still pass through review.
- **Spend guardrails beyond the original plan**: `--estimate` (prints flag/batch count and approximate cost, no API call) and `--max-calls` (aborts before dispatch if the run would exceed N requests).

## Model Testing (via OpenRouter)
- Frontier models are unnecessary — this is a narrow, constrained classification/correction task, not open-ended reasoning.
- `--model` makes swapping trivial; default is `google/gemini-2.5-flash`.
- 🔲 Not yet done: side-by-side testing across candidates (Gemini Flash, Qwen 2.5, Llama 3.1 8B/70B, other cheap options). Real transcripts are now on hand (`tests/data/`); this is unblocked but still not started — the LLM pass stays outside the regression gate per ADR-0005, so this remains a manual comparison.

## Evaluation Approach
- **Regression gate over the local pipeline, qualitative for the LLM pass** — see README's `## Evaluation` section and `docs/adr/0005-scope-eval-loop-to-local-pipeline.md` for the full rationale (a Scored corpus this small can't support a general accuracy claim, so the gate reports recall/precision/cold-flag rate as separate floors instead of a blended score; the LLM correction pass deliberately stays outside this loop).
- `correct --eval-out FILE` still writes the planned `flagged term | model's suggestion | verdict` markdown table for manual LLM-pass spot checks — unrelated to the regression gate, which never calls the LLM.
- ✅ `tests/data/` now holds 5 real-video fixtures (paired `.auto`/Reference-caption transcripts) plus `scored_corpus.json`, curated per ADR-0005; `pytest tests/test_regression_gate.py` runs the gate.
- A larger, unscored Smoke corpus run (`check` over a bigger transcript batch, eyeballed) is the informal generalization check — see README.

## Immediate Next Steps
- [x] Build SRT/VTT parser + CLI skeleton (Session 1)
- [x] Implement phonetic/statistical flagging (Session 2)
- [x] Wire up OpenRouter LLM correction step with model-swap abstraction (Session 3)
- [x] Add diff/export output — CLI (Session 4) and, beyond plan, a web review UI
- [x] Source real test videos and build the Regression gate + initial Scored corpus (Session 5)
- [ ] Keep growing the Scored corpus as more paired videos are sourced (ongoing, per ADR-0005 — not a one-time close-out)
- [ ] Run a Smoke corpus pass (larger unscored batch) to sanity-check generalization
- [ ] A/B a couple of cheap OpenRouter models against each other on a real transcript set (LLM pass stays outside the regression gate)
