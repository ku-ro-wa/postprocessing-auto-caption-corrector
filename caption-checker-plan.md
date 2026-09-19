# Auto-Generated Caption Error Detection Tool — Project Plan

> **Status (2026-09-19):** Sessions 1–4 below are done and shipped, plus a web
> review UI that wasn't in the original scope. Current frontier is Session 5:
> a qualitative pass over real transcripts. This file is kept as the
> historical design record; day-to-day roadmap tracking lives in
> [`README.md`](README.md)'s Roadmap section and
> [`docs/plan-llm-correction.md`](docs/plan-llm-correction.md). Domain
> language is in [`CONTEXT.md`](CONTEXT.md); decisions with rationale are in
> [`docs/adr/`](docs/adr/).

## Overview
A post-hoc checker that takes an existing auto-generated transcript (SRT/VTT) — regardless of what tool produced it — and flags likely ASR errors (mistranscribed technical terms, slang, domain jargon), then suggests corrections. Chosen over a pre-processing approach because it's more original, workflow-agnostic, and solves the harder problem of catching *unanticipated* errors.

Framed as a personal build / portfolio project, not for commercialization. Prioritized ahead of the privacy-oriented Anki clone (Tauri + React + local LLM + custom FSRS) because it's lower-friction and better suited to a busy school/self-study period; the Anki clone is deferred to a less busy stretch.

## Core Pipeline
1. **Input parsing** — read SRT/VTT, extract cue text + timestamps, track word-level position across the transcript. ✅ `parser.py` / `models.py` (`Cue`, `Word` with per-cue and global position).
2. **Anomaly detection** (cheap, local, no API calls): ✅ implemented as five independent detectors in `detectors/`, ordered cheapest-first and merged, beyond what was originally sketched:
   - `oov` — token is neither common English (`wordfreq`) nor in the domain vocabulary.
   - `phonetic_vocab` — Double Metaphone match against the curated domain vocabulary.
   - `phonetic_internal` — Double Metaphone match against a known-good word used elsewhere in the same transcript.
   - `split_word` — 2–3 adjacent tokens that sound like one real term or common word (not in the original plan; added after real ASR output showed split-word garbles like "con sensus" → "consensus").
   - `context_embedding` — local MiniLM sentence-embedding fit, optional (`--no-embeddings` / not installed by default, since it pulls in torch).
3. **Context-based correction** (LLM step, only for flagged residue): ✅ `correct` command / `corrector.py`. Each flag is packaged with sentence context and sent to OpenRouter for a judged correction (replacement + confidence, or a not-an-error verdict).
4. **Output** — ✅ two paths now exist:
   - CLI (`correct -o`): interactive accept/skip/edit/accept-all review, writes corrected file + `<OUT>.flags.json` sidecar.
   - Web UI (`serve`, not in the original plan): upload, review each flag in a browser (accept/reject/edit), export a corrected file from accepted Review Decisions only. See ADRs 0003/0004.

## Stack
- **Language**: Python
- **SRT/VTT parsing**: `srt`, `webvtt-py`
- **Phonetic matching**: `jellyfish` (Double Metaphone), `metaphone`
- **Word frequency / OOV**: `wordfreq`
- **Local embeddings (optional)**: `sentence-transformers` (extras group `embeddings`, lazy-imported)
- **LLM calls**: OpenRouter, model swappable via `--model` (default `google/gemini-2.0-flash-001`)
- **CLI**: `click`
- **Web UI** (not originally planned): `fastapi`, `uvicorn`, `jinja2`, `python-multipart`
- **Interface**: CLI (`check`, `correct`, `roundtrip`) plus a locally-hosted web review UI (`serve`)

## Build Order
1. ✅ SRT/VTT parser + CLI skeleton that round-trips a file unchanged (`roundtrip` command)
2. ✅ Phonetic/statistical anomaly flagging — `check` command, text and JSON output
3. ✅ LLM correction step — `correct` command, structured JSON-in/JSON-out via OpenRouter
4. ✅ Diff output + corrected file export — CLI interactive review + sidecar; **web review UI added as an unplanned 4th phase**, now itself covered by tests (upload, per-flag decisions, dashboard listing/isolation/delete, export with mixed decisions, VTT format)
5. 🔲 Qualitative pass over real transcripts + detector-threshold retuning — **current next step**, not yet started

## Cost Control (limiting LLM calls)
- **Batch requests**: ✅ implemented — flags are batched per `correct` run, not sent one call per word.
- **Aggressive pre-filtering**: ✅ five-detector pipeline keeps residue to genuinely suspicious spans before anything reaches the LLM.
- **Caching**: ✅ decision cache (`--cache-file`, default `~/.cache/caption-checker/corrections.json`, `--no-cache` to disable) — keyed on the garbled span + model identity, on the assumption a given garble resolves the same way every time.
- **Skip LLM for high-confidence cases**: ✅ "internal-match bypass" (see `CONTEXT.md`) — a flagged span that sounds exactly like a known-good word used correctly elsewhere, with one clear-winning candidate, is corrected without an LLM call. Bypassed corrections still pass through review.
- **Spend guardrails beyond the original plan**: `--estimate` (prints flag/batch count and approximate cost, no API call) and `--max-calls` (aborts before dispatch if the run would exceed N requests).

## Model Testing (via OpenRouter)
- Frontier models are unnecessary — this is a narrow, constrained classification/correction task, not open-ended reasoning.
- `--model` makes swapping trivial; default is `google/gemini-2.0-flash-001`.
- 🔲 Not yet done: side-by-side testing across candidates (Gemini Flash, Qwen 2.5, Llama 3.1 8B/70B, other cheap options). Revisit once real transcripts are on hand for Session 5.

## Evaluation Approach
- **Qualitative, not formal benchmarking** — still the right call; no ground-truth dataset exists yet.
- `correct --eval-out FILE` already writes the planned `flagged term | model's suggestion | verdict` markdown table (verdict column left blank for manual fill-in) — the scaffolding for this exists ahead of the actual eval pass.
- 🔲 Not yet run against real (non-synthetic) transcripts. `tests/data/` currently holds only synthetic/sample fixtures for unit tests, not a qualitative-eval corpus.
- Revisit formal metrics after 3–5 real test videos, as originally planned.

## Immediate Next Steps
- [ ] User to source a handful of real test videos with known/suspected caption errors — still the blocking input for Session 5
- [x] Build SRT/VTT parser + CLI skeleton (Session 1)
- [x] Implement phonetic/statistical flagging (Session 2)
- [x] Wire up OpenRouter LLM correction step with model-swap abstraction (Session 3)
- [x] Add diff/export output — CLI (Session 4) and, beyond plan, a web review UI
- [ ] Run first qualitative pass across real test videos and log the `flagged term | suggestion | verdict` table (Session 5)
- [ ] A/B a couple of cheap OpenRouter models against each other on that same set once it exists
