# Auto-Generated Caption Error Detection Tool — Project Plan

## Overview
A post-hoc checker that takes an existing auto-generated transcript (SRT/VTT) — regardless of what tool produced it — and flags likely ASR errors (mistranscribed technical terms, slang, domain jargon), then suggests corrections. Chosen over a pre-processing approach because it's more original, workflow-agnostic, and solves the harder problem of catching *unanticipated* errors.

Framed as a personal build / portfolio project, not for commercialization. Prioritized ahead of the privacy-oriented Anki clone (Tauri + React + local LLM + custom FSRS) because it's lower-friction and better suited to a busy school/self-study period; the Anki clone is deferred to a less busy stretch.

## Core Pipeline
1. **Input parsing** — read SRT/VTT, extract cue text + timestamps, track word-level position across the transcript.
2. **Anomaly detection** (cheap, local, no API calls):
   - Phonetic similarity flags (Soundex / Metaphone / Double Metaphone) against a domain vocabulary or against other terms used elsewhere in the same transcript.
   - Statistical oddity: words appearing once that don't fit the surrounding context (e.g. rolling embedding similarity).
3. **Context-based correction** (LLM step, only for flagged residue):
   - Package each flagged span with surrounding sentence(s) and any similar/repeated terms elsewhere in the transcript.
   - Send to an LLM to judge plausibility and suggest a correction.
4. **Output** — diff view (original vs. suggested correction), optional confidence note, exportable corrected SRT/VTT.

## Stack
- **Language**: Python
- **SRT/VTT parsing**: `srt` or `webvtt-py`
- **Phonetic matching**: `jellyfish`
- **LLM calls**: via OpenRouter, model swappable through a thin abstraction layer (model name as config/CLI param) — enables A/B testing cheap models instead of committing to a frontier model
- **Interface**: CLI first (argparse/click), no GUI needed for the prototype

## Build Order (a few sessions)
1. SRT/VTT parser + CLI skeleton that round-trips a file unchanged
2. Phonetic/statistical anomaly flagging — print flagged terms with context
3. LLM correction step — send flagged spans + context, get structured suggestions back
4. Diff output + corrected file export — test against real auto-captioned videos

## Cost Control (limiting LLM calls)
- **Batch requests**: send all flags for a transcript (or chunks of ~200 cues) as one structured JSON-in/JSON-out request rather than one call per flagged word.
- **Aggressive pre-filtering**: tune phonetic/statistical thresholds so only ~5–10% of words reach the LLM tier; false positives are cheap to filter early, expensive once sent to the LLM.
- **Caching**: cache correction decisions for recurring terms (e.g. domain vocabulary across a lecture series).
- **Skip LLM for high-confidence cases**: if a flagged word has an exact phonetic match to a term already used correctly elsewhere in the same document, correct it directly without an LLM call.

## Model Testing (via OpenRouter)
- Frontier models are unnecessary — this is a narrow, constrained classification/correction task, not open-ended reasoning.
- Candidates to test: Gemini Flash, Qwen 2.5, Llama 3.1 (8B/70B), other cheap options.
- Keep the LLM call strictly JSON-in/JSON-out (flagged term + context → correction + confidence) — smaller models perform much better on constrained tasks than open-ended ones.

## Evaluation Approach
- **Qualitative, not formal benchmarking** — appropriate for prototype stage; no ground-truth dataset exists yet, and premature scoring risks optimizing for a metric rather than real failure modes.
- Lightweight structure to avoid pure vibes-based review: for each test video, log a simple table (markdown or JSON) with:
  - `flagged term | model's suggestion | verdict (correct / wrong / missed entirely)`
- This enables side-by-side comparison across models without re-watching videos, and leaves raw data available if formal scoring is added later.
- Revisit after 3–5 test videos to see if patterns justify more formal metrics.

## Immediate Next Steps
- [ ] User to source a handful of test videos with known/suspected caption errors
- [ ] Build SRT/VTT parser + CLI skeleton (Session 1)
- [ ] Implement phonetic/statistical flagging (Session 2)
- [ ] Wire up OpenRouter LLM correction step with model-swap abstraction (Session 3)
- [ ] Add diff/export output and run first qualitative pass across test videos (Session 4)
