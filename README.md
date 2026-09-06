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

## Roadmap

1. ~~SRT/VTT parser + CLI round-trip~~
2. ~~Phonetic / statistical anomaly flagging~~ ← current
3. LLM correction pass over the flagged JSON (OpenRouter, swappable model)
4. Diff view + corrected-file export
