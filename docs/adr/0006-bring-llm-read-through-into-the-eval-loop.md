---
status: accepted
---

# Bring an LLM Read-through into the eval loop, judged on Held-out sets

Partly supersedes ADR 0005, which kept the LLM pass outside the tuning loop
to protect LLM spend. Measured spend made that premise obsolete (29 requests,
73.6K tokens: $0.03; reading every word of a transcript costs pennies per hour
of audio), while the local pipeline hit a ceiling: the LLM only sees what the
detectors flag, and they catch 9 of 55 real-word errors on the Scored corpus.
Worse, every local number is a Dev set number -- vocab, thresholds and gate
floors were all tuned on those 5 videos.

**Decision**: Prototype a Read-through -- one LLM pass over the whole
transcript, taking detector Flags and Priming terms as hints, returning Flags
with Corrections -- and iterate on it inside the eval loop, on Dev sets only
(the current Scored corpus plus a few Earnings-21 calls). Judge it once,
against a rule fixed before any results: on **both** Held-out sets (5 new
Audited transcripts, and the Earnings-21 `eval-10` Auto-labelled corpus built
from Google's ASR output), the Read-through wins if

- real-word recall is at least 2x the frozen local pipeline's (`2de1655`),
  counting detection -- a Flag touching the error -- not an exact
  candidate match (Earnings-21 candidates are Rev's verbatim words, too
  noisy to require),
- Flag-level precision is no worse than the local pipeline's, and
- cost is under $0.10 per audio hour.

On a win, local work narrows to grounding (vocab, doc vocab, Priming terms,
candidates), cheap flags, the bypass and cache; the parked detector-gate
changes stay parked. On a loss, they come back. The per-flag correction pass
stays as the Read-through-off mode either way. The offline pytest Regression
gate is unchanged; LLM and Earnings-21 scoring run from a separate eval
command.

**Considered and rejected**: a whole-transcript LLM rewrite diffed back to
spans (loses faithfulness and span-level review); a masked-LM detector (about
one real catch per four extra flags); the context-embedding detector (caught
nothing). **Parked**: continued local recall work, other ASR vendors'
Earnings-21 outputs, a wider OpenRouter model comparison -- see the plan's map
issue for the full list and what would revive each.

## Result (2026-09-26)

Scored once with `eval --final`. Local is `2de1655`'s detection code; the
only change since is that Priming terms now feed the vocab. The
Read-through is the configuration frozen at `33ca55b` (`gemini-2.5-flash`).

| Held-out set | Real-word recall | Flag-level precision | Cost / audio hr |
|---|---|---|---|
| Audited | 22/79 vs 6/79 (3.7x) | 0.780 vs 0.343 | $0.05 |
| Earnings-21 `eval-10` | 409 vs 49 of 3985 (8.3x) | 0.874 vs 0.514 | $0.06 |
| ... with Priming terms | 435 vs 49 of 3985 (8.9x) | 0.884 vs 0.516 | $0.06 |

(Read-through vs local.) The Read-through meets every condition on both
sets. One caveat: on the Audited set it caught fewer non-word errors than
local (9/15 vs 14/15).
