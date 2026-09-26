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

## Follow-up (#25)

The Read-through is now the default LLM pass: `correct` runs it unless
`--per-flag` asks for the per-flag pass, and the web UI's Correct action
runs it, with Priming terms entered next to the API key. The web UI has no
per-flag mode; the Read-through-off mode this ADR keeps lives in the CLI.
ADR 0004 still holds: the LLM pass runs only when triggered, with the
session's key. Its ~5-10% pre-filter rationale no longer describes what
reaches the LLM, since the Read-through reads every word; spend stays gated
by the explicit trigger (about $0.05 per audio hour, above).

## Follow-up (#27)

The non-word caveat above was mostly the reply parser, not the model. Its
check for a formatting-only reply ignored spacing, so a hint whose span was
the wrong words came back not-an-error, even though the model had corrected
it ("con sensus" -> "consensus", "anthropics" -> "Anthropic's", "AIdriven" ->
"AI-driven"). A hint's verdict now counts as a Correction when it moves a word
boundary. A new find still has to change the letters: extending the rule to
new finds caught no more non-word errors and cost Earnings-21 Flag-level
precision (0.873 -> 0.857).

Measured on the same cached replies, before -> after (local in brackets):

| Dev set | Non-word recall | Flag-level precision |
|---|---|---|
| Scored | 39 -> 40 of 41 (41) | 0.771 -> 0.768 |
| Audited (was Held-out) | 9 -> 12 of 15 (14) | 0.780 -> 0.782 |
| Earnings-21 dev | 12 -> 12 of 14 (12) | 0.872 -> 0.873 |

The prompt is unchanged, so cost is unchanged ($0.05 per audio hour).

What is left is genuine dismissals, mostly K-pop names such as "Kaewan" and
"Yuha". Their local candidates are empty or wrong. A policy that kept every
dismissed OOV hint as a disagreement was rejected:
- only 7 of the 28 dismissed OOV hints touch an error;
- it would drop Audited Flag-level precision to about 0.67 for 3 more
  non-word catches;
- presetting those hints to their local candidates would apply wrong
  replacements.

The reviewer still sees these hints, preset to skip.

Looking at these misses made the Audited Held-out set a Dev set
(`audited-dev`). Any later claim about the Read-through needs fresh Held-out
data.
