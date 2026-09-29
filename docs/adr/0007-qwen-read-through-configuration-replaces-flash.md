---
status: accepted
---

# Switch the default Read-through to `qwen3.6-plus-p2`, judged on fresh Held-out sets

Follows ADR 0006, whose Read-through ran one model, `google/gemini-2.5-flash`,
with a prompt tuned to it (v4). The model had become the biggest lever left
on recall, Flash was more than a year old, and spend sat at about half the
$0.10-per-audio-hour cap. #28 compared **Read-through configurations** (model
+ prompt + reply format, judged and frozen as one unit) under a rule fixed
before any results.

**Decision**: The default Read-through becomes `qwen3.6-plus-p2`:
`qwen/qwen3.6-plus`, reasoning off, with prompt v4 plus the "two tests"
error rule (a change must fix words that don't make sense as they stand,
and must sound nearly the same). Its prompt really differs from v4, so the
switch needs the default to be a configuration, not a model ID: that is
#40. Until #40 lands, `correct` and the web UI still run Flash. The backup is
`flash-v4`; no challenger qualified as backup.

## The rule

Shortlist, all pinned OpenRouter IDs: `google/gemini-2.5-flash` (baseline,
v4 unchanged), `google/gemini-3.8-flash`, `google/gemini-3.5-flash-lite`,
`openai/gpt-5.6-luna`, `deepseek/deepseek-v4-pro`, `qwen/qwen3.6-plus`. Each
challenger got up to 3 prompt passes on the YouTube Dev sets, limited to the
reply format, what counts as an error, and brevity, and was then frozen
(#33).

**Dev picking**, on Earnings-21 dev with Priming terms (#34). A configuration
qualifies on its first run if Flag-level precision is at least 0.85, its
YouTube precision is no worse than Flash's, cost is under the cap, and it
loses no more chunks than Flash. Qualifiers are ranked by real-word recall,
the mean of 3 runs for the top two; within 3% relative, the cheaper wins.

**Final**, with each configuration scored once on the same commit
(`3bedf1f`). Switch only if all of these hold:
- on Earnings-21 `heldout-2` (10 calls never fetched before, seeded draw,
  primed), real-word recall is at least 1.10x Flash's;
- on `heldout-2`, Flag-level precision is within 0.02 of Flash's;
- on `audited-heldout-2` (6 new YouTube videos), real-word catches are at
  least Flash's, and Flag-level precision is at least Flash's minus 0.02;
- on both sets, cost is under $0.10 per audio hour.

Two rule changes were made after seeing Dev numbers, and before any
Earnings-21 run, each posted on #28 first. The cost cap was raised to $0.20,
as an experiment. The Scored precision gate was pooled with `audited-dev`;
that change added the `audited-heldout-2` precision condition. #34 reported
all four combinations of gate and cap, and each picked `qwen3.6-plus-p2`.
**Under the original rules (Scored gate, $0.10 cap) the winner is the same**,
and it costs $0.05 per audio hour, so ADR 0006's $0.10 cap stands.

## Dev result (#34)

Only `qwen3.6-plus-p2` qualified. On Earnings-21 dev, primed, it averaged
468.0 real-word catches over 3 runs, against Flash's 199.3. The rule didn't
say whether the gates apply to repeat runs too; the maintainer ruled (on #28)
that gates are read from the first run and repeats only feed the ranking.
Read the other way, Qwen's repeats would fail (run 2 precision 0.849, run 3
one failed chunk) and Flash would have stayed. The others were out for these
reasons:
- `gemini-3.8-flash-p2`: most recall (563), but 5 reply-format failed chunks,
  and over $0.10;
- `gemini-3.5-flash-lite-p2`: a failed chunk, and both precision gates;
- `gpt-5.6-luna-p2`: both precision gates, and over $0.10;
- `deepseek-v4-pro-p2`: both precision gates.

## Result (2026-09-29)

Scored once each with `eval --final` at `3bedf1f`.

| Held-out set | Real-word recall | Flag-level precision | Cost / audio hr | Failed chunks |
|---|---|---|---|---|
| Earnings-21 `heldout-2`, primed | 713 vs 383 of 3259 (1.86x) | 0.854 vs 0.841 | $0.05 vs $0.05 | 2 vs 1 |
| `audited-heldout-2` | 20 vs 14 of 66 | 0.663 vs 0.675 (-0.012) | $0.05 vs $0.05 | 0 vs 0 |

(`qwen3.6-plus-p2` vs `flash-v4`.) The finalist meets every condition on
both sets. Beyond real-word recall:
- **Non-word recall:** Qwen caught fewer non-word errors on `heldout-2` (15
  vs 19 of 21) but more on `audited-heldout-2` (27 vs 25 of 29).
- **Entity recall:** Qwen was higher on both sets: 206 vs 197 of 536, and 22
  vs 18 of 33.
- **Failed chunks:** the final rule doesn't gate on them. A chunk fails only
  when its retry fails too. Qwen lost 2 chunks on `heldout-2` to reply-format
  failures, against Flash's 1; none was a request error.

An earlier Flash run on `heldout-2`, at `abc3ee9`, crashed at scoring. One
case's context had lost a bare "." token, so it no longer matched; no
Read-through output was printed. `3bedf1f` fixed the corpus builder, and both
configurations were then scored at that commit. `eval` now checks that every
case can be found before any paid call. The rebuild changed only how case
contexts are written: spans, kinds, counts and transcripts are identical in
every Earnings-21 split, so #34's Dev numbers stand. Counting the crashed
run, #28 spent about $15.01 against its $15 budget for the whole experiment,
an overrun the maintainer approved.

Only aggregate scores were read from either Held-out set; no miss or flag
was inspected. Both stay registered as Held-out. They have now decided one
verdict, so a later claim should use fresh data, as `eval-10` did after
ADR 0006.

## Follow-up (#40)

`correct` and the web UI resolve the default through the configuration
registry, with `qwen3.6-plus-p2` as the default and `flash-v4` as the tested
backup. `--model` on `correct` gets the same meaning it has on `eval` (prompt
v4 with that model). Done in #40.
