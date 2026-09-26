# Apply corrections by character-offset splice, not token rebuild

When an accepted correction is written into a cue, we replace the exact
substring identified by the flagged span (using the first Word's character
offset within the cue plus the span length) and leave the rest of the cue text
byte-for-byte unchanged. We do **not** re-tokenise the cue and re-join it from
Words with the span swapped.

Token rebuild is more uniform but drifts on whitespace, interior punctuation,
and multi-space runs, and the parser round-trip is only verified for
*unedited* parse→serialize. A splice guarantees that everything outside the
corrected span is preserved exactly, and it handles the split-word case
(replacing two or three adjacent tokens with one) as a single substring
operation. This is load-bearing for the corrected-file export, hence recording
it. Covered by a round-trip-with-one-edit test.

## Amendment (2026-09-26): spans across a Cue boundary

The Read-through (ADR 0006) proposes Corrections whose span crosses a Cue
boundary; dropping them threw away errors it had already found (7 on Scored,
3 on the Audited Held-out set, 55–62 on Earnings-21 `eval-10`). Such a span is
now applied as several splices. Every byte outside the span is left
unchanged, except the whitespace a cut leaves at the start of a later Cue:

- the whole replacement goes into the Cue the span starts in, from the first
  Word's offset (after the outer-punctuation trim) to the end of that Cue's
  part of the span;
- the span's remaining Words are cut from the following Cue(s), up to the
  trimmed end of its last Word. Punctuation after the span stays where it
  was, and the whitespace the cut leaves at the start of a Cue is removed;
- a Cue left with no text is removed, and the SRT serializer renumbers the
  rest. Timings never change.

Reviewers see the text of every Cue the span touches, and a Flag's
`cue_index` stays the first Cue's.
