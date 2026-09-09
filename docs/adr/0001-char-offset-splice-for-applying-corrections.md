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
