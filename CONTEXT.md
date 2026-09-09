# caption-checker

Post-hoc detection and correction of errors in auto-generated (ASR) caption
files. Given an existing SRT/VTT transcript, the tool flags spans that are
likely mistranscriptions and proposes corrections; a later stage judges those
proposals with an LLM and writes a corrected file.

## Language

### Transcript structure

**Cue**:
One timed subtitle entry: an index, a start and end time, and its text. The
unit the parser reads and the serializer writes.
_Avoid_: subtitle, caption line, entry.

**Word**:
A single whitespace-delimited token from a cue, carrying its position both
within its cue (character offset) and across the whole transcript (global
index). Position is tracked so a correction can be written back to the exact
place it came from.
_Avoid_: token (reserve "token" for the raw pre-position string).

**Span**:
The stretch of transcript text a flag covers — one Word, or several adjacent
Words when the suspected error crosses token boundaries.

**Sentence context**:
The full sentence a flagged span sits in, reconstructed across cue boundaries.
Attached to every flag and sent to the LLM as the surrounding evidence.
_Avoid_: snippet, window.

### Detection

**Detector**:
One anomaly test that scans the transcript and emits flags. Detectors are
independent and ordered cheapest-first; their flags are merged afterwards.

**Flag**:
A detector's claim that a span is a likely ASR error, with the reason it was
raised, zero or more candidates, and a confidence. Flags from different
detectors that cover overlapping spans are merged into one.
_Avoid_: hit, match, warning.

**Candidate**:
A proposed replacement string attached to a flag by the detector that raised
it. A flag may have none (the span looks wrong but no detector knows the fix),
one, or several ranked by phonetic similarity.
_Avoid_: suggestion (reserve that for the LLM's output), guess.

**Known-good word**:
A token treated as correct without further scrutiny: either common English
(above a word-frequency floor) or a member of the domain vocabulary. Detectors
flag tokens by contrast with these.
_Avoid_: valid word, dictionary word.

**Domain vocabulary**:
The curated list of correct-spelling technical terms for a subject area,
bundled as a default and extensible per run. Supplies both membership checks
and the phonetic index that "sounds like a real term" detectors match against.
_Avoid_: glossary (that is this document), term list, dictionary.

**Out-of-vocabulary (OOV)**:
A token that is neither a known-good word nor a domain-vocabulary term. The
cheapest error signal, and often the only one for a badly garbled span.

**Phonetic code**:
The Double Metaphone encoding of a token, used to decide two spellings "sound
the same". The shared key behind the phonetic detectors and the correction
cache's collision reasoning.
_Avoid_: sound key, hash.

**Confidence**:
A 0–1 estimate of how likely a flag or a correction is right. Detector
confidence and LLM confidence are separate numbers and are both kept on the
record; they are never blended into one score.

### Correction

**Correction**:
An LLM verdict on one flag: a replacement string and a confidence, or a
declaration that the span is not an error. Distinct from a candidate, which is
a detector's cheaper guess made before the LLM sees anything.
_Avoid_: fix, edit (use "edit" only for the act of writing it into the file).

**Not-an-error verdict**:
A correction that declines to change the span — the LLM judged the flag a false
positive. Surfaced to the reviewer as a pre-declined item they can override,
never silently dropped.

**Residue**:
The flags left to send to the LLM after the cheap resolutions — the internal-
match bypass — have been applied. Keeping the residue small is the main cost
lever.
_Avoid_: remainder, leftovers.

**Internal-match bypass**:
Applying a correction without an LLM call when a flagged span sounds exactly
like a known-good word used correctly elsewhere in the same transcript and one
candidate clearly wins. A cost optimisation, not a statement of higher
confidence: bypassed corrections still pass through review.
_Avoid_: shortcut, fast path.

**Decision cache**:
A persistent store mapping a garbled span plus a model identity to the
correction that model returned, so a recurring mistranscription — later in the
same file, or across a lecture series — is resolved without paying again. Keyed
on the span alone, not its surrounding context, on the assumption a given
garble resolves the same way every time.
_Avoid_: memo, lookup table.

**Sidecar**:
The record written next to a corrected file listing every flag and what became
of it — applied, rejected, not-an-error, bypassed, cached (applied from the
decision cache), or skipped after a parse failure. The corrected transcript
itself carries only accepted changes.
_Avoid_: log, manifest.
