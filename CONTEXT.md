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
A claim that a span is a likely ASR error, with the reason it was raised,
zero or more candidates, and a confidence. Raised by a Detector, or by the
Read-through for an error no Detector raised; every Flag records which.
Flags from different detectors that cover overlapping spans are merged into
one.
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
bundled as a default and extended per run by Priming terms. Supplies both membership checks
and the phonetic index that "sounds like a real term" detectors match against.
_Avoid_: glossary (that is this document), term list, dictionary.

**Priming terms**:
Terms supplied alongside one transcript -- a speaker's name, a product, a
course's jargon -- that join the Domain vocabulary for that run and are also
given to the LLM directly. What a video title or a call's company name is to
the system. In the web UI they are entered with the `correct` run, after the
upload's local scan, so they reach only the LLM.
_Avoid_: custom vocab, hints, keywords.

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

**Read-through**:
An LLM stage that reads the whole transcript in chunks, with the detectors'
Flags and candidates and the Priming terms as hints, and returns Flags with
their Corrections in one pass -- including errors no Detector raised. Not a
Detector: it depends on their output rather than running independently.
The default LLM pass of both `correct` and the web UI; the older per-flag
pass, which sends only the Residue, is its opt-out (`--per-flag`, CLI only).
_Avoid_: LLM detector, LLM scan, rewrite.

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

### Evaluation

**Regression gate**:
A check that compares the local detector pipeline's output on the Scored
corpus against expected should-flag / should-not-flag cases, run whenever
detector thresholds change. Reports flag precision/recall and the Cold flag
rate as separate numbers — a floor that catches new false positives and
negatives, never a claim of general accuracy.
_Avoid_: benchmark, accuracy score.

**Cold flag**:
An OOV Flag matching neither the curated domain vocabulary nor that
transcript's own doc vocabulary — the system has no learned context for the
span at all. Tracked separately from other flags because its rate signals
both reviewer-facing false-positive risk on unfamiliar content and LLM cost
exposure, since a cold flag always becomes LLM residue.
_Avoid_: unknown flag, unmatched flag.

**Reference caption**:
A same-video caption track that isn't labeled auto-generated, formerly used as
an approximate answer key for the Regression gate (replaced by Audited
transcripts). Not verified: it may be a
genuine post-hoc transcript, a lightly touched-up auto pass, or a
pre-production script/TTS source that never touched the finished audio — a
mismatch against it is spot-checked against the actual audio before being
scored as a false positive or negative.
_Avoid_: ground truth, clean transcript, manual transcript (all overstate a
confidence this hasn't earned).

**Audited transcript**:
An auto-generated transcript whose errors a person has listed exhaustively by
listening to the audio against it. The closest thing this project has to
ground truth; supersedes Reference captions as the source of Scored corpus
entries.
_Avoid_: manual transcript, gold transcript.

**Scored corpus**:
Should-flag / should-not-flag cases drawn from Audited transcripts, small by
necessity, used to compute the Regression gate's numbers.
_Avoid_: eval set, golden set.

**Auto-labelled corpus**:
Cases derived without human review by aligning an ASR system's output against
a professional verbatim reference transcript of the same audio. Large and
varied but noisy, and its errors come from that ASR system rather than
YouTube's -- trusted for relative comparisons, not absolute numbers.
_Avoid_: synthetic corpus, external benchmark.

**Dev set**:
The transcripts a change may be tuned against. Everything the Regression gate
currently scores is Dev set.
_Avoid_: training set.

**Held-out set**:
Transcripts never tuned against, scored only to estimate how a frozen version
of the system generalises. Looking at one to motivate a change moves it to the
Dev set.
_Avoid_: test set (too easily confused with `tests/`).

**Flag-level precision**:
The share of all emitted Flags that touch a known error. Only meaningful where
errors are listed exhaustively (Audited transcripts, Auto-labelled corpora);
distinct from the Regression gate's case precision over hand-picked
should-not-flag spans.

**Smoke corpus**:
A larger, unscored set of transcripts run through detection only (no LLM
correction), used to catch a threshold overfit to the Scored corpus — flagged
by a spike in flag rate or an unfamiliar class of flags, not by comparison to
any answer key.
_Avoid_: test set (too easily confused with `tests/`).

### Web review UI

**Transcript**:
A caption file (SRT/VTT), parsed into Cues, uploaded through the web review
UI (`caption-checker serve`) and belonging to the Session that uploaded it.
Everything else on this page (Flags, Corrections) attaches to one. Its Flags
are the local scan's, plus any the Read-through found when `correct` ran.
_Avoid_: upload, file, document.

**Session**:
An anonymous, cookie-identified scope isolating which Transcripts, Review
Decisions, and OpenRouter API key belong to one browser. No login or account
sits behind it — it's an isolation boundary, not an identity.
_Avoid_: user, account.

**Review Decision**:
A reviewer's disposition on one Flag in the web UI: pending, accepted, or
rejected, with optional edited replacement text overriding the Flag's
Correction. Export reads these to decide what changes make it into the
corrected file. Distinct from the CLI `correct` command's own interactive
accept/skip/edit flow, which never persists a decision between runs.
_Avoid_: verdict (that's the LLM's Correction; a Review Decision is the
reviewer's response to it).

**Export**:
The corrected SRT/VTT the web UI produces by applying every accepted Review
Decision's text back into the original Cues. Flags left pending or rejected
keep their original text. Distinct from the CLI `correct` command's `-o`
output, which is written directly from its own interactive review.
