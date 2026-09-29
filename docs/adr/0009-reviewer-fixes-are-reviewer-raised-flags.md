# A reviewer's fix on an unflagged span is a reviewer-raised Flag

A reviewer who hears an error nothing flagged fixes it by editing the Cue's
text in the web UI's Cue view. The edit is diffed at Word level against the
Cue as it would export, and each changed stretch becomes a **Flag raised by
the reviewer** with an accepted Review Decision carrying their text. We
stretched the Flag's definition (already stretched once for the Read-through)
rather than add a second record type, so Review Decisions, the Flags list,
undo (reject) and Export's `(Flag, text)` splice (ADR 0001) all work
unchanged.

## Considered Options

- **A separate "Reviewer edit" record** that Export applies alongside
  accepted Flags. It keeps Flags a purely machine claim, but gives Export a
  second input, needs its own listing and undo, and needs overlap rules
  between two kinds of record. Rejected.
- **Whole-Cue text replacement on Export.** Easiest to build, but it
  conflicts with offset splices of accepted Flags in the same Cue. Rejected;
  the reviewer still edits whole Cues, but only the diff is stored.

## Consequences

- `apply_corrections` has no overlap guard, so no two accepted spans may
  overlap. An edit inside an existing Flag's span updates that Flag's Review
  Decision. An edit that partly overlaps one is merged into a single
  reviewer-raised Flag, and the old Flag is rejected as superseded.
- An inserted or deleted Word widens the span to include a neighbouring Word,
  so every reviewer-raised Flag covers at least one Word. A punctuation-only
  change is widened the same way, because Export trims a span's outer
  punctuation. The one exception is a change at the very start or end of a
  Cue, which can't be saved, and the reviewer is told so. Reviewer-raised
  Flags never cross a Cue boundary.
- The Read-through never sees reviewer-raised Flags. They are not sent as
  hints, so they are never judged or widened (widening resets an accepted
  Review Decision). A Flag the Read-through finds that overlaps a
  reviewer-raised Flag is dropped. Sending the reviewer's fixes to the model
  as context would change the frozen Read-through configuration (ADR 0007),
  so that is a separate experiment.
- Reviewer-raised Flags are not detections. They are left out of "found by
  the Read-through" counts and never feed the Scored corpus or Audited
  transcripts.
