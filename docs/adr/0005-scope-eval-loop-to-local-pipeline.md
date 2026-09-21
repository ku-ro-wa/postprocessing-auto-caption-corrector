# Scope the eval loop to the pre-LLM pipeline; source ground truth from paired caption tracks

Qualitative eyeballing showed mixed results in the local detector pipeline and
decent (but single-model-tested) results in the LLM correction pass,
prompting a design pass on introducing a numeric eval loop without
overstating confidence from a tiny corpus or scaling LLM spend with
iteration count.

**Decision**: Target the free, local pre-LLM pipeline as the primary object
of iteration, gated by a Regression gate — separate flag precision/recall and
Cold flag rate numbers (never blended), scored against a small Scored corpus
and sanity-checked against a larger unscored Smoke corpus. Ground truth is
sourced by pairing videos' auto-generated tracks with a same-video Reference
caption rather than hand-transcribing audio; personal review time goes to
spot-checking disagreements against actual audio instead of exhaustive
verification. The LLM correction pass stays outside the tuning loop,
evaluated sparingly rather than re-run per iteration.

**Considered and rejected**: a scored accuracy benchmark as the primary
metric (5 videos can't support a generalization claim); a self-tuning
automated loop / LLM-as-judge (no validated judge exists yet, infra cost
unjustified); hand-transcribing audio for ground truth (doesn't scale past
the user's own viewing habits); actively vetting Reference caption provenance
before trusting it (often undeterminable from the file alone, and the
spot-check step already catches divergence regardless of cause).
