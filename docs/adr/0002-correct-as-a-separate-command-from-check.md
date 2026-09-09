# `correct` is a separate command from `check`

Detection and correction are two CLI verbs, not one command with a `--fix`
flag. `check` stays read-only: it flags, prints a report or JSON, exits 1 when
anything is flagged, and never spends money or touches a file — safe for CI and
scripting. `correct` runs the same detection internally, then adds the LLM
pass, interactive review, and file output, and requires an explicit `-o`
destination.

The alternative (one command, output shape switched by a flag) was rejected
because the two paths have genuinely different contracts — side-effect-free and
free vs. file-writing, interactive, and billed — and collapsing them makes the
money-spending path easy to trigger by accident.
