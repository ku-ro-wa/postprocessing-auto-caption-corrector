# LLM correction stays manual and session-keyed; local detection does the automatic work

`correct` (the OpenRouter LLM pass) costs money per call; `check` (local
detectors) doesn't. In the web UI, `check` runs automatically on every
Transcript upload, but `correct` only ever runs from an explicit action,
and reads its OpenRouter key from the visiting Session rather than the
server's own `.env` — falling back to the server key only in today's
local/dev use. This keeps local detection as the layer that carries most of
the work (matching the original cost-control plan of a ~5-10% pre-filter
before any LLM call), and means a future public deployment doesn't silently
spend your API budget on visitors' automatic page loads.

## Considered Options

- **Automatic `correct` on upload, server-side key for everyone**: simplest
  UX, but every visitor's every upload burns your API budget with no gate.
- **Automatic `correct`, per-session key**: still fires without
  confirmation, which doesn't fit a pass that's meant to be the expensive
  last resort, not a default.
