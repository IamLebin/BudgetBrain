# Submission Status

## Ready
- File contract implemented:
  - read `/input/tasks.json`
  - write `/output/results.json`
  - input `[{"task_id", "prompt"}]`
  - output `[{"task_id", "answer"}]`
- Category routing implemented because category is not supplied.
- Local zero-token paths implemented for safe subsets of math, sentiment, NER, logic, and
  code debugging.
- Fireworks fallback implemented with `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
  `ALLOWED_MODELS`.
- Unit validation passes: `44/44`.
- Offline baseline, official, held-out, local-champion, and reasoning fixtures pass: `8/8`,
  `8/8`, `16/16`, `17/17`, and `8/8`.
- Final official-practice live run passes `8/8`, with Fireworks on `3/8` tasks and local
  solvers on `5/8`.
- Latest v3 official live token total: `256` across three calls.
- V3 held-out live suite passes `16/16` at `362` tokens; the Docker run uses `366` tokens.
- Final reasoning-stress Docker run passes `8/8` at `262` tokens, with six local answers.
- Docker build passes for `linux/amd64`.
- Current v3 local candidate: `budgetbrain-track1:champion-v3`, `linux/amd64`, `45,527,782`
  bytes by Docker's image content-size field.
- Public submission image:
  `lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v2`.
- Public image digest:
  `sha256:c287fee3ea4cc8d631c35734cef6ca315147ee7ee1a3ea22b87fa97bc0bdeb2a`.
- Anonymous `linux/amd64` manifest access and pull both pass from an empty Docker config.
- Official accuracy gate: `80%`.
- Real eval has `19` tasks, so target is at least `16/19` correct.
- Grading environment: `4 GB RAM`, `2 vCPU`.
- Fireworks model routing avoids Gemma first by default because Gemma is on-demand and may
  return 404 unless deployed.
- Shorthand model names are normalized to full Fireworks IDs, e.g.
  `accounts/fireworks/models/minimax-m3`.
- Judging-proxy model IDs remain exactly as injected by `ALLOWED_MODELS`; direct Fireworks
  shorthand expansion is local-development-only.
- Requests time out at 25 seconds, keeping the 19-task worst-case timeout budget under the
  official 10-minute runtime limit.

## Still Required Before Real Submission
- Push v3 under a new immutable public tag and verify anonymous pull.
- Enter the immutable image reference in the Track 1 submission form; do not use a mutable
  local-only tag.
- Watch for failure statuses in the updated guide:
  `PULL_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`,
  `IMAGE_TOO_LARGE`, `ACCURACY_GATE_FAILED`.
