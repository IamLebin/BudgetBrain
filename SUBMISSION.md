# Submission Status

## Ready
- File contract implemented:
  - read `/input/tasks.json`
  - write `/output/results.json`
  - input `[{"task_id", "prompt"}]`
  - output `[{"task_id", "answer"}]`
- Category routing implemented because category is not supplied.
- Local zero-token paths implemented for verified math and high-confidence logic subsets;
  semantic categories use Fireworks during normal scoring.
- Fireworks fallback implemented with `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
  `ALLOWED_MODELS`.
- Unit validation passes: `73/73`.
- Offline baseline, official, held-out, local-champion, and reasoning fixtures pass: `8/8`,
  `8/8`, `16/16`, `17/17`, and `8/8`.
- Official-practice fixture passes `8/8`; strict stress passes `10/10`; the self-authored
  19-task accuracy audit passes `19/19` in offline contract mode.
- Final reasoning-stress Docker run passes `8/8` at `262` tokens, with six local answers.
- Docker build passes for `linux/amd64`.
- Current v7 release candidate: `budgetbrain-track1:v7-combined-local`, single `linux/amd64`
  manifest, `45,539,072` bytes by Docker's image content-size field.
- Public submission image:
  `docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v7`.
- Public image digest:
  `sha256:b055c0f05104736c936620d7687639a1a4e4d50abb6344e3f3070efe265198db`.
- The prior v2 tag is public, but its OCI index includes an `unknown/unknown` provenance child;
  the evaluator reported `PULL_ERROR` before any accuracy score was produced.
- V7 restores a batch-wide compatible retry if the scoring proxy rejects optional
  `reasoning_effort`; a container-level mock-proxy test confirms the first request retries bare
  and subsequent requests omit the field.
- V7 is built with provenance and SBOM disabled. Empty-config anonymous manifest inspection
  and `linux/amd64` pull both pass.
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
- Enter this exact full image reference in the Track 1 submission form:
  `docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v7`.
- Watch for failure statuses in the updated guide:
  `PULL_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`,
  `IMAGE_TOO_LARGE`, `ACCURACY_GATE_FAILED`.
