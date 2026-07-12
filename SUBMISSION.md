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
- Unit validation passes: `56/56`.
- Offline baseline, official, held-out, local-champion, and reasoning fixtures pass: `8/8`,
  `8/8`, `16/16`, `17/17`, and `8/8`.
- V6 official-practice live run passes `8/8`.
- A real v6 19-task container run returned 19 nonempty schema-valid answers in about 19 seconds
  at `1,256` Fireworks tokens.
- Final reasoning-stress Docker run passes `8/8` at `262` tokens, with six local answers.
- Docker build passes for `linux/amd64`.
- Current v6 release candidate: `budgetbrain-track1:champion-v6-proxyfix`, single
  `linux/amd64` manifest, `45,530,446` bytes by Docker's image content-size field.
- Public submission image:
  `docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v6`.
- Public image digest:
  `sha256:de03483ec9b4f8b01394edd9a5e17f766d2e9813f5782225e46ba603c246d1b2`.
- The prior v2 tag is public, but its OCI index includes an `unknown/unknown` provenance child;
  the evaluator reported `PULL_ERROR` before any accuracy score was produced.
- V5 ran in the evaluator but scored `63.2%` (`12/19`). A controlled mock-proxy reproduction
  confirmed a matching compatibility failure: when `reasoning_effort` is rejected, v5 repeated
  the field for every candidate and could return an empty fallback.
- V6 retries that same model without `reasoning_effort` on HTTP 400, disables the optional field
  for the remaining batch, and shares one client across all tasks. The A/B reproduction passes.
- V6 is built with provenance and SBOM disabled. Empty-config anonymous manifest inspection and
  `linux/amd64` pull both pass.
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
  `docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v6`.
- Save this material v6 update once, then wait for scoring instead of repeatedly resubmitting.
- Watch for failure statuses in the updated guide:
  `PULL_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`,
  `IMAGE_TOO_LARGE`, `ACCURACY_GATE_FAILED`.
