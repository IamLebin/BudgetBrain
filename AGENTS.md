# BudgetBrain Agent Guide

## Project Context

BudgetBrain is an AMD Developer Hackathon ACT II Track 1 agent. The evaluator supplies prompts
without categories. The scoring container must:

- read `/input/tasks.json`
- write `/output/results.json`
- preserve every `task_id`
- return an `answer` string for every task
- run as a public `linux/amd64` Docker image under the Track 1 runtime contract

The score is accuracy-gated before token efficiency matters. Treat correctness, schema validity,
and reliable completion as higher priority than reducing Fireworks usage.

## Repository Shape

- `app/main.py`: Track 1 batch entrypoint.
- `app/agent.py`: routing and local-versus-Fireworks decision point.
- `solvers/`: narrow deterministic solvers for verifiable task patterns.
- `fireworks/client.py`: allowed-model routing, retries, validation, and token accounting.
- `Dockerfile.track1`: evaluator container; keep this independent from the demo container.
- `server.py` and `index.html`: Railway demo only.
- `eval/fixtures/` and `tests/`: regression coverage for routing, schema, and solver behavior.

## Modification Principles

1. Start from the evaluator's perspective: a plausible answer that misses a required detail,
   violates a requested format, times out, or writes invalid JSON is a failure.
2. Accuracy first. Do not replace a remote semantic answer with a local heuristic unless the
   trigger is narrow, the result is deterministic, and both positive and rejection cases are
   covered by tests.
3. Token optimization must be measurable. Prefer avoiding a remote call for a proven local
   result over shortening prompts blindly or lowering output limits in a way that drops facts.
4. Keep `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` runtime-configured.
   Never add credentials to source, docs, Docker images, fixtures, or logs.
5. Preserve the exact Track 1 file contract. Do not turn the batch runner into a web service or
   change its default `/input` and `/output` paths.
6. Avoid unrelated refactors, framework changes, new large dependencies, or style-only churn.
   Keep edits scoped to the requested behavior and the tests that prove it.
7. Keep README, SUBMISSION.md, media scripts, Docker tag, digest, and reported validation facts
   consistent whenever a release artifact is intentionally changed. Do not claim a score or
   verification that was not actually run.

## Verification

Run the smallest relevant checks first:

```bash
python3 -B -m unittest discover -s tests
python3 -B -m eval.run_local_eval --fixture eval/fixtures/strict_stress_20260712.json
python3 -B -m eval.run_local_eval --fixture eval/fixtures/accuracy_audit.json
```

For the existing broader offline suite:

```bash
sh scripts/validate_all.sh
```

Real Fireworks evaluation spends team credits. Run it only when needed to measure a proposed
accuracy or token change:

```bash
sh scripts/run_real_eval.sh --fixture eval/fixtures/official_practice.json
```

For evaluator-image changes, build and test the Track 1 container explicitly:

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -f Dockerfile.track1 -t budgetbrain-track1:local --load .
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1:local
```

Do not publish a Docker image, alter the submission form, or push external changes without the
user's explicit approval. Before recommending a new image, report the exact tested commit,
tag, digest, contract result, accuracy evidence, and token measurement.
