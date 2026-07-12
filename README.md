# BudgetBrain

AMD Developer Hackathon ACT II Track 1 agent.

The container follows the Participant Guide batch contract:

- reads `/input/tasks.json`
- writes `/output/results.json`
- input: `[{"task_id": "...", "prompt": "..."}]`
- output: `[{"task_id": "...", "answer": "..."}]`

Runtime environment:

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS`

Create local API config:

```bash
cp .env.example .env
```

Then edit `.env` and replace `replace_with_your_fireworks_key` with the real Fireworks key.
Do not commit `.env`.

Offline all-category validation:

```bash
python -m eval.run_local_eval
```

Official and expanded validations:

```bash
python3 -B -m eval.run_local_eval --fixture eval/fixtures/official_practice.json
python3 -B -m eval.run_local_eval --fixture eval/fixtures/held_out.json
python3 -B -m eval.run_local_eval --fixture eval/fixtures/reasoning_stress.json
```

Live Fireworks validation, which spends tokens:

```bash
sh scripts/run_real_eval.sh --fixture eval/fixtures/official_practice.json
```

Docker smoke run:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.track1 -t budgetbrain-track1 .
docker run --rm \
  --env-file .env \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1
```

Current verified local image:

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -f Dockerfile.track1 \
  -t budgetbrain-track1:local --load .
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1:local
```

The latest release candidate is `budgetbrain-track1:champion-v6-proxyfix`, is a single
`linux/amd64` manifest, and has Docker content size `45,530,446` bytes. V6 passes `56/56` unit
tests and every offline fixture. A real 19-task container run produced 19 nonempty, schema-valid
answers in about 19 seconds with 1,256 Fireworks tokens.

Published immutable public submission image:

```text
lebinbin/budgetbrain-track1:amd-act2-20260710
lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v2
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v3
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v5
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v6
```

The v2 tag is anonymously accessible, but its registry entry contains an extra provenance
manifest and the evaluator reported `PULL_ERROR`; do not resubmit that tag. V5 then ran but
scored `63.2%`. A controlled test matching the reported proxy behavior reproduced the failure:
the proxy rejects `reasoning_effort` with HTTP 400. V6 retries that same model without the
optional parameter and disables it for the rest of the batch. Use:

```text
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v6
```

The replacement passes anonymous manifest inspection and an anonymous `linux/amd64` pull.
Published replacement digest:

```text
sha256:de03483ec9b4f8b01394edd9a5e17f766d2e9813f5782225e46ba603c246d1b2
```

Anonymous verification:

```bash
docker pull --platform linux/amd64 \
  docker.io/lebinbin/budgetbrain-track1:amd-act2-20260712-champion-v6
```
