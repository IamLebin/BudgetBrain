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

The latest release candidate is `budgetbrain-track1:champion-v3-plain`, is a single
`linux/amd64` manifest, and has Docker content size `45,527,752` bytes. V3 validation produced
`8/8` official answers at 256 tokens, `16/16` held-out answers at 362 tokens (`366` inside
Docker), and `8/8` reasoning-stress answers at 262 tokens. It also passes `47/47` unit tests
plus adversarial classifier, sentiment-negation, and ambiguous-NER checks.

Published immutable public submission image:

```text
lebinbin/budgetbrain-track1:amd-act2-20260710
lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v2
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v3
```

The v2 tag is anonymously accessible, but its registry entry contains an extra provenance
manifest and the evaluator reported `PULL_ERROR`; do not resubmit that tag. Use the verified
single-manifest replacement:

```text
docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v3
```

The replacement passes anonymous manifest inspection and an anonymous `linux/amd64` pull.
Published replacement digest:

```text
sha256:ee7501852fe13c8bc8711f870d37871ee6d52a86b7074b38722ea10bf9c3e68a
```

Anonymous verification:

```bash
docker pull --platform linux/amd64 \
  docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v3
```
