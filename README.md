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
docker buildx build --platform linux/amd64 -t budgetbrain-track1 .
docker run --rm \
  --env-file .env \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1
```

Current verified local image:

```bash
docker buildx build --platform linux/amd64 -t budgetbrain-track1:local --load .
docker run --rm --platform linux/amd64 \
  --env-file .env \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1:local
```

The final verified image is `budgetbrain-track1:champion`, is `linux/amd64`, and has Docker
content size `45,522,179` bytes. Its latest official-practice container run produced 8 valid,
non-empty answers with 446 Fireworks tokens across three calls.

Published immutable public submission image:

```text
lebinbin/budgetbrain-track1:amd-act2-20260710
```

The image is public on Docker Hub and an anonymous `linux/amd64` pull has been verified.
Published digest:

```text
sha256:bb74ac8bf2d2c089a236f578ef82e10e0a9316430fc8f2293bf23468badfedc6
```

Anonymous verification:

```bash
docker pull --platform linux/amd64 lebinbin/budgetbrain-track1:amd-act2-20260710
```
