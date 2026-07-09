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

Offline all-category validation:

```bash
python -m eval.run_local_eval
```

Live Fireworks validation, which spends tokens:

```bash
python -m eval.run_local_eval --real-fireworks
```

Docker smoke run:

```bash
docker buildx build --platform linux/amd64 -t budgetbrain-track1 .
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}" \
  -e ALLOWED_MODELS="$ALLOWED_MODELS" \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1
```

Current verified local image:

```bash
docker buildx build --platform linux/amd64 -t budgetbrain-track1:local --load .
docker run --rm --platform linux/amd64 \
  -v "$PWD/tests/sample_inputs:/input:ro" \
  -v "$PWD/output:/output" \
  budgetbrain-track1:local
```

The latest verified `budgetbrain-track1:local` image is `linux/amd64` and about `187MB`.
