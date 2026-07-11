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

The latest local candidate is `budgetbrain-track1:champion-v3`, is `linux/amd64`, and has
Docker content size `45,527,782` bytes. V3 validation produced `8/8` official answers at
256 tokens, `16/16` held-out answers at 362 tokens (`366` inside Docker), and `8/8`
reasoning-stress answers at 262 tokens. The public v2 remains the submission image until v3
receives a new immutable tag.

Published immutable public submission image:

```text
lebinbin/budgetbrain-track1:amd-act2-20260710
lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v2
```

The first tag is the previous submission image. Use the optimized `20260711-champion-v2`
tag for scoring; its public `linux/amd64` manifest and anonymous pull are verified.
Optimized published digest:

```text
sha256:c287fee3ea4cc8d631c35734cef6ca315147ee7ee1a3ea22b87fa97bc0bdeb2a
```

Anonymous verification:

```bash
docker pull --platform linux/amd64 lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v2
```
