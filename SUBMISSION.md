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
- Offline all-category validation passes: `8/8`.
- Docker build passes for `linux/amd64`.
- Current local image: `budgetbrain-track1:local`, about `187MB`.

## Still Required Before Real Submission
- Set real Fireworks environment variables and run:

```bash
python -m eval.run_local_eval --real-fireworks
```

- Tag and push the verified image to a public registry:

```bash
docker tag budgetbrain-track1:local <registry>/<name>:<tag>
docker push <registry>/<name>:<tag>
```

- Confirm anonymous pull works from a clean environment.
