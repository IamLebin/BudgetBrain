# TASKS.md — Execution Checklist

Current after the latest Discord/guide clarification.

## 0. Confirm the Contract
- [x] Participant Guide / Discord contract captured
- [x] Input: read `/input/tasks.json`
- [x] Output: write `/output/results.json`
- [x] Input shape: `[{"task_id": "...", "prompt": "..."}]`
- [x] Output shape: `[{"task_id": "...", "answer": "..."}]`
- [x] Category is not provided and must be inferred
- [x] Env vars: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`
- [x] Accuracy gate is `80%`
- [x] Real eval has `19` tasks, so target is at least `16/19`
- [x] Grading environment: `4 GB RAM`, `2 vCPU`

## 1. Scaffold and Runtime
- [x] Batch runner entrypoint in `app/main.py`
- [x] Router in `router/classify.py`
- [x] Fireworks client in `fireworks/client.py`
- [x] Local solvers in `solvers/`
- [x] No third-party runtime dependencies

## 2. Local Solvers
- [x] Arithmetic/percentage/word-arithmetic math baseline
- [x] Regex NER baseline
- [x] Lexicon sentiment baseline
- [x] Python syntax-error debugging baseline
- [x] Tiny logic baseline
- [x] Broaden math word-problem coverage with safety gates for multi-step prompts
- [x] Broaden NER entity/date handling and confidence fallback
- [x] Broaden logic puzzle coverage with one-to-one assignments
- [x] Broaden code-debugging to return corrected implementations for safe local repairs

## 3. Fireworks Client
- [x] Per-category model preferences
- [x] Per-category short system prompts
- [x] Per-category `max_tokens`
- [x] Token usage logging
- [x] Retry next allowed model if a selected model is unavailable
- [x] Avoid relying on Gemma first by default because Gemma is on-demand
- [x] Normalize shorthand model names to full Fireworks model IDs
- [x] Run live Fireworks eval with real credentials
- [x] Tune model choices and reasoning effort using official, held-out, and stress live runs
- [x] Enforce 25-second request timeout and retry only permitted runtime models
- [x] Normalize final math, sentiment, logic, and code output formats

## 4. Evaluation
- [x] Unit tests pass
- [x] Offline all-category fixture passes: `8/8`
- [x] Harness prints the real Track 1 gate: `16/19`
- [x] Live Fireworks fixture passes: `8/8`
- [x] Track live token usage per category
- [x] Import all 8 official practice tasks from the updated guide
- [x] Add 16 held-out variants and 8 reasoning stress tasks
- [x] Final live official Docker run passes `8/8` with `446` Fireworks tokens
- [x] Final live reasoning stress run passes `8/8` with `1,497` Fireworks tokens

## 5. Docker
- [x] Dockerfile builds as `linux/amd64`
- [x] Image is far under `10GB`
- [x] Container reads `/input/tasks.json` and writes `/output/results.json`
- [x] Final image `budgetbrain-track1:champion` is `linux/amd64`, `45,522,179` bytes
- [x] Push immutable image to public Docker Hub registry
- [x] Confirm anonymous `linux/amd64` manifest access and pull work

## 6. Pre-Submission
- [x] No API keys baked into image
- [x] No exact-answer cache tables for real eval
- [x] README run instructions match the file contract
- [x] Set Fireworks env vars in local `.env`
- [x] Runtime consumes organizer-injected Fireworks env vars without baking secrets into image
- [ ] Submit image and watch for guide failure statuses:
  `PULL_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`,
  `IMAGE_TOO_LARGE`, `ACCURACY_GATE_FAILED`
- [ ] Use registry download counter to confirm evaluator pulled the image

## 7. Publication
- [x] Public release explicitly approved and pushed as
  `lebinbin/budgetbrain-track1:amd-act2-20260710`
- [x] Public repository and anonymous pull verified
- [x] Published digest recorded as
  `sha256:bb74ac8bf2d2c089a236f578ef82e10e0a9316430fc8f2293bf23468badfedc6`
