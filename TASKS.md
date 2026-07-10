# TASKS.md — Execution Checklist

Ordered so an agent (or teammate) can pick this up and work top-down. Don't skip step 0 —
everything downstream depends on it being accurate.

## 0. Confirm the contract (do this before writing code)
- [x] Open the official Participant Guide and FAQ doc
- [x] Fill in the "Open items to confirm" checklist in `PROJECT.md`
- [x] Update the I/O contract placeholder in `ARCHITECTURE.md` with the real schema
- [x] Confirm Fireworks API base URL + auth method + env var names
- [ ] Confirm the accuracy gate threshold if published

## 1. Scaffold the repo
- [x] Create directory layout from `ARCHITECTURE.md`
- [x] `requirements.txt`: currently empty because the runner uses only the Python standard
      library plus Fireworks' OpenAI-compatible HTTP endpoint via `urllib`
- [x] Get a minimal batch entry point running locally

## 2. Build local solvers first (these are your free wins)
- [x] Math solver + unit tests with several unseen-style examples
- [x] NER solver regex baseline + unit tests
- [x] Sentiment solver + unit tests
- [x] Logic puzzle solver broader constraint-based subset + unit tests
- [x] For each: measure accuracy against a larger hand-built test set; only mark "local-solvable"
      in `DESIGN.md` once accuracy looks solid

## 3. Build the router
- [x] Category classifier (category is not given in input)
- [x] Solvability gate: local solver if confident + accurate, else escalate
- [x] Log every decision (category, chosen path, tokens if any) for debugging

## 4. Build the Fireworks client
- [x] Wrapper function with model name, prompt, max_tokens as parameters
- [x] Per-category prompt templates (start from `DESIGN.md` notes)
- [x] Token usage logging on every call
- [ ] Test with a real Fireworks key against factual Q&A, summarization, code debugging,
      code generation categories
      - Blocked locally: `FIREWORKS_API_KEY` is not set in this shell

## 5. Wire it together end-to-end
- [x] Full pipeline: input → classify → local or Fireworks → output in required format
- [x] Run all 8 categories through offline fixture with fake Fireworks fallback
- [ ] Run all 8 categories through it with real Fireworks credentials
- [ ] Fix any category that silently fails or times out

## 6. Local evaluation harness
- [x] `eval/run_local_eval.py`: runs a small smoke batch through the real entrypoint
- [x] Expand harness to report all-category pass/fail with offline fake Fireworks fallback
- [x] Confirm the expanded harness catches wrong answers
- [ ] Add real Fireworks token totals after live credentialed test
- [ ] Re-run after any prompt/model change — track results over time in `DESIGN.md`

## 7. Docker packaging
- [x] Write `Dockerfile` targeting `linux/amd64`
- [x] Build: `docker buildx build --platform linux/amd64 -t budgetbrain-track1:local --load .`
- [x] Confirm image size is comfortably under 10GB
      - Current local image: about `187MB`
      - Docker inspect: `linux/amd64`
- [x] Run the container locally and hit it the same way the evaluator will
- [ ] Push to a public registry (Docker Hub / GHCR) and confirm anonymous `docker pull` works
      from a clean machine/account

## 8. Pre-submission checklist
- [ ] No hardcoded answers anywhere — grep for suspicious lookup tables or literal test inputs
- [ ] Env vars (API keys) are not baked into the image
- [x] README explains how to run it, matching the actual entry point
- [ ] One final full local eval run passes the accuracy gate with a comfortable margin
      - Offline fixture passes; real Fireworks eval still pending
- [ ] Submit — remember submissions are rate-limited, so don't submit until you're confident

## 9. Stretch (if time remains)
- [ ] Try a Gemma model on categories where accuracy is close, for the Gemma bonus prize
- [ ] Try to push more categories from "Fireworks-required" to "local-solvable"
- [ ] Tighten prompts further to shave tokens without losing accuracy
