# TASKS.md — Execution Checklist

Ordered so an agent (or teammate) can pick this up and work top-down. Don't skip step 0 —
everything downstream depends on it being accurate.

## 0. Confirm the contract (do this before writing code)
- [ ] Open the official Participant Guide and FAQ doc
- [ ] Fill in the "Open items to confirm" checklist in `PROJECT.md`
- [ ] Update the I/O contract placeholder in `ARCHITECTURE.md` with the real schema
- [ ] Confirm Fireworks API base URL + auth method + env var names
- [ ] Confirm the accuracy gate threshold if published

## 1. Scaffold the repo
- [ ] Create directory layout from `ARCHITECTURE.md`
- [ ] `requirements.txt`: fastapi/uvicorn (or whatever the real entrypoint needs), sympy,
      spacy (+ `en_core_web_sm` download), python-constraint or z3, requests/openai-sdk
      for Fireworks calls, pytest
- [ ] Get a minimal "hello world" version of the entry point running locally

## 2. Build local solvers first (these are your free wins)
- [ ] Math solver + unit tests with several unseen-style examples
- [ ] NER solver (spaCy) + unit tests
- [ ] Sentiment solver + unit tests
- [ ] Logic puzzle solver (constraint-based subset) + unit tests
- [ ] For each: measure accuracy against a hand-built test set; only mark "local-solvable"
      in `DESIGN.md` once accuracy looks solid

## 3. Build the router
- [ ] Category classifier (or pass-through if category is given in input)
- [ ] Solvability gate: local solver if confident + accurate, else escalate
- [ ] Log every decision (category, chosen path, tokens if any) for debugging

## 4. Build the Fireworks client
- [ ] Wrapper function with model name, prompt, max_tokens as parameters
- [ ] Per-category prompt templates (start from `DESIGN.md` notes)
- [ ] Token usage logging on every call
- [ ] Test against factual Q&A, summarization, code debugging, code generation categories

## 5. Wire it together end-to-end
- [ ] Full pipeline: input → classify → local or Fireworks → output in required format
- [ ] Run all 8 categories through it with sample inputs
- [ ] Fix any category that silently fails or times out

## 6. Local evaluation harness
- [ ] `eval/run_local_eval.py`: runs a batch of test inputs across all categories, reports
      per-category accuracy + total/average tokens
- [ ] Confirm the harness catches wrong answers (don't just check "did it return something")
- [ ] Re-run after any prompt/model change — track results over time in `DESIGN.md`

## 7. Docker packaging
- [ ] Write `Dockerfile` targeting `linux/amd64`
- [ ] Build: `docker buildx build --platform linux/amd64 -t <name> .`
- [ ] Confirm image size is comfortably under 10GB
- [ ] Run the container locally and hit it the same way the evaluator will
- [ ] Push to a public registry (Docker Hub / GHCR) and confirm anonymous `docker pull` works
      from a clean machine/account

## 8. Pre-submission checklist
- [ ] No hardcoded answers anywhere — grep for suspicious lookup tables or literal test inputs
- [ ] Env vars (API keys) are not baked into the image
- [ ] README explains how to run it, matching the actual entry point
- [ ] One final full local eval run passes the accuracy gate with a comfortable margin
- [ ] Submit — remember submissions are rate-limited, so don't submit until you're confident

## 9. Stretch (if time remains)
- [ ] Try a Gemma model on categories where accuracy is close, for the Gemma bonus prize
- [ ] Try to push more categories from "Fireworks-required" to "local-solvable"
- [ ] Tighten prompts further to shave tokens without losing accuracy
