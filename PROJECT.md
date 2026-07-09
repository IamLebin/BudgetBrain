# PROJECT.md — AMD Developer Hackathon ACT II · Track 1

## What we're building
A containerized general-purpose AI agent that solves tasks across 8 categories, optimized for
**minimum Fireworks AI token usage while staying above the accuracy gate**.

Task categories:
1. Factual Q&A
2. Math reasoning
3. Sentiment analysis
4. Summarization
5. Named Entity Recognition (NER)
6. Code debugging
7. Logic puzzles
8. Code generation

## Why this design, not "just call an LLM for everything"
Scoring works in two stages:
1. **Accuracy gate** — must clear a minimum accuracy threshold (exact number TBD, confirm from
   Participant Guide) or the submission doesn't get ranked at all.
2. **Token efficiency ranking** — among submissions that clear the gate, ranked by fewest
   Fireworks tokens consumed.

Only tokens spent through **Fireworks AI** count against the score. Local computation
(regex, sympy, spaCy, small local classifiers, rule-based solvers) is free. This means the
winning strategy is **NOT** "run a bigger/smarter model" — it's "avoid calling the model
whenever a deterministic method is accurate enough, and call the cheapest sufficient model
only when you must."

## Hard constraints (from hackathon rules)
- Submit a **Docker image**, publicly pullable, with a **linux/amd64** manifest.
- Image size **≤ 10GB**.
- **No hardcoded/cached answers** — eval uses unseen input variants. Any lookup table keyed on
  exact input text is disqualifying, not just risky.
- Submissions are **rate-limited** — must test the full pipeline locally before each submit.
- Allowed Fireworks models (Track 1 only):
  - `minimax-m3`
  - `kimi-k2p7-code`
  - `gemma-4-31b-it`
  - `gemma-4-26b-a4b-it`
  - `gemma-4-31b-it-nvfp4`
- Bonus: separate prize pool for best use of a **Gemma** model via Fireworks in Track 1 — worth
  slotting a Gemma model into at least one category if accuracy/tokens are competitive.

## Deadline
Hackathon runs **July 6–11, 2026**. Confirm exact submission cutoff + timezone from the
Event Schedule tab on lablab.ai — don't rely on a remembered date under time pressure.

## Open items to confirm before coding (from the official Participant Guide — behind login,
not something I could read directly)
- [x] Exact input/output JSON schema for the agent:
  - read `/input/tasks.json`
  - write `/output/results.json`
  - input shape: `[{"task_id": "...", "prompt": "..."}]`
  - output shape: `[{"task_id": "...", "answer": "..."}]`
- [x] Task category is not given explicitly and must be inferred from each prompt.
- [x] Required environment variables:
  - `FIREWORKS_API_KEY`
  - `FIREWORKS_BASE_URL`
  - `ALLOWED_MODELS`
- [ ] The accuracy gate threshold and how it's measured per category
- [x] Entrypoint is a file-based batch runner, not an HTTP endpoint:
  read `/input/tasks.json`, write `/output/results.json`.
- [ ] Per-request or per-run token accounting (does session context carry over?)

**Current implementation task:** keep the Docker runner aligned to the Participant Guide
contract above and avoid hardcoded/cached answers.

## Team roles (adjust to your headcount)
- **Router & local solvers** — classification logic + deterministic solutions (math, NER, logic)
- **Fireworks integration & prompts** — API client, per-category prompt design, token logging
- **Docker & test harness** — packaging, local eval harness, submission checklist

## Success criteria
- Passes accuracy gate on all 8 categories using held-out/unseen test variants
- Fewest possible Fireworks tokens per correct answer
- Docker image builds, runs, and is pullable clean on a fresh machine
- No category silently fails or times out
