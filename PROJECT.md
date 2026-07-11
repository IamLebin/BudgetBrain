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
1. **Accuracy gate** — must clear **80%**. The eval has **19 fixed tasks**, so the practical
   target is at least **16/19 correct**. Below the gate, the submission does not appear near
   the top of the leaderboard regardless of token count.
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
- Grading environment is constrained: **4 GB RAM, 2 vCPU**.
- Container maximum runtime is **10 minutes**; each model request is bounded below the
  guide's **30-second** response limit, and startup must complete within **60 seconds**.
- All answers must be in English.
- **No hardcoded/cached answers** — eval uses unseen input variants. Any lookup table keyed on
  exact input text is disqualifying, not just risky.
- Submissions are **rate-limited** — must test the full pipeline locally before each submit.
- The current rate limit is **10 submissions per hour per team**.
- Allowed Fireworks models (Track 1 only):
  - `minimax-m3`
  - `kimi-k2p7-code`
  - `gemma-4-31b-it`
  - `gemma-4-26b-a4b-it`
  - `gemma-4-31b-it-nvfp4`
- Bonus: separate prize pool for best use of a **Gemma** model via Fireworks in Track 1 — worth
  slotting a Gemma model into at least one category if accuracy/tokens are competitive.
- Gemma models are allowed but on-demand: a 404 from Fireworks may mean the model is not
  deployed. The cheapest recommended Gemma deployment still bills while idle, so do not rely
  on Gemma unless actively testing or intentionally competing for the Gemma prize.

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
- [x] Accuracy gate: 80%; eval has 19 fixed tasks, so aim for at least 16/19 correct.
- [x] Entrypoint is a file-based batch runner, not an HTTP endpoint:
  read `/input/tasks.json`, write `/output/results.json`.
- [x] Token score counts only tokens routed through `FIREWORKS_BASE_URL`; correct local
  answers count for accuracy with zero Fireworks tokens.

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

## Current competitive target (July 11, 2026, 20:18 team screenshot)
- Lowest visible nonzero result: `1,763` tokens at `84.2%` (`16/19`).
- Best visible nonzero 100% result: `2,520` tokens.
- Our v5 official-practice live run: `8/8`, `551` Fireworks tokens, with two tasks
  solved locally and six model calls.
- Our v5 held-out live run: `16/16` at `1,023` Fireworks tokens, with four tasks
  solved locally and twelve model calls.
- The v5 policy prioritizes semantic accuracy while retaining deterministic zero-token math
  and logic paths only when confidence is high.
- Published image: `docker.io/lebinbin/budgetbrain-track1:amd-act2-20260711-champion-v5`, built
  as a single `linux/amd64` manifest without provenance/SBOM attachments and verified by an
  empty-config anonymous manifest inspection and pull.
- Published digest:
  `sha256:ae93738ccde9c56c0f20ff2a9e13ea29e2917907727d6406dcd00b45c937bc9c`.

Leaderboard values can move before the deadline; the implementation target is therefore
`>=18/19` accuracy and comfortably below the best comparable nonzero result, not merely
clearing the gate.
