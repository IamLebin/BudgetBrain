# ARCHITECTURE.md — Track 1 Agent

## High-level flow

```
Input task
   │
   ▼
[1] Category Classifier  ──► which of the 8 categories is this?
   │
   ▼
[2] Solvability Check    ──► can a local/deterministic method solve this
   │                          accurately, with no API call?
   ├── YES ──► [3a] Local Solver ──────────────► Answer (0 Fireworks tokens)
   │
   └── NO  ──► [3b] Confidence-Gated Fireworks Call
                     │
                     ▼
              pick cheapest allowed model for this category
                     │
                     ▼
              minimal prompt, low max_tokens ──► Answer
```

## Components

### 1. Category Classifier (`router/classify.py`)
- Input: raw task text (+ any metadata the I/O contract provides)
- Output: one of the 8 category labels + confidence score
- Implementation options, cheapest first:
  - If the input schema **already tells you the category** (check the Participant Guide),
    skip classification entirely — don't spend compute re-deriving known information.
  - Else: keyword/pattern heuristics first (fast, free, surprisingly effective for categories
    like "code debugging" — presence of a traceback/code block — or "math reasoning" —
    presence of numbers + operators).
  - Fallback: a small local model only if heuristics are unreliable — but this local step
    is dev-only and doesn't affect score either way.

### 2. Local Solvers (`solvers/`)
Deterministic, zero-token paths. Only use these where accuracy is genuinely reliable —
a wrong "free" answer still fails the accuracy gate.

| Category | Local strategy | Library |
|---|---|---|
| Math reasoning | Arithmetic, percentages, ratios, weighted speed | pure Python `ast`, `fractions` |
| NER | Regex extraction for dates, URLs, emails, capitalized people/orgs/locations | pure Python `re` |
| Sentiment | Lexicon/rule-based scoring | custom lexicon |
| Logic puzzles | Assignments, ordering, implication, truth puzzles | custom parser |
| Factual Q&A | Usually **not** locally solvable — route to model | — |
| Summarization | Usually **not** locally solvable — route to model | — |
| Code debugging | Syntax, extrema, index, mutable-default repairs | pure Python `ast`, `re` |
| Code generation | Not locally solvable — route to model | — |

### 3. Fireworks Client (`fireworks/client.py`)
- Wraps calls to the Fireworks AI API (OpenAI-compatible chat completions endpoint).
- One function per category, each with:
  - A minimal, tuned system prompt (short — every token counts)
  - The cheapest allowed model that clears accuracy for that category in your local tests
  - A conservative `max_tokens` cap
- Model routing prefers non-Gemma defaults first because Gemma is on-demand and can return
  404 if not deployed. Gemma remains an allowed fallback/bonus path when explicitly available.
- Live token tests route concise factual, summarization, sentiment, NER, and code tasks to
  `kimi-k2p7-code`; complex math and logic retain `minimax-m3` as the accuracy-first fallback.
- If a selected model returns an availability-style HTTP error, try the next allowed model
  instead of failing the whole task.
- Preserve the exact model IDs injected by `ALLOWED_MODELS` when using the judging proxy.
  Shorthand IDs are expanded only for direct local calls to `api.fireworks.ai`.
- Disable reasoning for short factual, summarization, sentiment, NER, and simple code tasks;
  retain low reasoning for complex math, logic, debugging, and code generation.
- Bound every HTTP call to 25 seconds and recover from unavailable, rate-limited, server-error,
  empty-content, or malformed model responses without calling a model outside the allow-list.
- Normalize Markdown code fences, sentiment labels, yes/no conclusions, and math final-answer
  lines before writing the result.
- Central token counter/logger so you can see running token spend during local testing.

### 4. Batch Entry Point (`app/main.py`)
The Participant Guide contract is a file-based batch runner:

```python
# Read:  /input/tasks.json
# Input: [{"task_id": str, "prompt": str}]
# Write: /output/results.json
# Output: [{"task_id": str, "answer": str}]
```

Task category is not supplied in the input, so `router/classify.py` infers it from the
prompt. The runner should continue producing one output row per input row even if one task
fails, using a short error-safe fallback answer rather than crashing the entire batch.

### 5. Docker (`Dockerfile`)
- Base image: slim Python (e.g. `python:3.11-slim`) — no need for CUDA/ROCm base images or
  local model weights, since local inference isn't scored. Keep the image small and simple.
- Explicit `--platform linux/amd64` build target.
- No secrets baked into the image — Fireworks API key comes from environment variable at
  runtime, not hardcoded.
- Current implementation has no third-party Python dependencies, keeping the image small.
- The final Dockerfile performs no package-install step, so builds do not depend on a package
  index and the runtime image stays near `45.5 MB` by Docker's content-size measurement.
- Grading resources are 4 GB RAM and 2 vCPU, so any local model strategy must fit comfortably
  inside that budget. The current implementation uses rule-based local solvers instead of
  bundled local LLM weights.

## Directory layout

```
.
├── PROJECT.md
├── ARCHITECTURE.md
├── DESIGN.md
├── TASKS.md
├── Dockerfile
├── requirements.txt
├── app/
│   └── main.py
├── router/
│   └── classify.py
├── solvers/
│   ├── math_solver.py
│   ├── ner_solver.py
│   ├── sentiment_solver.py
│   └── logic_solver.py
├── fireworks/
│   └── client.py
├── tests/
│   ├── sample_inputs/
│   └── test_categories.py
└── eval/
    └── run_local_eval.py   # local harness — run this before every real submission
```

## Environment variables
```
FIREWORKS_API_KEY=
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4
```

## Token accounting
Log every Fireworks call's `usage.total_tokens` plus prompt/completion breakdown to stderr
during testing, so per-category and total spend are visible without changing
`/output/results.json`.

## Timing budget
- Startup: Python-only imports, comfortably under 60 seconds.
- Per Fireworks request: 25-second socket timeout.
- Worst-case sequential timeout budget for 19 tasks: `19 x 25 = 475` seconds, below the
  10-minute container limit even before considering that local tasks return immediately.
