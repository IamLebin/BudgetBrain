# DESIGN.md — Per-Category Strategy

For each category: detection signal, solve strategy, model choice (if needed), and prompt notes.
Fill in the "measured accuracy / avg tokens" columns as you test locally — this file should
become your evidence log, not just a plan.

---

## 1. Factual Q&A
- **Detection**: question phrasing, no code/numbers-heavy content.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: `kimi-k2p7-code` first after live comparisons, with `minimax-m3` as fallback.
  Keep Gemma optional because its deployment availability is less predictable.
- **Prompt**: short system prompt instructing direct, concise answers — no chain-of-thought
  dump, since verbose reasoning costs tokens without being graded.
- Live validation: correct on capital/geography, concept explanation, and HTTP-status variants;
  `58-62` total tokens on tested prompts with reasoning disabled.

## 2. Math reasoning
- **Detection**: presence of numeric expressions, math keywords ("solve", "calculate",
  equations).
- **Strategy**: parse and evaluate safe arithmetic/percentage prompts locally first. Only
  fall back to Fireworks for word problems the parser can't structure.
- **Model (fallback only)**: cheapest allowed model; ask for the final numeric answer only.
- Validation: arithmetic, percentages, inventory changes, averages, one-variable equations,
  compound projections, discounts, sequential percentage changes, ratios, and weighted
  average-speed problems pass locally at 0 tokens. Unrecognized variants fall back safely.

## 3. Sentiment analysis
- **Detection**: short text, opinion/review-style content.
- **Strategy**: local lexicon-based scoring (small custom lexicon) first — this
  category is usually solvable locally with good accuracy.
- **Model (fallback)**: only if local confidence is low.
- Validation: positive, negative, negation, constrained neutral, and explicit mixed examples
  pass locally. Zero-hit or justification-required prompts fall back to Kimi, then MiniMax.

## 4. Summarization
- **Detection**: long input text, "summarize" instruction.
- **Strategy**: no reliable local method for good summaries — route to Fireworks.
- **Model**: `kimi-k2p7-code` first, with MiniMax fallback and a tight completion cap.
- **Prompt**: explicit target length/format to avoid the model over-generating.
- Live validation: exact sentence and bullet/word-limit variants pass at `73-101`
  total tokens per tested prompt with reasoning disabled.

## 5. Named Entity Recognition (NER)
- **Detection**: instruction asks to extract names/orgs/locations/dates.
- **Strategy**: local regex entity extraction for dates, URLs, emails, people, organizations,
  and simple preposition-marked locations. Consider spaCy only if the image budget allows and
  held-out NER needs more coverage.
- **Model (fallback)**: only if regex confidence is low or the expected schema is more complex.
- Validation: people, titled people, organizations, locations, absolute dates, and relative
  dates pass locally. Unresolved capitalized spans lower confidence and trigger model fallback.

## 6. Code debugging
- **Detection**: input contains a code block, possibly a traceback/error message.
- **Strategy**: local static analysis (`ast.parse` for syntax errors, simple linting) catches
  a subset for free. Logical bugs (wrong output, off-by-one, etc.) generally need the model.
- **Model**: use the code-oriented allowed model (`kimi-k2p7-code`) for anything static
  analysis can't resolve.
- **Prompt**: ask for the fix only, not an explanation, unless the output format requires
  an explanation — verbose responses cost tokens.
- Validation: missing-colon, max/min, `len(sequence)` index, and mutable-default bugs are
  repaired locally. Other logical/runtime bugs route to Kimi.

## 7. Logic puzzles
- **Detection**: structured constraint language ("if A then B", "exactly one of").
- **Strategy**: if the puzzle can be modeled as constraints, solve locally with
  `python-constraint` or `z3` — genuinely free and exact. This overlaps with your FYP1
  CSP-solver experience.
- **Model (fallback)**: unstructured/natural-language logic puzzles that don't cleanly map
  to a constraint model.
- Validation: truth-teller, one-to-one assignment, unique ordering, modus tollens, and simple
  universal-rule subsets pass locally. Other constraints route to MiniMax when needed.

## 8. Code generation
- **Detection**: instruction asks to write code for a spec.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: code-oriented allowed model (`kimi-k2p7-code`) first; compare against Gemma
  variants for the bonus prize if accuracy is close.
- **Prompt**: request code only, minimal comments, no extended explanation unless required
  by the output schema.
- Validation: functions, SQL, balanced brackets, and interval merging pass with Kimi reasoning
  disabled (`80-140` tokens in tested cases). Low reasoning is reserved for genuinely complex
  dynamic-programming, graph, parser, and concurrency prompts.

---

## Prompt engineering ground rules (apply to every Fireworks call)
- Keep system prompts under a few sentences — every token in the prompt counts against you,
  not just the completion.
- Set `max_tokens` per category to the smallest value that comfortably fits a correct answer;
  don't use one global generous cap for everything.
- No few-shot examples unless local testing shows accuracy actually requires them — each
  example is tokens spent on every single call.
- Ask for the answer in the exact output format required, so you don't need a second call
  to reformat.

## Testing protocol
1. Build a local test set per category (start with the FAQ/guide's examples if given, plus
   your own unseen variants — remember the real eval uses unseen inputs, so don't tune
   against a fixed set only).
2. Run `eval/run_local_eval.py` to get accuracy + token totals per category before touching
   the real Fireworks submission budget more than necessary.
3. Only escalate a category to a bigger/more expensive model if the cheaper one fails your
   local accuracy bar.

## Current validation snapshot
- Unit tests: `39/39` passing.
- Offline fixtures: baseline `8/8`, official practice `8/8`, held-out `16/16`, local champion
  `12/12` with zero model calls, and reasoning stress `8/8` with two model calls.
- Final live official-practice run: `8/8`, three Fireworks calls, `251` tokens.
- Final live held-out run: `16/16`, seven Fireworks calls, `483` tokens.
- Final reasoning-stress Docker run: `8/8`, two Fireworks calls, `262` tokens; the prior
  implementation used `1,497` tokens on the same prompts.
- Official accuracy gate: `80%`; real eval has `19` tasks, so clear target is at least
  `16/19` correct.
- Local zero-token paths currently cover arithmetic and structured word math, lexicon
  sentiment, regex NER, assignment/ordering/implication logic, and safe Python repairs.
- Live Fireworks validation works after normalizing shorthand model names to full Fireworks
  model IDs such as `accounts/fireworks/models/minimax-m3`.
- Docker image `budgetbrain-track1:champion-v2` builds and runs as `linux/amd64`; Docker content
  size is `45,525,751` bytes, comfortably under the `10GB` compressed-size limit.
- Grading environment: `4 GB RAM`, `2 vCPU`.
- Failure statuses from the updated guide to watch for: `PULL_ERROR`, `RUNTIME_ERROR`,
  `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`, `IMAGE_TOO_LARGE`,
  `ACCURACY_GATE_FAILED`.
