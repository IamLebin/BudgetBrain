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
- **Strategy**: restructure an exact matching set of short source sentences into requested
  bullet points locally; route all abstractive, compressed, or ambiguous summaries to Fireworks.
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
  dates pass locally, including geographic prefixes and common organization suffixes.
  Unresolved capitalized spans lower confidence and trigger model fallback.

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
- Treat optional request parameters as proxy capabilities. If the injected judging proxy
  rejects `reasoning_effort` with HTTP 400, retry the same model once without it and omit the
  parameter for all later tasks in that batch.

## Testing protocol
1. Build a local test set per category (start with the FAQ/guide's examples if given, plus
   your own unseen variants — remember the real eval uses unseen inputs, so don't tune
   against a fixed set only).
2. Run `eval/run_local_eval.py` to get accuracy + token totals per category before touching
   the real Fireworks submission budget more than necessary.
3. Only escalate a category to a bigger/more expensive model if the cheaper one fails your
   local accuracy bar.

## Current validation snapshot
- Unit tests: `56/56` passing.
- Offline fixtures: baseline `8/8`, official practice `8/8`, held-out `16/16`, local champion
  `17/17`, reasoning stress `8/8`, adversarial routing `18/18`, and factual concepts `6/6`.
- V3 live official-practice run: `8/8`, three Fireworks calls, `256` tokens.
- V3 live held-out run: `16/16`, five Fireworks calls, `362` tokens; the v2 route used
  seven calls and `483` tokens.
- V3 Docker held-out run: `16/16`, five Fireworks calls, `366` tokens.
- Final reasoning-stress Docker run: `8/8`, two Fireworks calls, `262` tokens; the prior
  implementation used `1,497` tokens on the same prompts.
- Official accuracy gate: `80%`; real eval has `19` tasks, so clear target is at least
  `16/19` correct.
- Normal-scoring zero-token paths cover verified arithmetic, structured word math, and
  high-confidence assignment/ordering/implication logic. Semantic categories use Fireworks;
  their local solvers remain emergency fallbacks only.
- Remote answers are validated for Python syntax, summary bullet/word constraints, and
  requested NER JSON before accepting them; invalid outputs fall back to the next allowed model.
- Live Fireworks validation works after normalizing shorthand model names to full Fireworks
  model IDs such as `accounts/fireworks/models/minimax-m3`.
- The v5 evaluator result was `63.2%` (`12/19`). A controlled mock-proxy A/B run reproduced
  empty answers when every request included rejected `reasoning_effort`, then produced correct
  nonempty answers after v6 retried bare and disabled the field batch-wide.
- V6 reuses one Fireworks client across all tasks so the discovered proxy capability persists.
  A real 19-task container run returned 19 nonempty schema-valid answers in about 19 seconds at
  `1,256` Fireworks tokens.
- Docker image `budgetbrain-track1:champion-v6-proxyfix` builds and runs as a single
  `linux/amd64` manifest; Docker content size is `45,530,446` bytes, comfortably under the
  `10GB` compressed-size limit.
- Public v6 manifest inspection and an empty-config anonymous pull pass at digest
  `sha256:de03483ec9b4f8b01394edd9a5e17f766d2e9813f5782225e46ba603c246d1b2`.
- Adversarial routing tests prevent broad phrases from misclassifying factual, math, grammar,
  and non-sentiment classification prompts. Sentiment negation stops at sentence/contrast
  boundaries, and ambiguous capitalized NER spans fall back to Fireworks instead of guessing.
- Grading environment: `4 GB RAM`, `2 vCPU`.
- Failure statuses from the updated guide to watch for: `PULL_ERROR`, `RUNTIME_ERROR`,
  `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`, `IMAGE_TOO_LARGE`,
  `ACCURACY_GATE_FAILED`.
