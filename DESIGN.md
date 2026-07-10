# DESIGN.md — Per-Category Strategy

For each category: detection signal, solve strategy, model choice (if needed), and prompt notes.
Fill in the "measured accuracy / avg tokens" columns as you test locally — this file should
become your evidence log, not just a plan.

---

## 1. Factual Q&A
- **Detection**: question phrasing, no code/numbers-heavy content.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: start with `minimax-m3` unless live testing shows a better accuracy/token tradeoff.
  Keep Gemma as an optional fallback/bonus path only when deployed, because Gemma is on-demand.
- **Prompt**: short system prompt instructing direct, concise answers — no chain-of-thought
  dump, since verbose reasoning costs tokens without being graded.
- Live validation: correct on capital/geography, concept explanation, and HTTP-status variants;
  `149-174` total tokens on short tested prompts with reasoning disabled.

## 2. Math reasoning
- **Detection**: presence of numeric expressions, math keywords ("solve", "calculate",
  equations).
- **Strategy**: parse and evaluate safe arithmetic/percentage prompts locally first. Only
  fall back to Fireworks for word problems the parser can't structure.
- **Model (fallback only)**: cheapest allowed model; ask for the final numeric answer only.
- Validation: arithmetic, percentages, inventory changes, averages, one-variable equations,
  compound projections, and discounts pass locally at 0 tokens. Multi-stage percentages,
  ratios, and average-speed problems intentionally fall back to MiniMax with low reasoning.

## 3. Sentiment analysis
- **Detection**: short text, opinion/review-style content.
- **Strategy**: local lexicon-based scoring (small custom lexicon) first — this
  category is usually solvable locally with good accuracy.
- **Model (fallback)**: only if local confidence is low.
- Validation: positive, negative, negation, constrained neutral, and explicit mixed examples
  pass locally. Zero-hit or justification-required prompts fall back to MiniMax.

## 4. Summarization
- **Detection**: long input text, "summarize" instruction.
- **Strategy**: no reliable local method for good summaries — route to Fireworks.
- **Model**: start with `minimax-m3`; cap `max_tokens` tightly to the expected summary length.
  Try Gemma only if deployed and live eval shows it improves accuracy enough to justify risk/cost.
- **Prompt**: explicit target length/format to avoid the model over-generating.
- Live validation: exact sentence and bullet/word-limit variants pass at roughly `172-210`
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
- Validation: missing-colon and simple max/min implementation bugs are repaired locally.
  Logical/runtime bugs route to Kimi; tested fixes used `73-89` total tokens.

## 7. Logic puzzles
- **Detection**: structured constraint language ("if A then B", "exactly one of").
- **Strategy**: if the puzzle can be modeled as constraints, solve locally with
  `python-constraint` or `z3` — genuinely free and exact. This overlaps with your FYP1
  CSP-solver experience.
- **Model (fallback)**: unstructured/natural-language logic puzzles that don't cleanly map
  to a constraint model.
- Validation: truth-teller and one-to-one assignment subsets pass locally. Other constraints
  route to MiniMax with low reasoning only when needed.

## 8. Code generation
- **Detection**: instruction asks to write code for a spec.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: code-oriented allowed model (`kimi-k2p7-code`) first; compare against Gemma
  variants for the bonus prize if accuracy is close.
- **Prompt**: request code only, minimal comments, no extended explanation unless required
  by the output schema.
- Validation: simple functions and SQL pass with Kimi reasoning disabled (`78-129` tokens in
  tested cases); complex interval-style tasks retain low reasoning for accuracy.

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
- Unit tests: `33/33` passing.
- Offline fixtures: baseline `8/8`, official practice `8/8`, held-out `16/16`, reasoning
  stress `8/8`.
- Final live official-practice Docker run: `8/8`, three Fireworks calls and five local answers.
- Latest final Docker token use: factual `149`, summarization `210`, code generation `87`,
  total `446` (`444-446` observed across final runs).
- Final live reasoning stress: `8/8`, all eight model-routed, total `1,497` tokens.
- Held-out live suite: every unambiguous item was correct; one deliberately borderline
  `as scheduled` sentiment example varied between `neutral` and `positive` across runs.
- Official accuracy gate: `80%`; real eval has `19` tasks, so clear target is at least
  `16/19` correct.
- Local zero-token paths currently passing fixture coverage:
  - arithmetic math
  - lexicon sentiment
  - regex NER baseline
  - Python syntax-error debugging
- Live Fireworks validation works after normalizing shorthand model names to full Fireworks
  model IDs such as `accounts/fireworks/models/minimax-m3`.
- Docker image `budgetbrain-track1:champion` builds and runs as `linux/amd64`; Docker content
  size is `45,522,179` bytes, comfortably under the `10GB` compressed-size limit.
- Grading environment: `4 GB RAM`, `2 vCPU`.
- Failure statuses from the updated guide to watch for: `PULL_ERROR`, `RUNTIME_ERROR`,
  `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`, `IMAGE_TOO_LARGE`,
  `ACCURACY_GATE_FAILED`.
