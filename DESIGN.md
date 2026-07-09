# DESIGN.md — Per-Category Strategy

For each category: detection signal, solve strategy, model choice (if needed), and prompt notes.
Fill in the "measured accuracy / avg tokens" columns as you test locally — this file should
become your evidence log, not just a plan.

---

## 1. Factual Q&A
- **Detection**: question phrasing, no code/numbers-heavy content.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: start with the cheapest allowed model (`gemma-4-26b-a4b-it`); escalate to a
  larger model only for questions your local eval shows it getting wrong.
- **Prompt**: short system prompt instructing direct, concise answers — no chain-of-thought
  dump, since verbose reasoning costs tokens without being graded.
- Measured accuracy: ___  Avg tokens: ___

## 2. Math reasoning
- **Detection**: presence of numeric expressions, math keywords ("solve", "calculate",
  equations).
- **Strategy**: parse and evaluate safe arithmetic/percentage prompts locally first. Only
  fall back to Fireworks for word problems the parser can't structure.
- **Model (fallback only)**: cheapest allowed model; ask for the final numeric answer only.
- Smoke result: local arithmetic examples pass with 0 Fireworks tokens. Held-out accuracy TBD.

## 3. Sentiment analysis
- **Detection**: short text, opinion/review-style content.
- **Strategy**: local lexicon-based scoring (small custom lexicon) first — this
  category is usually solvable locally with good accuracy.
- **Model (fallback)**: only if local confidence is low.
- Smoke result: simple positive/negative examples pass with 0 Fireworks tokens. Held-out
  accuracy TBD.

## 4. Summarization
- **Detection**: long input text, "summarize" instruction.
- **Strategy**: no reliable local method for good summaries — route to Fireworks.
- **Model**: test whether a smaller model produces acceptable summaries before defaulting
  to a bigger one; cap `max_tokens` tightly to the expected summary length.
- **Prompt**: explicit target length/format to avoid the model over-generating.
- Measured accuracy: ___  Avg tokens: ___

## 5. Named Entity Recognition (NER)
- **Detection**: instruction asks to extract names/orgs/locations/dates.
- **Strategy**: local regex entity extraction for dates, URLs, emails, people, organizations,
  and simple preposition-marked locations. Consider spaCy only if the image budget allows and
  held-out NER needs more coverage.
- **Model (fallback)**: only if spaCy's entity types don't match the expected schema.
- Smoke result: simple entity extraction examples pass with 0 Fireworks tokens. Held-out
  accuracy TBD.

## 6. Code debugging
- **Detection**: input contains a code block, possibly a traceback/error message.
- **Strategy**: local static analysis (`ast.parse` for syntax errors, simple linting) catches
  a subset for free. Logical bugs (wrong output, off-by-one, etc.) generally need the model.
- **Model**: use a code-oriented allowed model (`kimi-k2p7-code`) for anything static
  analysis can't resolve.
- **Prompt**: ask for the fix only, not an explanation, unless the output format requires
  an explanation — verbose responses cost tokens.
- Measured accuracy: ___  Avg tokens: ___

## 7. Logic puzzles
- **Detection**: structured constraint language ("if A then B", "exactly one of").
- **Strategy**: if the puzzle can be modeled as constraints, solve locally with
  `python-constraint` or `z3` — genuinely free and exact. This overlaps with your FYP1
  CSP-solver experience.
- **Model (fallback)**: unstructured/natural-language logic puzzles that don't cleanly map
  to a constraint model.
- Measured accuracy: ___  Avg tokens: ___

## 8. Code generation
- **Detection**: instruction asks to write code for a spec.
- **Strategy**: no reliable local method — route to Fireworks.
- **Model**: code-oriented allowed model (`kimi-k2p7-code`) first; compare against Gemma
  variants for the bonus prize if accuracy is close.
- **Prompt**: request code only, minimal comments, no extended explanation unless required
  by the output schema.
- Measured accuracy: ___  Avg tokens: ___

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
- Offline all-category fixture: `8/8` passing with fake Fireworks fallback.
- Fake Fireworks calls: `4/8` tasks; the other `4/8` were solved locally.
- Local zero-token paths currently passing fixture coverage:
  - arithmetic math
  - lexicon sentiment
  - regex NER baseline
  - Python syntax-error debugging
- Live Fireworks validation is still pending because `FIREWORKS_API_KEY` is not set in the
  current shell.
- Docker image `budgetbrain-track1:local` builds and runs as `linux/amd64`; size is about
  `187MB`, comfortably under the `10GB` limit.
