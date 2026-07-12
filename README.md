# BudgetBrain Track 1 Champion Agent

BudgetBrain is a hybrid general-purpose agent for AMD Developer Hackathon ACT II Track 1. It
routes prompts to narrow deterministic solvers when correctness can be verified and uses allowed
Fireworks models for harder or ambiguous tasks. Accuracy is the first constraint; token savings
matter only after the accuracy gate is cleared.

## Official Leaderboard Baseline

| Evidence | Official baseline value |
| --- | --- |
| Leaderboard commit | `1393b47` |
| Official leaderboard | Rank 34 |
| Official hidden accuracy | 89.5% |
| Official token usage | 5,093 tokens |

This is the source snapshot measured by the official leaderboard. It is not the commit for the
current, uncommitted release worktree.

## Current Release Validation

| Evidence | Current release value |
| --- | --- |
| Release commit | Pending until commit |
| Unit tests | 79/79 |
| Offline strict fixture | 10/10 |
| Offline accuracy audit | 19/19 |
| Docker tag | Pending / to be filled |
| Docker digest | Pending / to be filled |

The offline fixtures use deterministic/fake Fireworks responses to verify routing, formatting,
and solver behavior. They are regression evidence, not a substitute for the official hidden
leaderboard result.

## Track 1 Contract

The evaluator container:

- reads `/input/tasks.json`
- writes `/output/results.json`
- accepts `[{"task_id": "...", "prompt": "..."}]`
- returns `[{"task_id": "...", "answer": "..."}]`

Runtime configuration is supplied through `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and
`ALLOWED_MODELS`. Secrets are not stored in the repository or image.

## Reviewer Quick Start

Run the default safe validation. It does not call Fireworks or spend API credits:

```bash
sh scripts/validate_all.sh
```

Run the local demo:

```bash
cp .env.example .env
# Add a Fireworks key only if remote demo paths need to be exercised.
python3 server.py
```

Open `http://localhost:8000`. The demo server is separate from the Track 1 batch entrypoint.

## Optional Validation Layers

Build and run a local `linux/amd64` evaluator image, without publishing it:

```bash
sh scripts/validate_all.sh --docker
```

Run a paid real-Fireworks check only when team credits are intentionally approved:

```bash
sh scripts/validate_all.sh --paid-real
```

The current Vercel entrypoint in `pyproject.toml` is not the documented review path. The supported
paths in this snapshot are `server.py` for the demo and `Dockerfile.track1` for the evaluator.

## Release Status

The current release commit, public Docker tag, and digest have not been finalized. Fill them only
after committing the reviewed worktree, then building, testing, publishing, and anonymously
inspecting that exact image. Do not reuse a historical tag or digest as current release evidence.
