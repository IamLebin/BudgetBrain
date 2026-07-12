# Submission Status

## Official Leaderboard Baseline

- Leaderboard commit: `1393b47`
- Official leaderboard: rank 34
- Official hidden accuracy: 89.5%
- Official token usage: 5,093 tokens

This official score belongs to commit `1393b47`; it is not a score for the current uncommitted
release worktree.

## Current Release Validation

- Release commit: Pending until commit
- Unit tests: 79/79
- Offline strict fixture: 10/10
- Offline accuracy audit: 19/19
- Docker tag: Pending / to be filled
- Docker digest: Pending / to be filled

Offline/fake evaluation verifies routing, formatting, and deterministic solver regressions. It
does not reproduce the official hidden evaluator and must not be presented as hidden accuracy.

## Evaluator Contract

- Read `/input/tasks.json`.
- Write `/output/results.json`.
- Preserve each `task_id` and return an `answer` string.
- Infer task category because category is not supplied.
- Use only runtime-provided `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`.
- Target a public single-platform `linux/amd64` image under the event size limit.

## Current Evidence

- Safe validation command: `sh scripts/validate_all.sh`.
- Optional local Docker validation: `sh scripts/validate_all.sh --docker`.
- Optional paid-real validation: `sh scripts/validate_all.sh --paid-real`.
- The paid-real path is disabled by default and spends team Fireworks credits when explicitly run.
- The demo (`server.py`) and evaluator (`app/main.py` through `Dockerfile.track1`) are separate
  entrypoints.

## Before Publishing Another Submission Image

1. Re-run the free unit, strict, and audit checks.
2. Build and run the exact `linux/amd64` Track 1 image locally.
3. Verify `/input/tasks.json` to `/output/results.json` schema and task ordering.
4. Record the exact source commit, immutable Docker tag, digest, image platform, and image size.
5. Publish only with explicit approval, then verify anonymous registry access.
6. Update this file, README, media script, and submission form from the same release snapshot.

The release commit, Docker tag, and digest remain pending until those steps are completed for the
reviewed worktree.
