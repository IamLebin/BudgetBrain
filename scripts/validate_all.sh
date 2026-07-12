#!/usr/bin/env sh
set -eu

RUN_DOCKER=0
RUN_PAID_REAL=0
DOCKER_IMAGE=${BUDGETBRAIN_VALIDATION_IMAGE:-budgetbrain-track1:validation-local}
PAID_REAL_FIXTURE=${BUDGETBRAIN_PAID_REAL_FIXTURE:-eval/fixtures/official_practice.json}

usage() {
  cat <<'EOF'
Usage: sh scripts/validate_all.sh [--docker] [--paid-real]

Default: free offline validation only. No Fireworks API calls are made.

  --docker     Also build and run a local linux/amd64 Track 1 image. Never publishes it.
  --paid-real  Explicitly run a real Fireworks fixture. This spends team API credits.
EOF
}

for option in "$@"; do
  case "$option" in
    --docker) RUN_DOCKER=1 ;;
    --paid-real) RUN_PAID_REAL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $option" >&2; usage >&2; exit 2 ;;
  esac
done

echo "[1/3] Offline safe validation (free; fake/deterministic Fireworks responses)"
python3 -B -m unittest discover -s tests -v
python3 -B -m eval.run_local_eval --fixture eval/fixtures/strict_stress_20260712.json
python3 -B -m eval.run_local_eval --fixture eval/fixtures/accuracy_audit.json
echo "Offline safe validation passed. This is regression evidence, not official hidden accuracy."

if [ "$RUN_DOCKER" -eq 1 ]; then
  echo "[2/3] Optional local Docker validation (free; no publish or pull)"
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is unavailable. Run this layer manually on a Docker host." >&2
    exit 2
  fi

  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/budgetbrain-docker-validation.XXXXXX")
  trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
  mkdir -p "$tmpdir/input" "$tmpdir/output"
  printf '%s' '[{"task_id":"schema-check","prompt":"Calculate (12 + 8) * 3."}]' \
    > "$tmpdir/input/tasks.json"

  docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f Dockerfile.track1 -t "$DOCKER_IMAGE" --load .
  docker run --rm --platform linux/amd64 \
    -v "$tmpdir/input:/input:ro" \
    -v "$tmpdir/output:/output" \
    "$DOCKER_IMAGE"

  platform=$(docker image inspect "$DOCKER_IMAGE" --format '{{.Os}}/{{.Architecture}}')
  if [ "$platform" != "linux/amd64" ]; then
    echo "Unexpected image platform: $platform" >&2
    exit 1
  fi

  python3 - "$tmpdir/output/results.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
expected = [{"task_id": "schema-check", "answer": "60"}]
if rows != expected:
    raise SystemExit(f"unexpected results.json: {rows!r}")
print("Docker contract passed: linux/amd64 with valid results.json schema.")
PY
else
  echo "[2/3] Optional Docker validation skipped. Use --docker to enable it."
fi

if [ "$RUN_PAID_REAL" -eq 1 ]; then
  echo "[3/3] Optional paid-real Fireworks validation (spends team credits)"
  sh scripts/run_real_eval.sh --fixture "$PAID_REAL_FIXTURE"
else
  echo "[3/3] Paid-real Fireworks validation skipped. Use --paid-real only with approval."
fi

echo "Requested validation layers passed. Official leaderboard results remain separate evidence."
