#!/usr/bin/env sh
set -eu

if [ ! -f ".env" ]; then
  echo "Missing .env. Create it from .env.example and add your real FIREWORKS_API_KEY." >&2
  exit 2
fi

set -a
. ./.env
set +a

python3 -B -m eval.run_local_eval --real-fireworks "$@"
