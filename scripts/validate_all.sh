#!/usr/bin/env sh
set -eu

python3 -B -m unittest discover -s tests -v

for fixture in \
  eval/fixtures/all_categories.json \
  eval/fixtures/official_practice.json \
  eval/fixtures/held_out.json \
  eval/fixtures/local_champion.json \
  eval/fixtures/reasoning_stress.json
do
  python3 -B -m eval.run_local_eval --fixture "$fixture"
done

echo "All offline checks passed."
