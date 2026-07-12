from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from app.agent import solve_prompt
from fireworks.client import FireworksClient, FireworksError


DEFAULT_INPUT = Path("/input/tasks.json")
DEFAULT_OUTPUT = Path("/output/results.json")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array")
    for item in data:
        if not isinstance(item, dict) or "task_id" not in item or "prompt" not in item:
            raise ValueError('each task must include "task_id" and "prompt"')
    return data


def run_batch(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> list[dict[str, str]]:
    tasks = load_tasks(input_path)
    results: list[dict[str, str]] = []
    client: FireworksClient | None = None
    try:
        client = FireworksClient.from_env()
    except FireworksError as exc:
        print(f"warning: shared Fireworks client unavailable: {exc}", file=sys.stderr)

    for task in tasks:
        task_id = str(task["task_id"])
        prompt = str(task["prompt"])
        try:
            solved = solve_prompt(prompt, client=client)
            answer = solved.answer
            print(
                f"{task_id}: {solved.category} via {solved.source}",
                file=sys.stderr,
            )
        except Exception as exc:  # Keep the batch contract even if one prompt is malformed.
            print(f"{task_id}: failed: {exc}", file=sys.stderr)
            answer = ""
        results.append({"task_id": task_id, "answer": answer})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="AMD Track 1 batch runner")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_batch(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
