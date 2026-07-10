from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from app.agent import SolveResult, solve_prompt
from app.main import run_batch
from router.classify import classify_prompt


TRACK1_TOTAL_TASKS = 19
ACCURACY_GATE = 0.80
MIN_CORRECT_TO_PASS = 16

SAMPLE_TASKS = [
    {"task_id": "math-1", "prompt": "Calculate (12 + 8) * 3."},
    {"task_id": "sentiment-1", "prompt": 'Classify the sentiment: "The setup was smooth and the result is excellent."'},
    {"task_id": "ner-1", "prompt": 'Extract named entities from: "Lisa Wong met OpenAI in Paris on July 8, 2026."'},
]


class FakeFireworksClient:
    ANSWERS = {
        "factual_qa": "Tokyo",
        "summarization": (
            "Budget tracking helps people understand spending patterns and make clearer "
            "financial tradeoffs."
        ),
        "logic": "Alice",
        "code_generation": "def square(n):\n    return n * n",
        "code_debugging": "Fix the syntax or logic error in the code.",
        "math": "0",
        "sentiment": "neutral",
        "ner": "[]",
    }

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.calls: list[dict[str, str]] = []
        self.responses = responses or {}

    def solve(self, prompt: str, category: str) -> str:
        self.calls.append({"category": category, "prompt": prompt})
        return self.responses.get(prompt, self.ANSWERS.get(category, ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny local smoke eval")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path("eval/fixtures/all_categories.json"))
    parser.add_argument(
        "--real-fireworks",
        action="store_true",
        help="Use the real Fireworks API for model-needed tasks. This spends tokens.",
    )
    args = parser.parse_args()

    if args.fixture.exists() and args.input is None:
        return run_fixture_eval(args.fixture, fake_fireworks=not args.real_fireworks)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        input_path = args.input or tmpdir / "tasks.json"
        output_path = tmpdir / "results.json"
        if args.input is None:
            input_path.write_text(json.dumps(SAMPLE_TASKS), encoding="utf-8")
        results = run_batch(input_path, output_path)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def run_fixture_eval(path: Path, fake_fireworks: bool = False) -> int:
    if not fake_fireworks and not os.getenv("FIREWORKS_API_KEY", "").strip():
        print("FIREWORKS_API_KEY is missing; run without --real-fireworks for offline validation.")
        return 2

    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, list):
        raise ValueError("fixture must be a JSON array")

    fake_responses = {
        str(item["prompt"]): str(item["fake_answer"])
        for item in fixture
        if "fake_answer" in item
    }
    fake_client = FakeFireworksClient(fake_responses) if fake_fireworks else None
    rows: list[dict[str, Any]] = []
    correct = 0

    for item in fixture:
        prompt = str(item["prompt"])
        expected_category = str(item["category"])
        classification = classify_prompt(prompt)
        solved = solve_prompt(prompt, client=fake_client)
        answer_ok = _answer_matches(solved.answer, item)
        category_ok = classification.category == expected_category
        passed = answer_ok and category_ok
        correct += int(passed)
        rows.append(
            {
                "task_id": item["task_id"],
                "expected_category": expected_category,
                "actual_category": classification.category,
                "source": solved.source,
                "answer": solved.answer,
                "category_ok": category_ok,
                "answer_ok": answer_ok,
                "passed": passed,
            }
        )

    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"{status} {row['task_id']} category={row['actual_category']} "
            f"source={row['source']} answer={row['answer']!r}"
        )
    total = len(rows)
    calls = len(fake_client.calls) if fake_client else "real/env"
    print(f"score={correct}/{total} fake_fireworks_calls={calls}")
    print(
        f"track1_gate={ACCURACY_GATE:.0%} real_eval_requires_at_least={MIN_CORRECT_TO_PASS}/{TRACK1_TOTAL_TASKS}"
    )
    return 0 if correct == total else 1


def _answer_matches(answer: str, item: dict[str, Any]) -> bool:
    if "expected" in item:
        return answer.strip().lower() == str(item["expected"]).strip().lower()
    contains = item.get("contains")
    if isinstance(contains, list):
        lowered = answer.lower()
        if not all(str(part).lower() in lowered for part in contains):
            return False
        contains_any = item.get("contains_any")
        if isinstance(contains_any, list):
            return any(str(part).lower() in lowered for part in contains_any)
        return True
    return bool(answer.strip())


if __name__ == "__main__":
    raise SystemExit(main())
