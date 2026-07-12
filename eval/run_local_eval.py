from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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

    fake_responses: dict[str, str] = {}
    for item in fixture:
        prompt = str(item["prompt"])
        if "fake_answer" in item:
            fake_responses[prompt] = str(item["fake_answer"])
        elif "expected" in item:
            fake_responses[prompt] = str(item["expected"])
        elif isinstance(item.get("contains"), list):
            fake_responses[prompt] = " ".join(str(part) for part in item["contains"])
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
    if not answer.strip():
        return False
    if "expected" in item:
        if answer.strip().lower() != str(item["expected"]).strip().lower():
            return False
    contains = item.get("contains")
    if isinstance(contains, list):
        lowered = answer.lower()
        if not all(str(part).lower() in lowered for part in contains):
            return False
        contains_any = item.get("contains_any")
        if isinstance(contains_any, list):
            if not any(str(part).lower() in lowered for part in contains_any):
                return False

    contains_any_groups = item.get("contains_any_groups")
    if isinstance(contains_any_groups, list):
        lowered = answer.lower()
        for group in contains_any_groups:
            if not isinstance(group, list) or not any(
                str(part).lower() in lowered for part in group
            ):
                return False

    allowed_labels = item.get("allowed_labels")
    if isinstance(allowed_labels, list):
        label = re.match(r"^\s*[*_`]*(\w+)", answer)
        if label is None or label.group(1).lower() not in {
            str(value).lower() for value in allowed_labels
        }:
            return False

    exact_sentences = item.get("exact_sentences")
    if isinstance(exact_sentences, int) and _strict_sentence_count(answer) != exact_sentences:
        return False

    bullets = [
        line
        for line in answer.splitlines()
        if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)
    ]
    exact_bullets = item.get("exact_bullets")
    if isinstance(exact_bullets, int) and len(bullets) != exact_bullets:
        return False
    max_words = item.get("max_words_per_bullet")
    if isinstance(max_words, int):
        if not bullets:
            return False
        for bullet in bullets:
            content = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", bullet)
            if len(re.findall(r"\b[\w'-]+\b", content)) > max_words:
                return False

    entity_labels = item.get("entity_labels")
    if isinstance(entity_labels, list):
        for entity in entity_labels:
            if not isinstance(entity, dict):
                return False
            text = str(entity.get("text", ""))
            label = str(entity.get("label", ""))
            if not text or not label or re.search(
                re.escape(text) + r"[^\n]{0,40}\b" + re.escape(label) + r"\b",
                answer,
                re.I,
            ) is None:
                return False
    exact_entity_count = item.get("exact_entity_count")
    if isinstance(exact_entity_count, int):
        labels = re.findall(
            r"\b(?:PERSON|ORGANIZATION|LOCATION|DATE)\b(?=\s*(?:[;|\n]|$))",
            answer,
            re.I,
        )
        if len(labels) != exact_entity_count:
            return False
    exact_examples = item.get("exact_examples")
    if isinstance(exact_examples, int):
        markers = re.findall(r"\bExample\s+([1-9]\d*)\b", answer, re.I)
        if markers != [str(index) for index in range(1, exact_examples + 1)]:
            return False
    return True


def _strict_sentence_count(text: str) -> int:
    normalized = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", ""),
        text,
        flags=re.I,
    )
    parts = [part for part in re.split(r"(?<=[.!?])(?:\s+|$)", normalized.strip()) if part.strip()]
    return len(parts) if parts else int(bool(normalized.strip()))


if __name__ == "__main__":
    raise SystemExit(main())
