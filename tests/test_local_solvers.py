from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from app.agent import solve_prompt
from app.main import run_batch
from eval.run_local_eval import FakeFireworksClient
from fireworks.client import (
    FireworksClient,
    FireworksError,
    clean_model_answer,
    model_id_for_request,
    normalize_model_id,
    parse_allowed_models,
)
from router.classify import classify_prompt
from solvers.code_debug_solver import solve_code_debug
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment
from solvers.logic_solver import solve_logic


class LocalSolverTests(unittest.TestCase):
    def test_classifies_math(self) -> None:
        self.assertEqual(classify_prompt("What is 7 * (8 + 2)?").category, "math")

    def test_math_safe_eval(self) -> None:
        solved = solve_math("What is 7 * (8 + 2)?")
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "70")

    def test_math_official_inventory_problem(self) -> None:
        solved = solve_math(
            "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. "
            "How many items remain?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "144")

    def test_math_percent_change(self) -> None:
        solved = solve_math("A price of 80 increases by 25 percent. What is the new price?")
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "100")
        dated = solve_math(
            "In 2025, a price of 80 increases by 25 percent. What is the new price?"
        )
        self.assertIsNotNone(dated)
        self.assertEqual(dated.answer, "100")

    def test_math_average_and_linear_equation(self) -> None:
        self.assertEqual(solve_math("Find the average of 4, 6, and 8.").answer, "6")
        self.assertEqual(solve_math("Solve 3x - 7 = 11.").answer, "6")

    def test_math_compound_projection(self) -> None:
        solved = solve_math(
            "A population starts at 1000 and grows by 5% per year for 2 years. "
            "What is the projection?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "1102.5")

    def test_math_sequential_percentages(self) -> None:
        solved = solve_math(
            "A price starts at 100, increases by 20%, then decreases by 10%. "
            "What is the final price?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "108")
        variant = solve_math(
            "The original price is $200, decreased by 25 percent and then increased by "
            "10 percent. What is the final value?"
        )
        self.assertIsNotNone(variant)
        self.assertEqual(variant.answer, "165")
        self.assertIsNone(solve_math("What is 20% of 100, then add 10?"))
        self.assertIsNone(
            solve_math(
                "A price starts at 100, increases by 20%, then decreases by 10%, "
                "then adds a 5 dollar fee. What is the final price?"
            )
        )

    def test_math_weighted_average_speed(self) -> None:
        metric = solve_math(
            "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. "
            "What is its average speed for the entire trip?"
        )
        self.assertIsNotNone(metric)
        self.assertEqual(metric.answer, "75 km/h")
        imperial = solve_math(
            "A car covers 60 miles at 30 mph and 120 miles at 60 mph. "
            "Calculate the average speed."
        )
        self.assertIsNotNone(imperial)
        self.assertEqual(imperial.answer, "45 mph")
        self.assertIsNone(
            solve_math(
                "A train travels 120 km at 60 km/h, rests for one hour, then travels "
                "180 km at 90 km/h. What is its average speed?"
            )
        )

    def test_math_ratio_share(self) -> None:
        solved = solve_math(
            "Red and blue marbles are in a 3:5 ratio. If there are 64 marbles total, "
            "how many are blue?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "40")
        variant = solve_math(
            "The ratio of cats to dogs is 2:3. There are 25 animals total. How many dogs?"
        )
        self.assertIsNotNone(variant)
        self.assertEqual(variant.answer, "15")
        self.assertIsNone(
            solve_math(
                "Red and blue marbles are in a 3:5 ratio. There are 64 total, then 4 "
                "blue marbles are removed. How many are blue?"
            )
        )

    def test_sentiment_positive(self) -> None:
        solved = solve_sentiment('Sentiment: "Excellent, smooth, and wonderful."')
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "positive")

    def test_sentiment_mixed_official_example(self) -> None:
        solved = solve_sentiment(
            "Classify the sentiment of this review: The battery life is great, "
            "but the screen scratches too easily."
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "mixed")

    def test_sentiment_balanced_uses_neutral_when_labels_are_constrained(self) -> None:
        solved = solve_sentiment(
            'Classify as positive, negative, or neutral: "The design is great but the app is slow."'
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "neutral")

    def test_sentiment_justification_routes_to_model(self) -> None:
        solved = solve_sentiment('Classify and justify the sentiment: "This is excellent."')
        self.assertIsNone(solved)

    def test_sentiment_sarcasm_routes_to_model(self) -> None:
        self.assertIsNone(
            solve_sentiment('Classify the sentiment: "Just great, another crash."')
        )

    def test_ner_json_shape(self) -> None:
        solved = solve_ner('Extract named entities from: "Maya Lee visited Google Labs in Paris on July 8, 2026."')
        self.assertIsNotNone(solved)
        entities = json.loads(solved.answer)
        self.assertIn({"text": "Maya Lee", "label": "PERSON"}, entities)
        self.assertIn({"text": "Google Labs", "label": "ORG"}, entities)
        self.assertIn({"text": "Paris", "label": "LOCATION"}, entities)

    def test_ner_official_relative_date(self) -> None:
        solved = solve_ner(
            "Extract all named entities and their types from: "
            "Maria Sanchez joined Fireworks AI in Berlin last March."
        )
        self.assertIsNotNone(solved)
        entities = json.loads(solved.answer)
        self.assertIn({"text": "Maria Sanchez", "label": "PERSON"}, entities)
        self.assertIn({"text": "Fireworks AI", "label": "ORG"}, entities)
        self.assertIn({"text": "Berlin", "label": "LOCATION"}, entities)
        self.assertIn({"text": "last March", "label": "DATE"}, entities)

    def test_ner_common_multiword_location_and_news_org(self) -> None:
        location = solve_ner("Extract named entities from: New York hosted the summit.")
        self.assertIsNotNone(location)
        self.assertIn(
            {"text": "New York", "label": "LOCATION"},
            json.loads(location.answer),
        )
        organization = solve_ner("Extract named entities from: The New York Times reported it.")
        self.assertIsNotNone(organization)
        self.assertIn(
            {"text": "The New York Times", "label": "ORG"},
            json.loads(organization.answer),
        )

    def test_logic_exactly_one_truth(self) -> None:
        solved = solve_logic(
            'Logic puzzle: exactly one statement is true. Alice says "Bob is lying". '
            'Bob says "Alice is lying". Carol says "Alice is lying". Who is telling the truth?'
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "Alice")

    def test_logic_official_assignment_problem(self) -> None:
        solved = solve_logic(
            "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. "
            "Sam does not own the bird. Jo owns the dog. Who owns the cat?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "Sam")

    def test_logic_unique_ordering(self) -> None:
        solved = solve_logic(
            "Logic puzzle: Ria, Sol, and Tom sit in one row. Ria sits left of Sol, "
            "and Tom sits right of Sol. What is their order from left to right?"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "Ria, Sol, Tom")
        ambiguous = solve_logic(
            "Logic puzzle: Ria, Sol, and Tom sit in one row. Ria sits left of Sol. "
            "What is their order from left to right?"
        )
        self.assertIsNone(ambiguous)

    def test_logic_conditional_and_universal_deduction(self) -> None:
        conditional = solve_logic(
            "Logical deduction: If the server is offline, the alert is red. "
            "The alert is not red. Can the server be offline?"
        )
        self.assertIsNotNone(conditional)
        self.assertEqual(conditional.answer, "No")
        universal = solve_logic(
            "Logical deduction: All red keys open the vault. Key A is red. "
            "Does Key A open the vault?"
        )
        self.assertIsNotNone(universal)
        self.assertEqual(universal.answer, "Yes")
        self.assertEqual(
            classify_prompt(
                "If the server is offline, the alert is red. The alert is not red. "
                "Can the server be offline?"
            ).category,
            "logic",
        )
        self.assertEqual(
            classify_prompt(
                "All red keys open the vault. Key A is red. Does Key A open the vault?"
            ).category,
            "logic",
        )

    def test_code_debug_returns_corrected_code(self) -> None:
        solved = solve_code_debug(
            "Find the syntax issue in this Python code:\n"
            "```python\ndef add(a, b)\n    return a + b\n```"
        )
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "def add(a, b):\n    return a + b")

    def test_code_debug_official_max_problem(self) -> None:
        solved = solve_code_debug(
            "This function should return the max of a list but has a bug: "
            "def get_max(nums): return nums[0]. Find and fix it."
        )
        self.assertIsNotNone(solved)
        self.assertIn("return max(nums)", solved.answer)

    def test_code_debug_mutable_default_and_len_index(self) -> None:
        mutable = solve_code_debug(
            "Find and fix the mutable-default bug:\n"
            "```python\ndef add_item(item, bucket=[]):\n"
            "    bucket.append(item)\n    return bucket\n```"
        )
        self.assertIsNotNone(mutable)
        self.assertIn("bucket=None", mutable.answer)
        self.assertIn("if bucket is None", mutable.answer)
        nonempty = solve_code_debug(
            "Fix the mutable default bug:\n"
            "```python\ndef collect(value, items=[1]):\n"
            "    items.append(value)\n    return items\n```"
        )
        self.assertIsNotNone(nonempty)
        self.assertIn("items = [1]", nonempty.answer)
        index = solve_code_debug(
            "This function raises IndexError for non-empty lists. Find and fix the bug:\n"
            "```python\ndef last(items):\n    return items[len(items)]\n```"
        )
        self.assertIsNotNone(index)
        self.assertIn("return items[-1]", index.answer)

    def test_official_practice_categories(self) -> None:
        cases = {
            "What is the capital of Australia, and what body of water is it near?": "factual_qa",
            "A store has 240 items. It sells 15% on Monday. How many remain?": "math",
            "Classify the sentiment of this review: It is great.": "sentiment",
            "Summarize the following in exactly one sentence: A long passage.": "summarization",
            "Extract all named entities and their types from: Ada joined Acme Inc.": "ner",
            "This function has a bug: def f(x): return x[0]. Find and fix it.": "code_debugging",
            "Three friends each own a different pet. Who owns the cat?": "logic",
            "Write a Python function that returns the second-largest number.": "code_generation",
        }
        for prompt, category in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_prompt(prompt).category, category)

    def test_version_comparison_is_factual_not_math(self) -> None:
        self.assertEqual(
            classify_prompt("What is the difference between Python 2 and Python 3?").category,
            "factual_qa",
        )
        self.assertEqual(
            classify_prompt("How should a company implement ISO 27001 controls?").category,
            "factual_qa",
        )

    def test_fireworks_model_selection_prefers_code_model(self) -> None:
        client = FireworksClient(
            api_key="test",
            base_url="https://example.invalid",
            allowed_models=["gemma-4-26b-a4b-it", "kimi-k2p7-code"],
        )
        self.assertEqual(client.pick_model("code_generation"), "kimi-k2p7-code")

    def test_fireworks_model_selection_prefers_kimi_for_low_token_language_tasks(self) -> None:
        client = FireworksClient(
            api_key="test",
            base_url="https://example.invalid",
            allowed_models=["minimax-m3", "kimi-k2p7-code"],
        )
        self.assertEqual(client.pick_model("factual_qa"), "kimi-k2p7-code")
        self.assertEqual(client.pick_model("summarization"), "kimi-k2p7-code")
        self.assertEqual(client.pick_model("sentiment"), "kimi-k2p7-code")

    def test_fireworks_model_selection_avoids_gemma_by_default(self) -> None:
        client = FireworksClient(
            api_key="test",
            base_url="https://example.invalid",
            allowed_models=["gemma-4-26b-a4b-it", "minimax-m3"],
        )
        self.assertEqual(client.pick_model("factual_qa"), "minimax-m3")

    def test_allowed_models_accepts_json_array(self) -> None:
        self.assertEqual(
            parse_allowed_models('["gemma-4-26b-a4b-it", "kimi-k2p7-code"]'),
            ["gemma-4-26b-a4b-it", "kimi-k2p7-code"],
        )

    def test_allowed_models_are_required_and_deduplicated(self) -> None:
        self.assertEqual(parse_allowed_models(""), [])
        self.assertEqual(parse_allowed_models("minimax-m3,minimax-m3"), ["minimax-m3"])

    def test_normalize_model_id_adds_fireworks_prefix(self) -> None:
        self.assertEqual(
            normalize_model_id("minimax-m3"),
            "accounts/fireworks/models/minimax-m3",
        )

    def test_proxy_keeps_exact_allowed_model_id(self) -> None:
        self.assertEqual(
            model_id_for_request("minimax-m3", "https://judge.example/proxy/v1"),
            "minimax-m3",
        )
        self.assertEqual(
            model_id_for_request("minimax-m3", "https://api.fireworks.ai/inference/v1"),
            "accounts/fireworks/models/minimax-m3",
        )

    def test_full_model_ids_still_follow_category_preference(self) -> None:
        client = FireworksClient(
            api_key="test",
            base_url="https://judge.example/proxy/v1",
            allowed_models=[
                "accounts/fireworks/models/gemma-4-26b-a4b-it",
                "accounts/fireworks/models/kimi-k2p7-code",
            ],
        )
        self.assertEqual(
            client.pick_model("code_generation"),
            "accounts/fireworks/models/kimi-k2p7-code",
        )

    def test_code_fences_are_removed_from_model_answer(self) -> None:
        self.assertEqual(
            clean_model_answer("```python\ndef square(n):\n    return n * n\n```", "code_generation"),
            "def square(n):\n    return n * n",
        )

    def test_short_classification_answers_are_normalized(self) -> None:
        self.assertEqual(
            clean_model_answer("**Neutral**\n\nThis is factual.", "sentiment", "Classify sentiment."),
            "neutral",
        )
        self.assertEqual(
            clean_model_answer("Yes, the key opens the vault.", "logic", "Does it open?"),
            "Yes",
        )
        self.assertEqual(
            clean_model_answer(
                "**Final price: 108**\n\nStep-by-step:\n- Start: 100",
                "math",
                "What is the final price?",
            ),
            "108",
        )
        self.assertEqual(
            clean_model_answer("**75 km/h", "math", "What is the average speed?"),
            "75 km/h",
        )
        self.assertEqual(
            clean_model_answer(
                "Positive because it is reliable.",
                "sentiment",
                "Classify and explain the sentiment.",
            ),
            "Positive because it is reliable.",
        )
        self.assertEqual(
            normalize_model_id("accounts/fireworks/models/minimax-m3"),
            "accounts/fireworks/models/minimax-m3",
        )

    def test_remote_category_uses_injected_client(self) -> None:
        solved = solve_prompt("What is the capital city of Japan?", client=FakeFireworksClient())
        self.assertEqual(solved.category, "factual_qa")
        self.assertEqual(solved.source, "fireworks")
        self.assertEqual(solved.answer, "Tokyo")

    def test_fireworks_retries_unavailable_model(self) -> None:
        class RetryClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://api.fireworks.ai/inference/v1",
                    allowed_models=["minimax-m3", "gemma-4-26b-a4b-it"],
                )
                self.models: list[str] = []

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.models.append(payload["model"])
                if payload["model"] == "accounts/fireworks/models/minimax-m3":
                    raise FireworksError("HTTP 404: model unavailable", status_code=404)
                return {"choices": [{"message": {"content": "fallback ok"}}]}

        client = RetryClient()
        self.assertEqual(client.solve("Question?", "factual_qa"), "fallback ok")
        self.assertEqual(
            client.models,
            ["accounts/fireworks/models/minimax-m3", "accounts/fireworks/models/gemma-4-26b-a4b-it"],
        )

    def test_batch_contract_preserves_order_and_writes_strings(self) -> None:
        tasks = [
            {"task_id": "one", "prompt": "What is 6 * 7?"},
            {
                "task_id": "two",
                "prompt": 'Classify as positive, negative, or neutral: "This is excellent."',
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "tasks.json"
            output_path = Path(tmp) / "results.json"
            input_path.write_text(json.dumps(tasks), encoding="utf-8")
            results = run_batch(input_path, output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(results, written)
        self.assertEqual([row["task_id"] for row in results], ["one", "two"])
        self.assertEqual([row["answer"] for row in results], ["42", "positive"])


if __name__ == "__main__":
    unittest.main()
