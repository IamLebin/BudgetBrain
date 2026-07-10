from __future__ import annotations

import json
import unittest

from app.agent import solve_prompt
from eval.run_local_eval import FakeFireworksClient
from fireworks.client import FireworksClient, parse_allowed_models
from router.classify import classify_prompt
from solvers.logic_solver import solve_logic
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment


class LocalSolverTests(unittest.TestCase):
    def test_classifies_math(self) -> None:
        self.assertEqual(classify_prompt("What is 7 * (8 + 2)?").category, "math")

    def test_math_safe_eval(self) -> None:
        solved = solve_math("What is 7 * (8 + 2)?")
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "70")

    def test_sentiment_positive(self) -> None:
        solved = solve_sentiment('Sentiment: "Excellent, smooth, and wonderful."')
        self.assertIsNotNone(solved)
        self.assertEqual(solved.answer, "positive")

    def test_ner_json_shape(self) -> None:
        solved = solve_ner('Extract named entities from: "Maya Lee visited Google Labs in Paris on July 8, 2026."')
        self.assertIsNotNone(solved)
        entities = json.loads(solved.answer)
        self.assertIn({"text": "Maya Lee", "label": "PERSON"}, entities)
        self.assertIn({"text": "Google Labs", "label": "ORG"}, entities)
        self.assertIn({"text": "Paris", "label": "LOCATION"}, entities)

    def test_fireworks_model_selection_prefers_code_model(self) -> None:
        client = FireworksClient(
            api_key="test",
            base_url="https://example.invalid",
            allowed_models=["gemma-4-26b-a4b-it", "kimi-k2p7-code"],
        )
        self.assertEqual(client.pick_model("code_generation"), "kimi-k2p7-code")

    def test_allowed_models_accepts_json_array(self) -> None:
        self.assertEqual(
            parse_allowed_models('["gemma-4-26b-a4b-it", "kimi-k2p7-code"]'),
            ["gemma-4-26b-a4b-it", "kimi-k2p7-code"],
        )

    def test_remote_category_uses_injected_client(self) -> None:
        solved = solve_prompt("What is the capital city of Japan?", client=FakeFireworksClient())
        self.assertEqual(solved.category, "factual_qa")
        self.assertEqual(solved.source, "fireworks")
        self.assertEqual(solved.answer, "Tokyo")


class LogicSolverTests(unittest.TestCase):
    """Tests for the expanded logic puzzle solver."""

    # --- Truth-teller puzzles ---

    def test_two_person_exactly_one(self) -> None:
        prompt = (
            'Logic puzzle: exactly one statement is true. '
            'Alice says "Bob is lying". Bob says "Alice is lying". '
            'Who is telling the truth?'
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertIn(result.answer, ("Alice", "Bob"))

    def test_three_person_exactly_one(self) -> None:
        # Alice says Bob is truthful, Bob says Carol is lying, Carol says Alice is lying.
        # Exactly one is telling the truth.
        prompt = (
            'Logic puzzle: exactly one of the following statements is true. '
            'Alice says "Bob is telling the truth". '
            'Bob says "Carol is lying". '
            'Carol says "Alice is lying". '
            'Who is telling the truth?'
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        # Should find exactly one truth-teller
        self.assertEqual(len(result.answer.split(", ")), 1)
        self.assertIn(result.answer, ("Alice", "Bob", "Carol"))

    def test_truth_teller_honest_variant(self) -> None:
        # Dan says Eve is lying, Eve says Dan is lying. Exactly one is truthful.
        prompt = (
            'Exactly one statement is true. '
            'Dan says "Eve is lying". Eve says "Dan is dishonest". '
            'Who is telling the truth?'
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertIn(result.answer, ("Dan", "Eve"))

    def test_truth_teller_returns_none_for_unparseable(self) -> None:
        # A statement we can't evaluate should cause a None return
        prompt = (
            'Exactly one statement is true. '
            'Alice says "the sky is blue". Bob says "water is wet". '
            'Who is telling the truth?'
        )
        result = solve_logic(prompt)
        self.assertIsNone(result)

    # --- Ordering puzzles ---

    def test_ordering_tallest(self) -> None:
        prompt = (
            "Alice is taller than Bob. Bob is taller than Carol. "
            "Who is the tallest?"
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Alice")

    def test_ordering_shortest(self) -> None:
        prompt = (
            "Alice is taller than Bob. Bob is taller than Carol. "
            "Who is the shortest?"
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Carol")

    def test_ordering_oldest(self) -> None:
        prompt = (
            "Dan is older than Eve. Eve is older than Frank. "
            "Frank is older than Grace. Who is the oldest?"
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Dan")

    def test_ordering_ambiguous_returns_none(self) -> None:
        # Two independent chains — can't determine single tallest
        prompt = (
            "Alice is taller than Bob. Carol is taller than Dave. "
            "Who is the tallest?"
        )
        result = solve_logic(prompt)
        self.assertIsNone(result)

    # --- Deduction puzzles ---

    def test_modus_tollens(self) -> None:
        prompt = (
            "If it rains, the ground is wet. "
            "The ground is dry. "
            "Did it rain?"
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "No")

    def test_modus_ponens(self) -> None:
        prompt = (
            "If the alarm rings, the door opens. "
            "The alarm rings. "
            "Does the door open?"
        )
        result = solve_logic(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")


if __name__ == "__main__":
    unittest.main()
