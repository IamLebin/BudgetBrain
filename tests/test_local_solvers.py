from __future__ import annotations

import json
import unittest

from app.agent import solve_prompt
from eval.run_local_eval import FakeFireworksClient
from fireworks.client import FireworksClient, parse_allowed_models
from router.classify import classify_prompt
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


if __name__ == "__main__":
    unittest.main()
