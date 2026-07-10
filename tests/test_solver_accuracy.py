"""Broader accuracy tests for all local solvers.

Each solver gets 5-8 diverse test cases including edge cases.
Run with: python -m pytest tests/test_solver_accuracy.py -v
"""
from __future__ import annotations

import json
import unittest

from solvers.code_debug_solver import solve_code_debug
from solvers.logic_solver import solve_logic
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment


class MathSolverAccuracy(unittest.TestCase):
    """Diverse arithmetic and word-problem tests."""

    def test_basic_addition(self) -> None:
        r = solve_math("What is 15 + 27?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "42")

    def test_nested_parens(self) -> None:
        r = solve_math("Evaluate: (3 + 5) * (10 - 2)")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "64")

    def test_word_sum(self) -> None:
        r = solve_math("What is the sum of 120 and 55?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "175")

    def test_word_product(self) -> None:
        r = solve_math("What is the product of 14 and 6?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "84")

    def test_percentage(self) -> None:
        r = solve_math("What is 25% of 200?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "50")

    def test_percentage_decimal(self) -> None:
        r = solve_math("What is 15% of 80?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "12")

    def test_division(self) -> None:
        r = solve_math("Divide 144 by 12")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "12")

    def test_multiplication_times(self) -> None:
        r = solve_math("What is 9 times 7?")
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "63")


class SentimentSolverAccuracy(unittest.TestCase):
    """Diverse sentiment cases including negation and mixed signals."""

    def test_strong_positive(self) -> None:
        r = solve_sentiment('Classify the sentiment: "This product is amazing and wonderful!"')
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "positive")

    def test_strong_negative(self) -> None:
        r = solve_sentiment('Classify the sentiment: "Terrible service, awful experience."')
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "negative")

    def test_negated_positive(self) -> None:
        r = solve_sentiment('Classify the sentiment: "This is not good at all."')
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "negative")

    def test_negated_negative(self) -> None:
        r = solve_sentiment('Classify the sentiment: "The movie was not bad."')
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "positive")

    def test_neutral_no_signals(self) -> None:
        r = solve_sentiment('Classify the sentiment: "The meeting is at 3pm tomorrow."')
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "neutral")

    def test_multiple_positive_keywords(self) -> None:
        r = solve_sentiment(
            'Sentiment: "Excellent quality, reliable delivery, and the team was great."'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "positive")

    def test_mixed_slightly_negative(self) -> None:
        r = solve_sentiment(
            'Classify the sentiment as positive, negative, or neutral: '
            '"The food was good but the service was terrible and slow."'
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "negative")


class NERSolverAccuracy(unittest.TestCase):
    """Diverse NER extraction tests."""

    def _entities(self, prompt: str) -> list[dict[str, str]]:
        r = solve_ner(prompt)
        self.assertIsNotNone(r)
        return json.loads(r.answer)

    def test_person_and_org(self) -> None:
        entities = self._entities(
            'Extract named entities from: "John Smith works at Microsoft Corp."'
        )
        self.assertIn({"text": "John Smith", "label": "PERSON"}, entities)
        self.assertIn({"text": "Microsoft Corp", "label": "ORG"}, entities)

    def test_date_iso(self) -> None:
        entities = self._entities(
            'Extract named entities from: "The deadline is 2026-07-11."'
        )
        labels = [e["label"] for e in entities]
        self.assertIn("DATE", labels)

    def test_date_verbose(self) -> None:
        entities = self._entities(
            'Extract named entities from: "We launch on March 15, 2026."'
        )
        date_entities = [e for e in entities if e["label"] == "DATE"]
        self.assertTrue(len(date_entities) >= 1)

    def test_location(self) -> None:
        entities = self._entities(
            'Extract named entities from: "She traveled from London to Berlin."'
        )
        locations = [e for e in entities if e["label"] == "LOCATION"]
        self.assertTrue(len(locations) >= 1)

    def test_email(self) -> None:
        entities = self._entities(
            'Extract named entities from: "Contact us at hello@example.com for details."'
        )
        emails = [e for e in entities if e["label"] == "EMAIL"]
        self.assertEqual(len(emails), 1)
        self.assertEqual(emails[0]["text"], "hello@example.com")

    def test_multiple_people(self) -> None:
        entities = self._entities(
            'Identify entities in: "Alice Johnson and Bob Williams met at the conference."'
        )
        people = [e for e in entities if e["label"] == "PERSON"]
        self.assertTrue(len(people) >= 2)

    def test_no_entities_returns_none(self) -> None:
        # No entity-extraction instruction -> should return None
        r = solve_ner("The weather is nice today.")
        self.assertIsNone(r)


class CodeDebugSolverAccuracy(unittest.TestCase):
    """Tests for syntax error detection."""

    def test_missing_colon(self) -> None:
        r = solve_code_debug(
            "Find the bug:\n```python\ndef hello()\n    print('hi')\n```"
        )
        self.assertIsNotNone(r)
        self.assertIn("Syntax error", r.answer)

    def test_unclosed_paren(self) -> None:
        r = solve_code_debug(
            "Debug this code:\n```python\nprint('hello'\n```"
        )
        self.assertIsNotNone(r)
        self.assertIn("Syntax error", r.answer)

    def test_invalid_indentation(self) -> None:
        r = solve_code_debug(
            "What's wrong?\n```python\ndef foo():\nreturn 1\n```"
        )
        self.assertIsNotNone(r)
        self.assertIn("Syntax error", r.answer)

    def test_valid_code_returns_none(self) -> None:
        r = solve_code_debug(
            "Debug this:\n```python\ndef add(a, b):\n    return a + b\n```"
        )
        # Valid syntax — can't detect logical bugs, returns None
        self.assertIsNone(r)

    def test_logic_bug_returns_none(self) -> None:
        # Off-by-one is a logic bug, not a syntax error
        r = solve_code_debug(
            "This function should return the last element but returns the wrong one:\n"
            "```python\ndef last(lst):\n    return lst[len(lst)]\n```"
        )
        # Valid syntax — should return None (can't detect logic bugs)
        self.assertIsNone(r)


class LogicSolverAccuracy(unittest.TestCase):
    """Expanded logic solver accuracy tests."""

    def test_two_person_basic(self) -> None:
        r = solve_logic(
            'Logic puzzle: exactly one statement is true. '
            'Alice says "Bob is lying". Bob says "Alice is lying". '
            'Who is telling the truth?'
        )
        self.assertIsNotNone(r)
        self.assertIn(r.answer, ("Alice", "Bob"))

    def test_three_person_chain(self) -> None:
        r = solve_logic(
            'Exactly one of the following statements is true. '
            'Alice says "Bob is telling the truth". '
            'Bob says "Carol is lying". '
            'Carol says "Alice is lying". '
            'Who is telling the truth?'
        )
        self.assertIsNotNone(r)
        self.assertIn(r.answer, ("Alice", "Bob", "Carol"))

    def test_ordering_three(self) -> None:
        r = solve_logic(
            "Alice is taller than Bob. Bob is taller than Carol. "
            "Who is the tallest?"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "Alice")

    def test_ordering_four_oldest(self) -> None:
        r = solve_logic(
            "Dan is older than Eve. Eve is older than Frank. "
            "Frank is older than Grace. Who is the oldest?"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "Dan")

    def test_ordering_four_youngest(self) -> None:
        r = solve_logic(
            "Dan is older than Eve. Eve is older than Frank. "
            "Frank is older than Grace. Who is the youngest?"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "Grace")

    def test_modus_tollens_rain(self) -> None:
        r = solve_logic(
            "If it rains, the ground is wet. "
            "The ground is dry. "
            "Did it rain?"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "No")

    def test_modus_ponens_alarm(self) -> None:
        r = solve_logic(
            "If the alarm rings, the door opens. "
            "The alarm rings. "
            "Does the door open?"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.answer, "Yes")

    def test_unparseable_falls_back(self) -> None:
        # A complex natural-language puzzle we can't handle deterministically
        r = solve_logic(
            "Three friends each ordered a different drink. "
            "No one drank the same thing. What did Alice order?"
        )
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
