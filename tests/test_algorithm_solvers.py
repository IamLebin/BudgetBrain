from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import unittest

from app.agent import solve_prompt
from eval.run_local_eval import FakeFireworksClient
from solvers.code_generation_solver import solve_code_generation
from solvers.factual_solver import solve_factual
from solvers.math_solver import solve_math
from solvers.sentiment_solver import solve_sentiment
from solvers.summarization_solver import solve_summarization


class AlgorithmSolverTests(unittest.TestCase):
    def test_python_generators_execute(self) -> None:
        cases = (
            (
                "Write a Python function that returns the second-largest number in a list.",
                "second_largest", ([3, 1, 3, 2],), 2,
            ),
            (
                "Write a Python function is_palindrome(text) ignoring case and punctuation.",
                "is_palindrome", ("A man, a plan, a canal: Panama!",), True,
            ),
            (
                "Write a Python function is_balanced(s) for (), [], and {} brackets.",
                "is_balanced", ("{[()]}",), True,
            ),
            (
                "Write a Python function merge_intervals(intervals) that merges all overlapping intervals.",
                "merge_intervals", ([[1, 3], [2, 6], [8, 10]],), [[1, 6], [8, 10]],
            ),
            (
                "Write a Python function called square that returns n multiplied by itself.",
                "square", (7,), 49,
            ),
            (
                "Can you write a Python function that reverses a list?",
                "reverse_list", ([1, 2, 3],), [3, 2, 1],
            ),
            (
                "Write a Python function is_even(n) that checks whether n is even.",
                "is_even", (14,), True,
            ),
            (
                "Write a Python function sum_list(numbers) that returns the sum of a list.",
                "sum_list", ([2, 3, 5],), 10,
            ),
            (
                "Write a Python function reverse_string(text) that reverses a string.",
                "reverse_string", ("abc",), "cba",
            ),
            (
                "Write a Python function count_vowels(text) that counts vowels in a string.",
                "count_vowels", ("Fireworks",), 3,
            ),
        )
        for prompt, name, args, expected in cases:
            with self.subTest(prompt=prompt):
                solved = solve_code_generation(prompt)
                self.assertIsNotNone(solved)
                namespace: dict[str, object] = {}
                exec(solved.answer, namespace)  # noqa: S102 - deterministic templates only.
                self.assertEqual(namespace[name](*args), expected)  # type: ignore[operator]

        self.assertIsNone(
            solve_code_generation(
                "Write a Python function returning the second-largest value including duplicates."
            )
        )
        self.assertIsNone(
            solve_code_generation("Write a Python function that reverses a list in-place.")
        )
        self.assertIsNone(solve_code_generation("Write a Python function that computes an FFT."))

    def test_symbolic_single_step_equations_are_isolated_exactly(self) -> None:
        cases = {
            "Solve for x: x + y = z.": "x = z - y",
            "Solve for y: x + y = z.": "y = z - x",
            "Solve for z: x - y = z.": "z = x - y",
            "Solve for y: x - y = z.": "y = x - z",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_math(prompt)
                self.assertIsNotNone(solved)
                self.assertEqual(solved.answer, expected)
                self.assertEqual(solved.method, "symbolic_isolation")

        self.assertIsNone(solve_math("Solve for x: x + y + q = z."))

    def test_median_of_explicit_values_is_exact(self) -> None:
        cases = {
            "Find the median of 9, 1, and 3.": "3",
            "Calculate the median of 1, 2, 8, and 9.": "5",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_math(prompt)
                self.assertIsNotNone(solved)
                self.assertEqual(solved.answer, expected)
                self.assertEqual(solved.method, "median")
        self.assertIsNone(solve_math("Find the median salary for each department."))

    def test_short_unambiguous_sentiment_uses_strong_local_path(self) -> None:
        cases = {
            "Analyze the sentiment of: The setup is excellent.": "positive",
            "Determine whether this review is favorable or unfavorable: It works perfectly.": "favorable",
            "Classify the sentiment: I don't think this is good.": "negative",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_sentiment(prompt)
                self.assertIsNotNone(solved)
                self.assertEqual(solved.answer, expected)
                self.assertIn(solved.method, {"strong_single_lexicon", "explicit_negated_lexicon"})

        emphatic_negative = solve_sentiment("Classify the sentiment: This is perfectly awful.")
        self.assertIsNotNone(emphatic_negative)
        self.assertEqual(emphatic_negative.answer, "negative")

        ambiguous_prompts = (
            "Classify the sentiment: The setup might be good, but unclear.",
            "Classify the sentiment: Is this actually good?",
            "Classify the sentiment: This is hardly bad.",
        )
        for ambiguous in ambiguous_prompts:
            with self.subTest(prompt=ambiguous):
                routed = solve_prompt(ambiguous, client=FakeFireworksClient({ambiguous: "neutral"}))
                self.assertEqual(routed.source, "fireworks")

    def test_grouped_average_sql_executes(self) -> None:
        solved = solve_code_generation(
            "Write an SQL query that returns each department_id and its average salary "
            "from employees, ordered by average salary descending."
        )
        self.assertIsNotNone(solved)
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE employees (department_id INTEGER, salary REAL)")
            connection.executemany(
                "INSERT INTO employees VALUES (?, ?)",
                [(1, 10), (1, 20), (2, 40), (2, 60)],
            )
            self.assertEqual(connection.execute(solved.answer).fetchall(), [(2, 50.0), (1, 15.0)])

    def test_stdlib_http_status_is_not_an_answer_cache(self) -> None:
        for code, phrase in ((404, "Not Found"), (503, "Service Unavailable"), (201, "Created")):
            with self.subTest(code=code):
                solved = solve_factual(f"Explain HTTP status code {code}.")
                self.assertIsNotNone(solved)
                self.assertIn(phrase, solved.answer)
        self.assertIsNone(solve_factual("What is the capital of Australia?"))

    def test_stdlib_python_exception_descriptions(self) -> None:
        for name in ("TypeError", "ValueError", "IndexError", "KeyError"):
            with self.subTest(name=name):
                solved = solve_factual(f"Explain what a {name} means in Python.")
                self.assertIsNotNone(solved)
                self.assertIn(name, solved.answer)
                self.assertEqual(solved.method, "stdlib_python_exception")
                routed = solve_prompt(
                    f"Explain what a {name} means in Python.",
                    client=FakeFireworksClient(),
                )
                self.assertEqual(routed.source, "local:stdlib_python_exception")
        self.assertIsNone(solve_factual("Explain what an HTTPError means in a web API."))

    def test_summary_constraints_are_proven_locally(self) -> None:
        limited = solve_summarization(
            "Summarize in no more than 20 words: The library extended weekend hours, "
            "added quiet study rooms, and introduced free coding workshops after student requests."
        )
        self.assertIsNotNone(limited)
        self.assertLessEqual(len(limited.answer.split()), 20)

        joined = solve_summarization(
            "Summarize in exactly one sentence: The city replaced diesel buses with electric "
            "models. Officials expect lower costs and fewer emissions."
        )
        self.assertIsNotNone(joined)
        self.assertEqual(joined.method, "two_sentence_join")
        self.assertIn(";", joined.answer)

        acronym = solve_summarization(
            "Summarize in one sentence: The launch window moved to Friday. "
            "NASA approved the updated schedule."
        )
        self.assertIsNotNone(acronym)
        self.assertIn("NASA", acronym.answer)

        passthroughs = (
            "Provide a summary of this report: Revenue rose while costs stayed flat.",
            "Give a one-sentence overview of this report: Revenue rose and costs fell.",
        )
        for prompt in passthroughs:
            with self.subTest(prompt=prompt):
                solved = solve_summarization(prompt)
                self.assertIsNotNone(solved)
                self.assertIn(solved.method, {"short_source_passthrough", "already_one_sentence"})

    def test_fixture_routing_reduces_remote_calls_without_changing_answers(self) -> None:
        fixture = json.loads(Path("eval/fixtures/held_out.json").read_text(encoding="utf-8"))
        responses = {str(item["prompt"]): str(item.get("fake_answer", "ok")) for item in fixture}
        client = FakeFireworksClient(responses)
        for item in fixture:
            solve_prompt(str(item["prompt"]), client=client)
        self.assertLessEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
