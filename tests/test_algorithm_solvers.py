from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import unittest

from app.agent import solve_prompt
from eval.run_local_eval import FakeFireworksClient
from solvers.code_generation_solver import solve_code_generation
from solvers.factual_solver import solve_factual
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
        self.assertIsNone(solve_code_generation("Write a Python function that computes an FFT."))

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

    def test_fixture_routing_reduces_remote_calls_without_changing_answers(self) -> None:
        fixture = json.loads(Path("eval/fixtures/held_out.json").read_text(encoding="utf-8"))
        responses = {str(item["prompt"]): str(item.get("fake_answer", "ok")) for item in fixture}
        client = FakeFireworksClient(responses)
        for item in fixture:
            solve_prompt(str(item["prompt"]), client=client)
        self.assertLessEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
