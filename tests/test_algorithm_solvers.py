from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import unittest

from app.agent import solve_prompt
from eval.run_local_eval import FakeFireworksClient, _answer_matches
from solvers.code_debug_solver import solve_code_debug
from solvers.code_generation_solver import solve_code_generation
from solvers.factual_solver import solve_factual
from solvers.logic_solver import solve_logic
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment
from solvers.summarization_solver import solve_summarization


class AlgorithmSolverTests(unittest.TestCase):
    def test_strict_eval_rejects_missing_content_and_format(self) -> None:
        item = {
            "contains": ["benefit", "risk"],
            "exact_bullets": 2,
            "max_words_per_bullet": 4,
        }
        self.assertTrue(_answer_matches("- Benefit is clear.\n- Risk remains high.", item))
        self.assertFalse(_answer_matches("- Benefit is clear.\n- Response is ready.", item))
        self.assertFalse(_answer_matches("- Benefit is very clearly described here.\n- Risk remains high.", item))
        self.assertFalse(_answer_matches("Benefit and risk are covered.", item))
        grouped = {"contains_any_groups": [["set up", "setup"], ["dented", "damaged"]]}
        self.assertTrue(_answer_matches("The dented device was easy to set up.", grouped))
        self.assertFalse(_answer_matches("The device worked well.", grouped))

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
            (
                "Write a Python function unique_words(text) that returns a set of lowercase words "
                "from text, ignoring punctuation.",
                "unique_words", ("Hello, HELLO world!",), {"hello", "world"},
            ),
            (
                "Write a Python function merge_sorted(a, b) that returns one sorted list "
                "containing the values from two already sorted lists.",
                "merge_sorted", ([1, 4, 7], [2, 3, 9]), [1, 2, 3, 4, 7, 9],
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

    def test_remaining_audit_algorithms_are_local_and_executable(self) -> None:
        factual_cases = {
            "In one concise sentence, explain the difference between authentication and authorization.": (
                "identity", "permission"
            ),
            "What do the ACID properties of a database transaction stand for?": (
                "Atomicity", "Consistency", "Isolation", "Durability"
            ),
        }
        for prompt, concepts in factual_cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_prompt(prompt, client=FakeFireworksClient())
                self.assertEqual(solved.source, "local:standard_factual_definition")
                for concept in concepts:
                    self.assertIn(concept.lower(), solved.answer.lower())

    def test_standard_technology_definitions_are_complete_and_local(self) -> None:
        fixture = json.loads(
            Path("eval/fixtures/practice_factual_10_20260712.json").read_text(encoding="utf-8")
        )
        deterministic_ids = {"Q2", "Q4", "Q5", "Q7", "Q8", "Q9", "Q10"}
        for item in fixture:
            if item["task_id"] not in deterministic_ids:
                continue
            with self.subTest(task_id=item["task_id"]):
                client = FakeFireworksClient()
                solved = solve_prompt(item["prompt"], client=client)
                self.assertEqual(solved.source, "local:standard_factual_definition")
                self.assertTrue(_answer_matches(solved.answer, item))
                self.assertEqual(client.calls, [])

        unsupported = (
            "Define cloud computing and list three providers by market share in 2026.",
            "Explain the history of neural networks and how they process information.",
            "Explain blockchain and name two industries with current adoption rates.",
            "Define cybersecurity and list three threats with their costs.",
        )
        for prompt in unsupported:
            with self.subTest(prompt=prompt):
                self.assertIsNone(solve_factual(prompt))

    def test_role_based_ner_is_complete_and_local(self) -> None:
        fixture = json.loads(
            Path("eval/fixtures/strict_stress_20260712.json").read_text(encoding="utf-8")
        )
        ner_items = [item for item in fixture if item["category"] == "ner"]
        for item in ner_items:
            with self.subTest(task_id=item["task_id"]):
                client = FakeFireworksClient()
                solved = solve_prompt(item["prompt"], client=client)
                self.assertEqual(solved.source, "local:regex_entities")
                self.assertTrue(_answer_matches(solved.answer, item))
                self.assertEqual(client.calls, [])

        ambiguous = solve_ner(
            "Extract named entities from: Apple CEO spoke with Tim Cook in California."
        )
        self.assertIsNotNone(ambiguous)
        self.assertLess(ambiguous.confidence, 0.9)

        debug_prompt = (
            "Fix this Python function so it returns the sum of all numbers, including the final item. "
            "Return code only.\n```python\ndef total(values):\n    total = 0\n"
            "    for i in range(len(values) - 1):\n        total += values[i]\n    return total\n```"
        )
        repaired = solve_code_debug(debug_prompt)
        self.assertIsNotNone(repaired)
        namespace: dict[str, object] = {}
        exec(repaired.answer, namespace)  # noqa: S102 - deterministic repair only.
        self.assertEqual(namespace["total"]([1, 2, 3]), 6)  # type: ignore[operator]

        logic = solve_logic(
            "If a report is approved, it is archived. This report is not archived. "
            "Can it be approved? Answer Yes or No."
        )
        self.assertIsNotNone(logic)
        self.assertEqual(logic.answer, "No")

        sql_prompt = (
            "Write an SQL query that returns the names of customers who have placed no orders. "
            "Tables are customers(id, name) and orders(customer_id)."
        )
        sql = solve_code_generation(sql_prompt)
        self.assertIsNotNone(sql)
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
            connection.execute("CREATE TABLE orders (customer_id INTEGER)")
            connection.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "Ada"), (2, "Bo")])
            connection.execute("INSERT INTO orders VALUES (2)")
            self.assertEqual(connection.execute(sql.answer).fetchall(), [("Ada",)])

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

    def test_retired_public_multi_step_math(self) -> None:
        inventory = solve_math(
            "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
            "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
            "How many units remain at the end of Q3?"
        )
        self.assertIsNotNone(inventory)
        self.assertEqual(inventory.answer, "1672")

        recipe = solve_math(
            "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed "
            "for 30 cookies? If sugar costs $2.40 per cup, what is the total cost?"
        )
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.answer, "1.875 cups; $4.50")

    def test_practice_q11_to_q20_math_is_deterministic(self) -> None:
        cases = {
            "A company has 5,000 employees. 60% are in the US, 25% in Europe, and the rest in Asia. How many employees are in Asia?": "750",
            "If a train travels at 80 km/h for 2.5 hours, then at 100 km/h for 1.5 hours, what is the total distance traveled?": "350 km",
            "A store offers a 15% discount on a laptop originally priced at $1,200. After the discount, sales tax of 8% is added. What is the final price?": "$1101.60",
            "A project requires 120 hours of work. If 4 people work on it for 6 hours per day, how many days will it take to complete?": "5 days",
            "A population of bacteria doubles every 3 hours. If it starts with 500 bacteria, how many will there be after 12 hours?": "8000",
            "A rectangle has a perimeter of 48 meters and a length that is twice its width. What are the dimensions of the rectangle?": "Length 16 m; Width 8 m",
            "An investment of $5,000 grows at 4% annual interest compounded annually. What is the value after 3 years?": "$5624.32",
            "A pizza is cut into 8 slices. If you eat 3/8 of the pizza and your friend eats 1/4 of the pizza, what fraction of the pizza is left?": "3/8",
            "A car rental company charges $45 per day plus $0.25 per mile driven. If a customer rents for 4 days and drives 300 miles, what is the total cost?": "$255.00",
            "A container holds 2.5 liters of liquid. How many 200-milliliter cups can be filled from this container?": "12.5 cups",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_math(prompt)
                self.assertIsNotNone(solved)
                self.assertEqual(solved.answer, expected)
                self.assertGreaterEqual(solved.confidence, 0.98)

    def test_requested_mixed_sentiment_reason_uses_verified_local_contrast(self) -> None:
        prompts = (
            "Classify as Positive, Negative, or Neutral and give a one-sentence reason: "
            "'Delivery was late and the box was damaged, but the item worked perfectly "
            "and support resolved the issue.'",
            "Classify the sentiment as Positive, Negative, or Neutral and give a one-sentence "
            "reason: 'The meal was delicious, but service was slow and the waiter was rude.'",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                solved = solve_prompt(prompt, client=FakeFireworksClient())
                self.assertEqual(solved.source, "local:mixed_contrast_reason")
                self.assertEqual(solved.tokens_used, 0)
                self.assertEqual(sum(solved.answer.count(mark) for mark in ".!?"), 1)

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

    def test_verified_pure_and_factual_sentiment_routes_locally(self) -> None:
        cases = {
            "Classify the sentiment: The service is unreliable and frustrating.": "negative",
            "Classify as positive, negative, or neutral: The package arrived on Tuesday as scheduled.": "neutral",
            "Classify the sentiment: I don't think this is good.": "negative",
            (
                "Classify the sentiment as positive, negative, neutral, or mixed: "
                "The update fixed the crash, but startup is now painfully slow."
            ): "mixed",
            (
                "Classify the sentiment as positive, negative, neutral, or mixed: "
                "The battery is great, but the screen scratches easily."
            ): "mixed",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_prompt(prompt, client=FakeFireworksClient())
                self.assertEqual(solved.answer, expected)
                self.assertEqual(solved.tokens_used, 0)
                self.assertTrue(solved.source.startswith("local:"))

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

    def test_fixture_routing_keeps_unverifiable_semantic_tasks_remote(self) -> None:
        fixture = json.loads(Path("eval/fixtures/held_out.json").read_text(encoding="utf-8"))
        responses = {str(item["prompt"]): str(item.get("fake_answer", "ok")) for item in fixture}
        client = FakeFireworksClient(responses)
        for item in fixture:
            solve_prompt(str(item["prompt"]), client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["category"], "factual_qa")


if __name__ == "__main__":
    unittest.main()
