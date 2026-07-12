from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.agent import SolveResult, solve_prompt
from app.main import run_batch
from eval.run_local_eval import FakeFireworksClient
from fireworks.client import (
    FireworksClient,
    FireworksError,
    clean_model_answer,
    model_id_for_request,
    normalize_model_id,
    parse_allowed_models,
    validate_model_answer,
)
from router.classify import classify_prompt
from solvers.code_debug_solver import solve_code_debug
from solvers.factual_solver import solve_factual
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment
from solvers.summarization_solver import solve_summarization
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

    def test_standard_concept_comparisons_use_local_only_when_complete(self) -> None:
        cases = {
            (
                "Compare CPUs and GPUs. Explain how their core architecture and parallelism differ, "
                "and identify the workloads each is best suited for."
            ): "sequential",
            (
                "What is the difference between supervised and unsupervised machine learning? "
                "Explain the training data, goal, and one typical task for each."
            ): "labeled",
            (
                "Compare RAM and ROM by volatility, speed, and what each is used for in a computer."
            ): "firmware",
            "What is the difference between HTTP and HTTPS? Explain TLS encryption and web security.": "TLS",
            (
                "What is the difference between machine learning and deep learning? Explain feature "
                "engineering and neural networks."
            ): "manual feature",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                solved = solve_factual(prompt)
                self.assertIsNotNone(solved)
                self.assertEqual(solved.method, "standard_concept_comparison")
                self.assertIn(expected.lower(), solved.answer.lower())

        for prompt in (
            "Compare RAM and ROM in terms of manufacturing cost only.",
            "Compare CPUs and GPUs for power consumption in laptops.",
            "Compare HTTP and HTTPS by adoption rates in 2026.",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(solve_factual(prompt))

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
        discounted = solve_math(
            "A price is 100 dollars. It rises by 20%, then is discounted by 15%. "
            "What is the final price?"
        )
        self.assertIsNotNone(discounted)
        self.assertEqual(discounted.answer, "102")
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

    def test_sentiment_factual_neutral_and_contracted_negation(self) -> None:
        neutral = solve_sentiment(
            "Classify as positive, negative, or neutral: The package arrived on Tuesday as scheduled."
        )
        self.assertIsNotNone(neutral)
        self.assertEqual(neutral.answer, "neutral")
        self.assertGreaterEqual(neutral.confidence, 0.82)
        negated = solve_sentiment(
            'Classify the sentiment: "I don\'t think this is good."'
        )
        self.assertIsNotNone(negated)
        self.assertEqual(negated.answer, "negative")
        self.assertEqual(
            solve_sentiment("Classify the sentiment: The package arrived late on Tuesday.").answer,
            "negative",
        )

    def test_sentiment_negation_boundaries_and_allowed_labels(self) -> None:
        separated = solve_sentiment(
            "Classify as positive, negative, or neutral: Not fast. It is reliable."
        )
        self.assertIsNotNone(separated)
        self.assertEqual(separated.answer, "neutral")

        reordered = solve_sentiment(
            "Classify as negative, neutral, or positive: "
            "The battery is great, but the app crashes."
        )
        self.assertIsNotNone(reordered)
        self.assertEqual(reordered.answer, "neutral")

        binary = solve_sentiment(
            "Classify as positive or negative: The battery is great, but the app crashes."
        )
        self.assertIsNone(binary)

    def test_short_bullet_summary_is_local_only_when_structure_is_safe(self) -> None:
        solved = solve_summarization(
            "Summarize this update in two bullet points: "
            "The release fixes login failures on older phones. "
            "It also reduces image upload time and adds clearer error messages."
        )
        self.assertIsNotNone(solved)
        self.assertEqual(len(solved.answer.splitlines()), 2)
        self.assertIn("login", solved.answer)
        self.assertIn("upload", solved.answer)
        self.assertIsNone(
            solve_summarization(
                "Summarize in two bullet points: This is only one source sentence."
            )
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

    def test_ner_location_prefixes_and_org_suffixes(self) -> None:
        solved = solve_ner(
            "Extract named entities and types from: Maya Chen visited Mount Everest "
            "after meeting the European Union in Brussels."
        )
        self.assertIsNotNone(solved)
        entities = json.loads(solved.answer)
        self.assertIn({"text": "Mount Everest", "label": "LOCATION"}, entities)
        self.assertIn({"text": "European Union", "label": "ORG"}, entities)

    def test_ner_ambiguous_capitals_fall_back_instead_of_guessing_person(self) -> None:
        location = solve_ner(
            "Extract named entities and types from: Alice Johnson visited New Delhi."
        )
        self.assertIsNotNone(location)
        entities = json.loads(location.answer)
        self.assertIn({"text": "Alice Johnson", "label": "PERSON"}, entities)
        self.assertIn({"text": "New Delhi", "label": "LOCATION"}, entities)

        ambiguous = solve_ner(
            "Extract named entities and types from: Cedar Grove hosted the summit."
        )
        self.assertIsNone(ambiguous)

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
        modus_ponens = solve_logic(
            "If the alarm is armed, the light is green. The alarm is armed. "
            "Is the light green?"
        )
        self.assertIsNotNone(modus_ponens)
        self.assertEqual(modus_ponens.answer, "Yes")
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
        implicit_mutable = solve_code_debug(
            "Find and fix the Python bug that makes values persist across calls:\n"
            "```python\ndef add_name(name, names=[]):\n"
            "    names.append(name)\n    return names\n```"
        )
        self.assertIsNotNone(implicit_mutable)
        self.assertIn("names=None", implicit_mutable.answer)
        self.assertIn("if names is None", implicit_mutable.answer)
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

    def test_classifier_avoids_broad_instruction_false_positives(self) -> None:
        cases = {
            "What is the overall time complexity of binary search?": "factual_qa",
            "Classify the following triangle by its side lengths: 3, 3, 5.": "factual_qa",
            "Can you determine how many apples remain after selling 12?": "math",
            "What can you infer from the HTTP 404 response?": "factual_qa",
            "Fix this sentence for grammar: She go to school.": "factual_qa",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_prompt(prompt).category, expected)

        self.assertEqual(
            classify_prompt("Fix this Python function; it returns wrong output.").category,
            "code_debugging",
        )

    def test_code_explanation_does_not_execute_embedded_math_prompt(self) -> None:
        prompt = (
            "what is this code for, i dont understand\n\n"
            "def test_math_average_and_linear_equation(self):\n"
            "    self.assertEqual(solve_math(\"Find the average of 4, 6, and 8.\").answer, \"6\")"
        )
        explanation = "This unit test checks that the math solver calculates an average correctly."
        solved = solve_prompt(prompt, client=FakeFireworksClient({prompt: explanation}))
        self.assertEqual(solved.category, "factual_qa")
        self.assertEqual(solved.source, "fireworks")
        self.assertEqual(solved.answer, explanation)

        explanation_prompts = (
            "Explain this code: ```python\ndef total(values): return sum(values)\n```",
            "Help me understand this snippet: def first(items): return items[0]",
            "Walk me through this code: class User: pass",
        )
        for explanation_prompt in explanation_prompts:
            with self.subTest(prompt=explanation_prompt):
                self.assertEqual(classify_prompt(explanation_prompt).category, "factual_qa")

        self.assertEqual(
            classify_prompt(
                "Explain this code and fix the bug: ```python\ndef total(values): return values[0]\n```"
            ).category,
            "code_debugging",
        )

    def test_classifier_distinguishes_concepts_from_requested_tasks(self) -> None:
        factual_questions = (
            "What is sentiment analysis?",
            "Explain named entity recognition and why it is useful.",
            "What is a logic puzzle?",
            "What is the main idea behind binary search?",
            "What are the key points of the TCP three-way handshake?",
            "Explain how to write a function in Python.",
            "Explain what a TypeError means in Python.",
        )
        for prompt in factual_questions:
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_prompt(prompt).category, "factual_qa")

        requested_tasks = {
            'Analyze the sentiment of: "The setup is excellent."': "sentiment",
            "Extract the named entities from: Ada joined Acme Inc.": "ner",
            "Summarize the following passage: The launch was delayed by rain.": "summarization",
            "Provide a summary of this report: Revenue rose while costs stayed flat.": "summarization",
            "Can you write a Python function that reverses a list?": "code_generation",
            "I get a TypeError in this Python function; why does it fail?": "code_debugging",
        }
        for prompt, expected in requested_tasks.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_prompt(prompt).category, expected)

    def test_classifier_handles_hidden_style_instruction_variants(self) -> None:
        cases = {
            'Determine whether this review is favorable or unfavorable: "It works perfectly."': "sentiment",
            "Identify all people, companies, cities, and dates in the passage.": "ner",
            "Give a one-sentence overview of this report: Revenue rose and costs fell.": "summarization",
            "Spot the defect in this Python function and repair it.": "code_debugging",
            "Provide SQL that lists every active customer.": "code_generation",
            "Given these constraints, which assignment must be true?": "logic",
            "Solve for x: x + y = z.": "math",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_prompt(prompt).category, expected)

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
        self.assertEqual(client.pick_model("math"), "kimi-k2p7-code")
        self.assertEqual(client.pick_model("summarization"), "kimi-k2p7-code")
        self.assertEqual(client.pick_model("sentiment"), "kimi-k2p7-code")
        self.assertEqual(client.pick_model("logic"), "kimi-k2p7-code")
        self.assertEqual(
            client.candidate_models_for_prompt(
                "factual_qa",
                "What is the difference between machine learning and deep learning?",
            )[0],
            "minimax-m3",
        )
        self.assertEqual(
            client.candidate_models_for_prompt(
                "summarization",
                "Summarize in exactly three bullet points: text",
            )[0],
            "minimax-m3",
        )
        self.assertEqual(
            client.candidate_models_for_prompt(
                "sentiment",
                "Classify as Positive or Negative and give one sentence of reasoning: text",
            )[0],
            "minimax-m3",
        )

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
        self.assertEqual(
            clean_model_answer('```json\n[{"entity":"Ada","type":"PERSON"}]\n```', "ner"),
            '[{"entity":"Ada","type":"PERSON"}]',
        )
        self.assertEqual(
            clean_model_answer(
                "Ada: PERSON\nAcme: ORGANIZATION\nParis: GPE",
                "ner",
            ),
            "Ada: PERSON\nAcme: ORGANIZATION\nParis: LOCATION",
        )
        self.assertEqual(
            clean_model_answer(
                "Google: ORG\nZurich: GPE",
                "ner",
                "Label each as PERSON, ORGANIZATION, LOCATION, or DATE: text",
            ),
            "Google: ORGANIZATION\nZurich: LOCATION",
        )
        self.assertEqual(
            clean_model_answer(
                "World Health Organization - ORG",
                "ner",
                "Label as PERSON, ORGANIZATION, LOCATION, or DATE: text",
            ),
            "World Health Organization - ORGANIZATION",
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
                "Favorable.",
                "sentiment",
                "Determine whether this review is favorable or unfavorable.",
            ),
            "favorable",
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
            clean_model_answer(
                "Negative. The update deleted settings and crashes repeatedly.",
                "sentiment",
                "Classify and give exactly one sentence of reasoning: text",
            ),
            "Negative — the update deleted settings and crashes repeatedly.",
        )
        self.assertEqual(
            normalize_model_id("accounts/fireworks/models/minimax-m3"),
            "accounts/fireworks/models/minimax-m3",
        )

    def test_model_answer_validation_checks_python_and_summary_constraints(self) -> None:
        validate_model_answer(
            "def square(n):\n    return n * n",
            "code_generation",
            "Write a Python function square(n).",
        )
        with self.assertRaises(FireworksError):
            validate_model_answer(
                "def square(:\n    pass",
                "code_generation",
                "Write a Python function square(n).",
            )
        validate_model_answer(
            "- First item.\n- Second item.",
            "summarization",
            "Summarize in two bullet points: text",
        )
        with self.assertRaises(FireworksError):
            validate_model_answer(
                "Only one line.",
                "summarization",
                "Summarize in two bullet points: text",
            )
        with self.assertRaisesRegex(FireworksError, "per-bullet limit"):
            validate_model_answer(
                "- This bullet contains far too many words for the strict limit requested here.\n"
                "- Short second bullet.",
                "summarization",
                "Summarize in exactly two bullet points, each no longer than 8 words: text",
            )
        with self.assertRaisesRegex(FireworksError, "omitted.*sentiment reason"):
            validate_model_answer(
                "positive",
                "sentiment",
                "Classify and give exactly one sentence of reasoning: Great product.",
            )

    def test_code_debug_validation_requires_diagnosis_and_valid_fix(self) -> None:
        prompt = (
            "This function should return the max of a list but has a bug: "
            "def get_max(nums): return nums[0]. Find and fix it."
        )
        complete = (
            "The bug is that the function always returns the first item instead of the maximum.\n\n"
            "```python\ndef get_max(nums):\n    return max(nums)\n```"
        )
        validate_model_answer(complete, "code_debugging", prompt)

        with self.assertRaisesRegex(FireworksError, "omitted.*diagnosis"):
            validate_model_answer(
                "def get_max(nums):\n    return max(nums)",
                "code_debugging",
                prompt,
            )

        with self.assertRaisesRegex(FireworksError, "invalid Python fix"):
            validate_model_answer(
                "The function returns the wrong item.\n\n```python\ndef get_max(:\n    pass\n```",
                "code_debugging",
                prompt,
            )

        with self.assertRaisesRegex(FireworksError, "omitted.*Python fix"):
            validate_model_answer(
                "The function returns the first item instead of the maximum.",
                "code_debugging",
                prompt,
            )

        validate_model_answer(
            "The TypeError means the function combines incompatible operand types.",
            "code_debugging",
            "I get a TypeError in this Python function; why does it fail?",
        )
        validate_model_answer(
            "The Java method returns the wrong value; return the computed total instead.",
            "code_debugging",
            "Fix this Java method: ```java\nint total() { return 0; }\n```",
        )

        validate_model_answer(
            "def get_max(nums):\n    return max(nums)",
            "code_debugging",
            prompt + " Return code only.",
        )

    def test_remote_category_uses_injected_client(self) -> None:
        solved = solve_prompt("What is the capital city of Japan?", client=FakeFireworksClient())
        self.assertEqual(solved.category, "factual_qa")
        self.assertEqual(solved.source, "fireworks")
        self.assertEqual(solved.answer, "Tokyo")

    def test_accuracy_first_keeps_semantic_categories_remote(self) -> None:
        prompts = {
            "Extract named entities from: Maya Chen visited Paris.": "Maya Chen PERSON Paris LOCATION",
            "This Python function has a bug: def top(xs): return xs[0]. Fix it.": "def top(xs):\n    return max(xs)",
        }
        client = FakeFireworksClient(prompts)
        for prompt, expected in prompts.items():
            with self.subTest(prompt=prompt):
                solved = solve_prompt(prompt, client=client)
                self.assertEqual(solved.source, "fireworks")
                self.assertEqual(solved.answer, expected)

        sentiment_prompt = 'Classify the sentiment: "This is excellent."'
        client.responses[sentiment_prompt] = "positive"
        safe_sentiment = solve_prompt(sentiment_prompt, client=client)
        self.assertEqual(safe_sentiment.source, "fireworks")
        self.assertEqual(safe_sentiment.answer, "positive")

        ambiguous_sentiment = 'Classify the sentiment: "It might be good, but the result is unclear."'
        client.responses[ambiguous_sentiment] = "neutral"
        self.assertEqual(solve_prompt(ambiguous_sentiment, client=client).source, "fireworks")

        summary_prompt = "Summarize in one sentence: The release is stable and fast."
        client.responses[summary_prompt] = "The release is stable and fast."
        safe_summary = solve_prompt(summary_prompt, client=client)
        self.assertEqual(safe_summary.source, "fireworks")
        self.assertEqual(safe_summary.answer, "The release is stable and fast.")

        complex_summary = "Summarize this report in one sentence: Revenue increased. Costs fell. Risks remain."
        client.responses[complex_summary] = "Revenue increased and costs fell, though risks remain."
        solved_summary = solve_prompt(complex_summary, client=client)
        self.assertEqual(solved_summary.source, "fireworks")

        local_math = solve_prompt("What is 7 * (8 + 2)?", client=client)
        self.assertEqual(local_math.source, "local:safe_eval")

        local_truth = solve_prompt(
            'Logic puzzle: exactly one statement is true. Ava says "Ben is lying". '
            'Ben says "Ava is lying". Cora says "Ava is lying". Who is telling the truth?',
            client=client,
        )
        self.assertEqual(local_truth.source, "local:exactly_one_truth")
        self.assertEqual(local_truth.answer, "Ava")

    def test_explanation_requests_escalate_exact_local_answers(self) -> None:
        math_prompt = "What is 7 * (8 + 2)? Show your work."
        logic_prompt = (
            "All red keys open the vault. Key A is red. Does Key A open the vault? "
            "Explain your reasoning."
        )
        client = FakeFireworksClient(
            {
                math_prompt: "7 * 10 = 70, so the answer is 70.",
                logic_prompt: "Yes. Key A is red, and all red keys open the vault.",
            }
        )
        self.assertEqual(solve_prompt(math_prompt, client=client).source, "fireworks")
        self.assertEqual(solve_prompt(logic_prompt, client=client).source, "fireworks")

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

    def test_fireworks_retries_without_reasoning_effort_and_disables_it(self) -> None:
        class ProxyCompatibilityClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://judge.example/proxy/v1",
                    allowed_models=["kimi-k2p7-code"],
                )
                self.payloads: list[dict[str, object]] = []

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.payloads.append(dict(payload))
                if "reasoning_effort" in payload:
                    raise FireworksError("HTTP 400: unsupported field", status_code=400)
                return {"choices": [{"message": {"content": "Canberra"}}]}

        client = ProxyCompatibilityClient()
        self.assertEqual(client.solve("What is Australia's capital?", "factual_qa"), "Canberra")
        self.assertEqual(client.solve("Name Australia's capital.", "factual_qa"), "Canberra")
        self.assertEqual(len(client.payloads), 3)
        self.assertIn("reasoning_effort", client.payloads[0])
        self.assertNotIn("reasoning_effort", client.payloads[1])
        self.assertNotIn("reasoning_effort", client.payloads[2])

    def test_fireworks_retries_invalid_generated_python(self) -> None:
        class ValidationRetryClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://api.fireworks.ai/inference/v1",
                    allowed_models=["kimi-k2p7-code", "minimax-m3"],
                )
                self.models: list[str] = []

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.models.append(payload["model"])
                if payload["model"].endswith("kimi-k2p7-code"):
                    content = "```python\ndef broken(:\n    pass\n```"
                else:
                    content = "```python\ndef square(n):\n    return n * n\n```"
                return {"choices": [{"message": {"content": content}}]}

        client = ValidationRetryClient()
        answer = client.solve("Write a Python function square(n).", "code_generation")
        self.assertEqual(answer, "def square(n):\n    return n * n")
        self.assertEqual(len(client.models), 2)

    def test_fireworks_tracks_tokens_across_retries(self) -> None:
        class UsageClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://api.fireworks.ai/inference/v1",
                    allowed_models=["kimi-k2p7-code", "minimax-m3"],
                )
                self.calls = 0

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.calls += 1
                content = "def broken(:\n    pass" if self.calls == 1 else "def square(n):\n    return n * n"
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"total_tokens": 10 if self.calls == 1 else 7},
                }

        client = UsageClient()
        self.assertEqual(
            client.solve("Write a Python function square(n).", "code_generation"),
            "def square(n):\n    return n * n",
        )
        self.assertEqual(client.last_tokens_used, 17)

    def test_fireworks_retries_truncated_answers(self) -> None:
        class TruncationRetryClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://api.fireworks.ai/inference/v1",
                    allowed_models=["minimax-m3", "kimi-k2p7-code"],
                )
                self.calls = 0

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.calls += 1
                if self.calls == 1:
                    return {
                        "choices": [
                            {"message": {"content": "Sam owns"}, "finish_reason": "length"}
                        ]
                    }
                return {
                    "choices": [
                        {"message": {"content": "Sam owns the cat."}, "finish_reason": "stop"}
                    ]
                }

        client = TruncationRetryClient()
        self.assertEqual(client.solve("Who owns the cat?", "logic"), "Sam owns the cat.")
        self.assertEqual(client.calls, 2)

    def test_fireworks_recovers_complete_factual_sentence_from_truncation(self) -> None:
        class FactualTruncationClient(FireworksClient):
            def __init__(self) -> None:
                super().__init__(
                    api_key="test",
                    base_url="https://api.fireworks.ai/inference/v1",
                    allowed_models=["kimi-k2p7-code", "minimax-m3"],
                )
                self.calls = 0

            def _post_json(self, path, payload):  # type: ignore[no-untyped-def]
                self.calls += 1
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Sentiment analysis classifies emotion in text. Extra unfinished"
                            },
                            "finish_reason": "length",
                        }
                    ]
                }

        client = FactualTruncationClient()
        self.assertEqual(
            client.solve("What is sentiment analysis?", "factual_qa"),
            "Sentiment analysis classifies emotion in text.",
        )
        self.assertEqual(client.calls, 1)

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

    def test_batch_reuses_one_fireworks_client_for_all_tasks(self) -> None:
        tasks = [
            {"task_id": "one", "prompt": "First question?"},
            {"task_id": "two", "prompt": "Second question?"},
        ]
        shared_client = FakeFireworksClient()
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "tasks.json"
            output_path = Path(tmp) / "results.json"
            input_path.write_text(json.dumps(tasks), encoding="utf-8")
            with patch("app.main.FireworksClient.from_env", return_value=shared_client):
                with patch(
                    "app.main.solve_prompt",
                    side_effect=[
                        SolveResult("first", "factual_qa", "fireworks"),
                        SolveResult("second", "factual_qa", "fireworks"),
                    ],
                ) as mocked_solve:
                    run_batch(input_path, output_path)

        self.assertEqual(mocked_solve.call_count, 2)
        for call in mocked_solve.call_args_list:
            self.assertIs(call.kwargs["client"], shared_client)


if __name__ == "__main__":
    unittest.main()
