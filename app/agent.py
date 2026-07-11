from __future__ import annotations

from dataclasses import dataclass
import re
import sys

from fireworks.client import FireworksClient, FireworksError
from router.classify import classify_prompt
from solvers.code_debug_solver import solve_code_debug
from solvers.code_generation_solver import solve_code_generation
from solvers.factual_solver import solve_factual
from solvers.logic_solver import solve_logic
from solvers.math_solver import solve_math
from solvers.ner_solver import solve_ner
from solvers.sentiment_solver import solve_sentiment
from solvers.summarization_solver import solve_summarization


@dataclass(frozen=True)
class SolveResult:
    answer: str
    category: str
    source: str
    tokens_used: int | None = None


LOCAL_SOLVERS = {
    "math": solve_math,
    "sentiment": solve_sentiment,
    "summarization": solve_summarization,
    "ner": solve_ner,
    "logic": solve_logic,
    "code_debugging": solve_code_debug,
    "code_generation": solve_code_generation,
    "factual_qa": solve_factual,
}

LOCAL_MIN_CONFIDENCE = {
    "math": 0.9,
    "logic": 0.97,
}

# Semantic categories remain remote-first except for narrow methods whose outputs are
# structurally verifiable. The threshold is per method, not per broad category, so an unfamiliar
# wording falls through to Fireworks instead of receiving a plausible local guess.
VERIFIED_LOCAL_METHODS = {
    "sentiment": {
        "lexicon": 0.91,
        "factual_neutral": 0.92,
        "mixed_lexicon": 0.89,
    },
    "summarization": {
        "short_bullet_extraction": 0.93,
        "already_one_sentence": 0.92,
        "within_word_limit_passthrough": 0.99,
        "two_sentence_join": 0.96,
    },
    "ner": {
        "regex_entities": 0.9,
    },
    "code_debugging": {
        "missing_colon_repair": 0.97,
        "extremum_repair": 0.98,
        "len_index_repair": 0.99,
        "mutable_default_repair": 0.99,
    },
    "code_generation": {
        "second_largest_generation": 0.99,
        "palindrome_generation": 0.99,
        "balanced_brackets_generation": 0.99,
        "merge_intervals_generation": 0.99,
        "grouped_average_sql_generation": 0.99,
    },
    "factual_qa": {
        "stdlib_http_status": 0.99,
    },
}

CODE_DEBUG_DIAGNOSES = {
    "missing_colon_repair": "Bug: a Python block header is missing its trailing colon.",
    "extremum_repair": "Bug: the function returns one element instead of computing the requested extremum.",
    "len_index_repair": "Bug: len(sequence) is one past the final valid index.",
    "mutable_default_repair": "Bug: a mutable default argument is shared between calls.",
}


def solve_prompt(prompt: str, client: FireworksClient | None = None) -> SolveResult:
    classification = classify_prompt(prompt)
    solver = LOCAL_SOLVERS.get(classification.category)

    if solver is not None:
        local = solver(prompt)
        if local is not None and _can_use_local(
            prompt,
            classification.category,
            local.method,
            local.confidence,
        ):
            return SolveResult(
                answer=_format_local_answer(prompt, classification.category, local.method, local.answer),
                category=classification.category,
                source=f"local:{local.method}",
                tokens_used=0,
            )

    fireworks_client: FireworksClient | None = None
    try:
        fireworks_client = client or FireworksClient.from_env()
        answer = fireworks_client.solve(prompt, classification.category)
        return SolveResult(
            answer=answer.strip(),
            category=classification.category,
            source="fireworks",
            tokens_used=getattr(fireworks_client, "last_tokens_used", None),
        )
    except FireworksError as exc:
        print(
            f"warning: Fireworks fallback failed for {classification.category}: {exc}",
            file=sys.stderr,
        )
        return SolveResult(
            answer=_last_resort_answer(prompt, classification.category),
            category=classification.category,
            source="fallback",
            tokens_used=(
                getattr(fireworks_client, "last_tokens_used", None)
                if fireworks_client is not None
                else None
            ),
        )


def _can_use_local(prompt: str, category: str, method: str, confidence: float) -> bool:
    if re.search(
        r"\b(?:explain|explanation|justify|reasoning|derive|step[- ]by[- ]step|"
        r"show\s+(?:your\s+)?(?:work|steps?))\b",
        prompt,
        re.I,
    ):
        return False
    if category in LOCAL_MIN_CONFIDENCE:
        return confidence >= LOCAL_MIN_CONFIDENCE[category]
    verified = VERIFIED_LOCAL_METHODS.get(category, {})
    threshold = verified.get(method)
    if threshold is None or confidence < threshold:
        return False
    if category == "ner" and re.search(
        r"\b(?:table|csv|yaml|xml|markdown|one\s+per\s+line)\b",
        prompt,
        re.I,
    ):
        return False
    return True


def _format_local_answer(prompt: str, category: str, method: str, answer: str) -> str:
    cleaned = answer.strip()
    if category != "code_debugging" or re.search(
        r"\b(?:only|just)\s+(?:return|output|provide|show)\b.{0,30}\b(?:code|fix)\b|"
        r"\b(?:code|fix)\s+only\b",
        prompt,
        re.I,
    ):
        return cleaned
    diagnosis = CODE_DEBUG_DIAGNOSES.get(method)
    if diagnosis is None:
        return cleaned
    return f"{diagnosis}\n\n```python\n{cleaned}\n```"


def _last_resort_answer(prompt: str, category: str) -> str:
    solver = LOCAL_SOLVERS.get(category)
    if solver is not None:
        local = solver(prompt)
        if local is not None and local.answer.strip():
            return local.answer.strip()
    if category == "sentiment":
        return "neutral"
    if category == "ner":
        return "[]"
    return ""
