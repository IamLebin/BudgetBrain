from __future__ import annotations

from dataclasses import dataclass
import re
import sys

from fireworks.client import FireworksClient, FireworksError
from router.classify import classify_prompt
from solvers.code_debug_solver import solve_code_debug
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


LOCAL_SOLVERS = {
    "math": solve_math,
    "sentiment": solve_sentiment,
    "summarization": solve_summarization,
    "ner": solve_ner,
    "logic": solve_logic,
    "code_debugging": solve_code_debug,
}

# These categories are semantic enough that a plausible rule-based answer can still miss the
# LLM judge's intended meaning. Keep their solvers for emergency fallback and unit testing, but
# use Fireworks during normal scoring.
REMOTE_FIRST_CATEGORIES = {"sentiment", "summarization", "ner", "code_debugging"}

LOCAL_MIN_CONFIDENCE = {
    "math": 0.9,
    "logic": 0.97,
}


def solve_prompt(prompt: str, client: FireworksClient | None = None) -> SolveResult:
    classification = classify_prompt(prompt)
    solver = LOCAL_SOLVERS.get(classification.category)

    if solver is not None:
        local = solver(prompt)
        if local is not None and _can_use_local(prompt, classification.category, local.confidence):
            return SolveResult(
                answer=local.answer.strip(),
                category=classification.category,
                source=f"local:{local.method}",
            )

    try:
        fireworks_client = client or FireworksClient.from_env()
        answer = fireworks_client.solve(prompt, classification.category)
        return SolveResult(
            answer=answer.strip(),
            category=classification.category,
            source="fireworks",
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
        )


def _can_use_local(prompt: str, category: str, confidence: float) -> bool:
    if category in REMOTE_FIRST_CATEGORIES:
        return False
    if confidence < LOCAL_MIN_CONFIDENCE.get(category, 1.0):
        return False
    if category in {"math", "logic"} and re.search(
        r"\b(?:explain|explanation|justify|reasoning|derive|step[- ]by[- ]step|"
        r"show\s+(?:your\s+)?(?:work|steps?))\b",
        prompt,
        re.I,
    ):
        return False
    return True


def _last_resort_answer(prompt: str, category: str) -> str:
    if category == "sentiment":
        local = solve_sentiment(prompt)
        return local.answer if local else "neutral"
    if category == "math":
        local = solve_math(prompt)
        return local.answer if local else ""
    if category == "ner":
        local = solve_ner(prompt)
        return local.answer if local else "[]"
    return ""
