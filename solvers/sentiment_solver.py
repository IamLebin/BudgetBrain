from __future__ import annotations

import re

from solvers.common import LocalAnswer


POSITIVE = {
    "amazing",
    "awesome",
    "best",
    "delightful",
    "excellent",
    "fantastic",
    "good",
    "great",
    "happy",
    "impressive",
    "love",
    "loved",
    "perfect",
    "pleasant",
    "recommend",
    "reliable",
    "smooth",
    "useful",
    "wonderful",
}

NEGATIVE = {
    "awful",
    "bad",
    "broken",
    "confusing",
    "crash",
    "crashes",
    "disappointing",
    "hate",
    "hated",
    "horrible",
    "poor",
    "refund",
    "slow",
    "stuck",
    "terrible",
    "unhappy",
    "useless",
    "worst",
}

NEGATORS = {"not", "never", "no", "hardly", "barely"}


def solve_sentiment(prompt: str) -> LocalAnswer | None:
    text = _strip_instruction(prompt)
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return None
    score = 0
    hits = 0
    for idx, token in enumerate(tokens):
        if token in POSITIVE or token in NEGATIVE:
            sign = 1 if token in POSITIVE else -1
            if any(prev in NEGATORS for prev in tokens[max(0, idx - 3) : idx]):
                sign *= -1
            score += sign
            hits += 1
    if hits == 0:
        return LocalAnswer("neutral", 0.75, "lexicon")
    if score > 0:
        return LocalAnswer("positive", min(0.96, 0.72 + hits * 0.1), "lexicon")
    if score < 0:
        return LocalAnswer("negative", min(0.96, 0.72 + hits * 0.1), "lexicon")
    return LocalAnswer("neutral", 0.84, "lexicon")


def _strip_instruction(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    pieces = [a or b for a, b in quoted if a or b]
    return pieces[-1] if pieces else prompt
