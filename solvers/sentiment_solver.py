from __future__ import annotations

import re

from solvers.common import LocalAnswer


POSITIVE = {
    "accurate",
    "amazing",
    "awesome",
    "best",
    "comfortable",
    "convenient",
    "delightful",
    "durable",
    "easy",
    "efficient",
    "excellent",
    "fantastic",
    "fast",
    "good",
    "great",
    "happy",
    "helpful",
    "impressive",
    "intuitive",
    "love",
    "loved",
    "perfect",
    "pleasant",
    "powerful",
    "recommend",
    "reliable",
    "satisfied",
    "smooth",
    "stable",
    "useful",
    "worth",
    "wonderful",
}

NEGATIVE = {
    "annoying",
    "awful",
    "bad",
    "buggy",
    "broken",
    "crash",
    "crashed",
    "confusing",
    "crashes",
    "difficult",
    "disappointing",
    "disappointed",
    "expensive",
    "fail",
    "failed",
    "failing",
    "flimsy",
    "frustrating",
    "hate",
    "hated",
    "horrible",
    "laggy",
    "limited",
    "poor",
    "problem",
    "problems",
    "refund",
    "scratch",
    "scratched",
    "scratches",
    "slow",
    "stuck",
    "terrible",
    "unhappy",
    "unreliable",
    "useless",
    "worst",
}

NEGATORS = {"not", "never", "no", "hardly", "barely"}


def solve_sentiment(prompt: str) -> LocalAnswer | None:
    if re.search(r"\b(justify|explain|give (?:a )?reason|why)\b", prompt, re.I):
        return None

    text = _strip_instruction(prompt)
    if any(
        cue in text.lower()
        for cue in ("yeah right", "as if", "just great", "thanks a lot", "what a surprise", "/s")
    ):
        return None
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return None
    positive_hits = 0
    negative_hits = 0
    for idx, token in enumerate(tokens):
        if token in POSITIVE or token in NEGATIVE:
            sign = 1 if token in POSITIVE else -1
            if _is_negated(tokens, idx):
                sign *= -1
            if sign > 0:
                positive_hits += 1
            else:
                negative_hits += 1
    hits = positive_hits + negative_hits
    if hits == 0:
        return LocalAnswer("neutral", 0.72, "lexicon")
    if positive_hits and negative_hits:
        constrained_three_way = bool(
            re.search(
                r"positive\s*,?\s*negative\s*,?\s*(?:or|and)\s*neutral",
                prompt,
                re.I,
            )
        )
        label = "neutral" if constrained_three_way else "mixed"
        return LocalAnswer(label, min(0.95, 0.84 + hits * 0.03), "mixed_lexicon")
    if positive_hits:
        return LocalAnswer("positive", min(0.96, 0.72 + hits * 0.1), "lexicon")
    if negative_hits:
        return LocalAnswer("negative", min(0.96, 0.72 + hits * 0.1), "lexicon")
    return None


def _is_negated(tokens: list[str], index: int) -> bool:
    start = max(0, index - 3)
    for negator_index in range(start, index):
        if tokens[negator_index] not in NEGATORS:
            continue
        if tokens[negator_index] == "not" and negator_index + 1 < index:
            if tokens[negator_index + 1] == "only":
                continue
        return True
    return False


def _strip_instruction(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    pieces = [a or b for a, b in quoted if a or b]
    return pieces[-1] if pieces else prompt
