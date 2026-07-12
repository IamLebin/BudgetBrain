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
    "delicious",
    "durable",
    "easy",
    "efficient",
    "excellent",
    "fantastic",
    "fast",
    "flawless",
    "friendly",
    "good",
    "great",
    "happy",
    "helpful",
    "impressive",
    "intuitive",
    "like",
    "liked",
    "love",
    "loved",
    "perfect",
    "pleasant",
    "powerful",
    "quick",
    "recommend",
    "reliable",
    "responsive",
    "resolved",
    "replacement",
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
    "damaged",
    "dented",
    "difficult",
    "disappointing",
    "disappointed",
    "defective",
    "delayed",
    "expensive",
    "fail",
    "failed",
    "failing",
    "flimsy",
    "frustrating",
    "hate",
    "hated",
    "horrible",
    "inaccurate",
    "laggy",
    "limited",
    "late",
    "missing",
    "mediocre",
    "poor",
    "problem",
    "problems",
    "refund",
    "scratch",
    "scratched",
    "scratches",
    "slow",
    "rude",
    "noisy",
    "overcrowded",
    "impossible",
    "stuck",
    "terrible",
    "underwhelming",
    "unhappy",
    "unreliable",
    "useless",
    "worst",
}

NEGATORS = {
    "not",
    "never",
    "no",
    "hardly",
    "barely",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "wasn't",
    "weren't",
    "can't",
    "couldn't",
    "wouldn't",
    "shouldn't",
}

NEGATION_BOUNDARIES = {".", "!", "?", ";", ",", "but", "however", "although", "yet"}


def solve_sentiment(prompt: str) -> LocalAnswer | None:
    instruction = prompt.split(":", maxsplit=1)[0]
    if re.search(r"\b(justify|explain|reason|why)\b", instruction, re.I):
        return _solve_mixed_contrast_reason(prompt)

    text = _strip_instruction(prompt)
    if any(
        cue in text.lower()
        for cue in ("yeah right", "as if", "just great", "thanks a lot", "what a surprise", "/s")
    ):
        return None
    tokens = re.findall(r"[a-z']+|[.!?;,]", text.lower())
    if not tokens:
        return None
    positive_hits = 0
    negative_hits = 0
    negated_hits = 0
    for idx, token in enumerate(tokens):
        if token in POSITIVE or token in NEGATIVE:
            sign = 1 if token in POSITIVE else -1
            if _is_negated(tokens, idx):
                sign *= -1
                negated_hits += 1
            if sign > 0:
                positive_hits += 1
            else:
                negative_hits += 1
    works_perfectly = re.search(
        r"\b(?:works?|worked|working|runs?|ran|functions?|functioned|operates?|operated)\s+perfectly\b",
        text,
        re.I,
    )
    negated_perfectly = re.search(
        r"\b(?:not|never|hardly|barely|doesn['’]?t|didn['’]?t|isn['’]?t)\b.{0,24}"
        r"\b(?:work|works|worked|run|runs|function|functions|operate|operates)\s+perfectly\b",
        text,
        re.I,
    )
    if works_perfectly and not negated_perfectly:
        positive_hits += 1
    hits = positive_hits + negative_hits
    if hits == 0:
        if _looks_clearly_factual(text):
            return LocalAnswer("neutral", 0.92, "factual_neutral")
        return LocalAnswer("neutral", 0.72, "lexicon")
    if positive_hits and negative_hits:
        allowed_labels = set(
            re.findall(
                r"\b(?:positive|negative|neutral|mixed)\b",
                prompt.split(":", maxsplit=1)[0].lower(),
            )
        )
        if allowed_labels and "mixed" not in allowed_labels:
            if {"positive", "negative", "neutral"} <= allowed_labels:
                label = "neutral"
            else:
                return None
        else:
            label = "mixed"
        return LocalAnswer(label, min(0.95, 0.84 + hits * 0.03), "mixed_lexicon")
    if hits >= 2 and _is_short_unambiguous_statement(text):
        label = "positive" if positive_hits else "negative"
        return LocalAnswer(
            _requested_label_style(label, prompt),
            0.97,
            "strong_unanimous_lexicon",
        )
    if hits == 1 and _is_short_unambiguous_statement(text):
        label = "positive" if positive_hits else "negative"
        label = _requested_label_style(label, prompt)
        method = "explicit_negated_lexicon" if negated_hits else "strong_single_lexicon"
        return LocalAnswer(label, 0.95 if negated_hits else 0.93, method)
    if positive_hits:
        return LocalAnswer("positive", min(0.96, 0.72 + hits * 0.1), "lexicon")
    if negative_hits:
        return LocalAnswer("negative", min(0.96, 0.72 + hits * 0.1), "lexicon")
    return None


def _solve_mixed_contrast_reason(prompt: str) -> LocalAnswer | None:
    instruction = prompt.split(":", maxsplit=1)[0]
    if not re.search(r"\b(?:one|1)[- ]sentence\b", instruction, re.I):
        return None
    text = _strip_instruction(prompt).strip()
    parts = re.split(r"\s*\b(?:but|however|although|yet)\b\s*", text, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None
    left, right = (_sentiment_clause(part) for part in parts)
    if not left or not right:
        return None
    left_positive, left_negative = _simple_polarity(left)
    right_positive, right_negative = _simple_polarity(right)
    if not ((left_positive and right_negative) or (left_negative and right_positive)):
        return None
    label = "Mixed" if re.search(r"\bmixed\b", instruction, re.I) else "Neutral"
    return LocalAnswer(
        f"{label} — the review contrasts {left} with {right}.",
        0.99,
        "mixed_contrast_reason",
    )


def _sentiment_clause(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.!?]+", ";", text)).strip(" \t\r\n;,'\"")


def _simple_polarity(text: str) -> tuple[bool, bool]:
    words = set(re.findall(r"[a-z']+", text.lower()))
    positive = bool(words & POSITIVE) or bool(
        re.search(r"\b(?:works?|worked|working|runs?|ran|functions?|operates?)\s+perfectly\b", text, re.I)
    )
    negative = bool(words & NEGATIVE) or bool(
        re.search(r"\b(?:broke|breaks|could\s+barely|can\s+barely)\b", text, re.I)
    )
    return positive, negative


def _is_short_unambiguous_statement(text: str) -> bool:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not (2 <= len(words) <= 14) or text.strip().endswith("?"):
        return False
    return not re.search(
        r"\b(?:but|however|although|yet|maybe|perhaps|possibly|somewhat|unclear|"
        r"apparently|supposedly|seems?|might|could|hardly|barely)\b",
        text,
        re.I,
    )


def _requested_label_style(label: str, prompt: str) -> str:
    instruction = prompt.split(":", maxsplit=1)[0].lower()
    if "favorable" in instruction or "unfavorable" in instruction:
        return "favorable" if label == "positive" else "unfavorable"
    if "favourable" in instruction or "unfavourable" in instruction:
        return "favourable" if label == "positive" else "unfavourable"
    return label


def _is_negated(tokens: list[str], index: int) -> bool:
    words_seen = 0
    for negator_index in range(index - 1, -1, -1):
        token = tokens[negator_index]
        if token in NEGATION_BOUNDARIES:
            break
        words_seen += 1
        if words_seen > 4:
            break
        if token not in NEGATORS:
            continue
        if token == "not" and negator_index + 1 < index:
            if tokens[negator_index + 1] == "only":
                continue
        return True
    return False


def _strip_instruction(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    pieces = [a or b for a, b in quoted if a or b]
    if pieces:
        return pieces[-1]
    parts = re.split(r":\s*", prompt, maxsplit=1)
    return parts[-1]


def _looks_clearly_factual(text: str) -> bool:
    lower = text.lower()
    if "!" in text or re.search(r"\b(?:but|however|although|unfortunately|fortunately)\b", lower):
        return False
    if re.search(r"\b(?:i|we)\s+(?:feel|felt|think|thought|wish|hope)\b", lower):
        return False
    temporal = re.search(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december|today|yesterday|tomorrow|\d{1,4})\b",
        lower,
    )
    factual_verb = re.search(
        r"\b(?:arrived|began|closed|contains|delivered|ended|lasted|located|measures|"
        r"met|occurred|opened|released|reported|scheduled|shipped|weighs)\b",
        lower,
    )
    return bool(temporal and factual_verb)
