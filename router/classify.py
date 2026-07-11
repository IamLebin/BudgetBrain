from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Classification:
    category: str
    confidence: float


def classify_prompt(prompt: str) -> Classification:
    text = prompt.strip()
    lower = text.lower()

    if _looks_like_code_debugging(text, lower):
        return Classification("code_debugging", 0.94)
    if _looks_like_code_generation(lower):
        return Classification("code_generation", 0.93)
    if _looks_like_summarization(lower, text):
        return Classification("summarization", 0.94)
    if _looks_like_ner(lower):
        return Classification("ner", 0.94)
    if _looks_like_sentiment(lower):
        return Classification("sentiment", 0.93)
    if _looks_like_logic(lower):
        return Classification("logic", 0.89)
    if _looks_like_math(lower):
        return Classification("math", 0.9)
    return Classification("factual_qa", 0.55)


def _looks_like_code_generation(lower: str) -> bool:
    signals = (
        "write a function",
        "write a python function",
        "write a program",
        "write code",
        "write an sql",
        "create a script",
        "create a function",
        "define a function",
        "generate code",
        "return code",
    )
    return any(signal in lower for signal in signals) or bool(
        re.search(
            r"\b(write|create|implement|define|generate)\b.{0,60}"
            r"\b(function|method|class|program|script|query|algorithm)\b",
            lower,
        )
    )


def _looks_like_code_debugging(text: str, lower: str) -> bool:
    runtime_errors = (
        "traceback",
        "syntaxerror",
        "typeerror",
        "indexerror",
        "keyerror",
        "valueerror",
        "attributeerror",
        "nameerror",
        "nullpointerexception",
    )
    if any(error in lower for error in runtime_errors):
        return True
    if "syntax issue" in lower or "syntax error" in lower:
        return True
    signals = (
        "debug",
        "fix the bug",
        "find and fix",
        "has a bug",
        "contains a bug",
        "correct the code",
        "corrected implementation",
        "incorrect output",
        "wrong output",
        "doesn't work",
        "does not work",
        "why does this code",
    )
    if any(signal in lower for signal in signals):
        return True
    return "```" in text and any(
        word in lower for word in ("error", "bug", "wrong", "fix", "issue", "correct")
    )


def _looks_like_summarization(lower: str, text: str) -> bool:
    if any(
        word in lower
        for word in ("summarize", "summarise", "summary", "tl;dr", "tldr", "condense")
    ):
        return True
    return len(text.split()) > 140 and "?" not in text[:120]


def _looks_like_ner(lower: str) -> bool:
    signals = (
        "extract named entities",
        "named entities",
        "extract entities",
        "identify entities",
        "label the entities",
        "entities and their types",
        "entities with their types",
        "find all people",
        "find all persons",
        "find all organizations",
        "find all locations",
        "people, organizations",
        "persons, organizations",
        "person, org",
        "person, organization",
    )
    return any(signal in lower for signal in signals)


def _looks_like_sentiment(lower: str) -> bool:
    signals = (
        "sentiment",
        "positive, negative, or neutral",
        "positive negative neutral",
        "positive or negative",
        "opinion polarity",
        "classify the review",
        "classify this review",
        "is this review positive",
    )
    return any(signal in lower for signal in signals)


def _looks_like_logic(lower: str) -> bool:
    signals = (
        "logic puzzle",
        "exactly one",
        "exactly two",
        "at least one",
        "who is lying",
        "truth teller",
        "knights and knaves",
        "if and only if",
        "who owns the",
        "which person",
        "logical deduction",
        "can we conclude",
        "does it follow",
        "different pet",
        "different job",
        "different color",
        "different day",
        "order from left to right",
        "order from first to last",
    )
    if any(signal in lower for signal in signals):
        return True
    if re.search(r"\bif\b[^.;]{3,100}(?:\bthen\b|,)[^.;]+[.;]", lower) and re.search(
        r"\b(?:can|does|is|are|must|could)\b[^?]*\?", lower
    ):
        return True
    if re.search(r"\ball\b[^.]+\.[^.]+\b(?:is|are)\b[^.]+\.", lower) and re.search(
        r"\b(?:does|is|are|can)\b[^?]*\?", lower
    ):
        return True
    return bool(
        re.search(r"\beach\b.{0,100}\bdifferent\b", lower, flags=re.S)
        and re.search(r"\b(does not|doesn't|cannot|must|owns?|has)\b", lower)
    )


def _looks_like_math(lower: str) -> bool:
    math_words = (
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "how many",
        "how much",
        "how far",
        "sum",
        "product",
        "quotient",
        "squared",
        "square root",
        "cube root",
        "average",
        "ratio",
        "probability",
        "percent",
        "percentage",
        "discount",
        "interest",
        "increase",
        "decrease",
        "remain",
        "projection",
        "per year",
    )
    has_number = bool(
        re.search(r"\d", lower)
        or re.search(
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|hundred|thousand)\b",
            lower,
        )
    )
    has_operator = bool(re.search(r"[\d)\s][+\-*/^][\d(\s]", lower))
    numeric_difference = bool(
        re.search(
            r"\bdifference\s+between\s+-?\d+(?:\.\d+)?\s+and\s+-?\d+(?:\.\d+)?",
            lower,
        )
    )
    return has_number and (
        has_operator or numeric_difference or any(word in lower for word in math_words)
    )
