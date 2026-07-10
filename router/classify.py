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

    if _looks_like_code_generation(lower):
        return Classification("code_generation", 0.9)
    if _looks_like_code_debugging(text, lower):
        return Classification("code_debugging", 0.9)
    if _looks_like_summarization(lower, text):
        return Classification("summarization", 0.9)
    if _looks_like_ner(lower):
        return Classification("ner", 0.88)
    if _looks_like_sentiment(lower):
        return Classification("sentiment", 0.86)
    if _looks_like_logic(lower):
        return Classification("logic", 0.78)
    if _looks_like_math(lower):
        return Classification("math", 0.85)
    return Classification("factual_qa", 0.55)


def _looks_like_code_generation(lower: str) -> bool:
    signals = (
        "write a function",
        "write a python function",
        "write code",
        "implement ",
        "create a script",
        "generate code",
        "return code",
    )
    return any(signal in lower for signal in signals) or bool(
        re.search(r"\b(write|create|implement)\b.{0,40}\bfunction\b", lower)
    )


def _looks_like_code_debugging(text: str, lower: str) -> bool:
    if "traceback" in lower or "syntaxerror" in lower or "typeerror" in lower:
        return True
    if "syntax issue" in lower or "syntax error" in lower:
        return True
    if "debug" in lower or "fix the bug" in lower or "why does this code" in lower:
        return True
    return "```" in text and any(word in lower for word in ("error", "bug", "wrong", "fix"))


def _looks_like_summarization(lower: str, text: str) -> bool:
    if any(word in lower for word in ("summarize", "summary", "tl;dr", "tldr")):
        return True
    return len(text.split()) > 140 and "?" not in text[:120]


def _looks_like_ner(lower: str) -> bool:
    signals = (
        "extract named entities",
        "named entities",
        "extract entities",
        "identify entities",
        "find all people",
        "find all organizations",
        "find all locations",
        "person, organization",
    )
    return any(signal in lower for signal in signals)


def _looks_like_sentiment(lower: str) -> bool:
    signals = (
        "sentiment",
        "positive, negative, or neutral",
        "positive negative neutral",
        "classify the review",
        "classify this review",
        "is this review positive",
    )
    return any(signal in lower for signal in signals)


def _looks_like_logic(lower: str) -> bool:
    signals = (
        "logic puzzle",
        "exactly one",
        "who is lying",
        "truth teller",
        "knights and knaves",
        "if and only if",
    )
    if any(signal in lower for signal in signals):
        return True
        
    # Ordering / Ranking puzzles
    if re.search(r"\b(older|taller|faster|shorter|youngest|oldest|tallest)\b", lower):
        return True
        
    # Deduction puzzles (if P then Q)
    if re.search(r"\bif\b.+\bthen\b", lower) or (re.search(r"\bif\b", lower) and "," in lower):
        return True
        
    return False


def _looks_like_math(lower: str) -> bool:
    math_words = (
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "what is",
        "sum",
        "product",
        "difference",
        "percent",
        "percentage",
    )
    has_number = bool(re.search(r"\d", lower))
    has_operator = bool(re.search(r"[\d)\s][+\-*/^][\d(\s]", lower))
    return has_number and (has_operator or any(word in lower for word in math_words))
