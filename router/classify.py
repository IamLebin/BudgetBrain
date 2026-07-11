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
    if _looks_like_ner(lower):
        return Classification("ner", 0.94)
    if _looks_like_sentiment(lower):
        return Classification("sentiment", 0.93)
    if _looks_like_summarization(lower, text):
        return Classification("summarization", 0.94)
    if _looks_like_logic(lower):
        return Classification("logic", 0.89)
    if _looks_like_math(lower):
        return Classification("math", 0.9)
    return Classification("factual_qa", 0.55)


def _looks_like_code_generation(lower: str) -> bool:
    if re.match(
        r"\s*(?:what\s+(?:is|are)|how\s+(?:do|does|would)|explain\s+how|describe\s+how)\b",
        lower,
    ) and not re.search(
        r"\b(?:provide|return|output|give|show|include)\b.{0,35}\b(?:code|implementation|function)\b",
        lower,
    ):
        return False
    signals = (
        "write a function",
        "write a python function",
        "write a program",
        "write code",
        "write an sql",
        "write a sql",
        "create a script",
        "create a function",
        "define a function",
        "generate code",
        "return code",
        "code that",
        "give me code",
        "give me a function",
        "provide code",
        "provide a function",
        "provide an sql",
        "provide sql",
        "design a function",
        "design a class",
    )
    return any(signal in lower for signal in signals) or bool(
        re.search(
            r"\b(write|create|implement|define|generate|design|build|develop)\b.{0,60}"
            r"\b(function|method|class|program|script|query|algorithm|routine|procedure)\b",
            lower,
        )
    )


def _looks_like_code_debugging(text: str, lower: str) -> bool:
    code_context = "```" in text or bool(
        re.search(
            r"\b(?:code|function|method|class|program|script|algorithm|query|"
            r"python|javascript|typescript|java|sql|c\+\+|rust|traceback)\b",
            lower,
        )
    )
    repair_intent = bool(
        re.search(
            r"\b(?:debug|fix|repair|correct|wrong|broken|fails?|failure|"
            r"why\s+(?:does|is|am)|what(?:'s|\s+is)\s+wrong)\b",
            lower,
        )
    )
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
        "runtimeerror",
        "zerodivisionerror",
        "overflowerror",
        "filenotfounderror",
        "importerror",
        "recursionerror",
    )
    if any(error in lower for error in runtime_errors) and (code_context and repair_intent):
        return True
    if ("syntax issue" in lower or "syntax error" in lower) and repair_intent:
        return True
    strong_signals = (
        "debug",
        "fix the bug",
        "has a bug",
        "contains a bug",
        "correct the code",
        "corrected implementation",
        "why does this code",
        "spot the bug",
        "spot the defect",
        "locate the bug",
        "repair the code",
    )
    if any(signal in lower for signal in strong_signals):
        return True

    contextual_signals = (
        "find and fix",
        "incorrect output",
        "wrong output",
        "doesn't work",
        "does not work",
        "what's wrong",
        "what is wrong",
        "unexpected behavior",
        "unexpected behaviour",
        "unexpected result",
        "fix this",
        "repair",
        "behaves unexpectedly",
        "produces wrong",
        "gives wrong",
        "returns wrong",
        "has a defect",
        "contains a defect",
        "malfunctions",
    )
    if code_context and any(signal in lower for signal in contextual_signals):
        return True
    return "```" in text and any(
        word in lower for word in ("error", "bug", "wrong", "fix", "issue", "correct", "broken", "fails")
    )


def _looks_like_summarization(lower: str, text: str) -> bool:
    if any(word in lower for word in ("summarize", "summarise", "tl;dr", "tldr", "condense")):
        return True
    source_object_context = bool(
        re.search(
            r"\b(?:following|above|below|provided|this)\s+"
            r"(?:text|passage|article|report|document|email|update|story)\b",
            lower,
        )
    )
    if _is_concept_question(lower) and not source_object_context:
        return False
    source_context = bool(
        source_object_context
        or re.search(
            r"\b(?:summary|overview|key points?|main points?|main idea|gist|takeaways?)"
            r"\s+(?:of|for|from)\b",
            lower,
        )
        or (":" in text and len(text.split(":", maxsplit=1)[-1].split()) >= 8)
    )
    if source_context and any(
        phrase in lower
        for phrase in (
            "summary", "overview", "key point", "main point", "main idea", "gist",
            "takeaway", "in a nutshell",
        )
    ):
        return True
    if re.search(
        r"\b(?:in|into)\s+(?:one|two|three|four|five|[1-5])\s+(?:bullet\s+points?|sentences?)\b",
        lower,
    ):
        return True
    return False


def _looks_like_ner(lower: str) -> bool:
    if _is_concept_question(lower) and not re.search(
        r"\b(?:extract|identify|label|list|tag|recognize|find)\b", lower
    ):
        return False
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
        "list all entities",
        "tag the entities",
        "recognize entities",
        "what are the entities",
        "extract all named",
        "identify all named",
        "entities and types",
    )
    if any(signal in lower for signal in signals):
        return True
    return bool(
        re.search(r"\b(?:extract|identify|list|label|tag|find)\b", lower)
        and re.search(
            r"\b(?:people|persons?|names?|organizations?|organisations?|orgs?|companies|"
            r"locations?|places?|cities|dates?|entities)\b",
            lower,
        )
    )


def _looks_like_sentiment(lower: str) -> bool:
    if _is_concept_question(lower) and not re.search(
        r"\b(?:classify|analyze|analyse|determine|label|identify)\b", lower
    ):
        return False
    signals = (
        "sentiment",
        "positive, negative, or neutral",
        "positive, negative, neutral",
        "positive negative neutral",
        "positive or negative",
        "opinion polarity",
        "classify the review",
        "classify this review",
        "is this review positive",
        "tone of",
        "what is the tone",
        "favorable or unfavorable",
        "favourable or unfavourable",
        "emotion conveyed",
        "attitude expressed",
    )
    if any(signal in lower for signal in signals):
        return True
    return bool(
        re.search(
            r"\bclassify\b.{0,40}\b(?:as|into)\b.{0,30}"
            r"\b(?:positive|negative|neutral|favorable|unfavorable|favourable|unfavourable)\b",
            lower,
        )
    )


def _looks_like_logic(lower: str) -> bool:
    if _is_concept_question(lower) and not re.search(
        r"\b(?:deduce|conclude|solve|infer|must be true|does it follow)\b", lower
    ):
        return False
    signals = (
        "logic puzzle",
        "logical puzzle",
        "reasoning puzzle",
        "exactly one",
        "exactly two",
        "only one statement",
        "at least one",
        "who is lying",
        "truth teller",
        "telling the truth",
        "knights and knaves",
        "if and only if",
        "who owns the",
        "which person",
        "logical deduction",
        "logically deduce",
        "can we conclude",
        "what can you conclude",
        "what can we conclude",
        "what must be true",
        "does it follow",
        "can you deduce",
        "different pet",
        "different job",
        "different color",
        "different colour",
        "different day",
        "order from left to right",
        "order from first to last",
        "given these constraints",
        "given the constraints",
        "which must be true",
        "who can be",
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
        "what is the value",
        "find the value",
        "determine the result",
        "total cost",
        "total price",
        "sale price",
        "final price",
        "net pay",
    )
    if "=" in lower and re.search(
        r"\b(?:solve|find)\s+(?:the\s+value\s+of\s+|for\s+)?[a-z]\b",
        lower,
    ):
        return True
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


def _is_concept_question(lower: str) -> bool:
    return bool(
        re.match(
            r"\s*(?:what\s+(?:is|are|does)|define|explain|describe|how\s+does)\b",
            lower,
        )
    )
