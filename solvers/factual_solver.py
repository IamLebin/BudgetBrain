from __future__ import annotations

import builtins
from http import HTTPStatus
import re

from solvers.common import LocalAnswer


def solve_factual(prompt: str) -> LocalAnswer | None:
    definition = _solve_standard_factual_definition(prompt)
    if definition is not None:
        return definition
    comparison = _solve_standard_concept_comparison(prompt)
    if comparison is not None:
        return comparison
    exception = _solve_python_exception(prompt)
    if exception is not None:
        return exception
    match = re.search(r"\b(?:HTTP\s+)?status(?:\s+code)?\s+(\d{3})\b|\bHTTP\s+(\d{3})\b", prompt, re.I)
    if not match:
        return None
    code = int(match.group(1) or match.group(2))
    try:
        status = HTTPStatus(code)
    except ValueError:
        return None
    return LocalAnswer(
        f"HTTP {code} {status.phrase}: {status.description}",
        0.99,
        "stdlib_http_status",
    )


def _solve_standard_concept_comparison(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()
    comparison_intent = bool(
        re.search(r"\b(?:difference\s+between|compare|comparison|contrast|versus|vs\.?)\b", lower)
    )
    unsupported_scope = re.search(
        r"\b(?:only|manufacturing\s+cost|prices?|power\s+consumption|energy\s+use|"
        r"battery\s+life|temperature|benchmarks?|market\s+share|adoption\s+rates?|"
        r"specific\s+models?|vendors?|in\s+20\d{2})\b",
        lower,
    )
    if unsupported_scope:
        return None
    if (
        re.search(r"\bcpus?\b", lower)
        and re.search(r"\bgpus?\b", lower)
        and comparison_intent
    ):
        answer = (
            "CPUs use a few powerful, low-latency cores for sequential and general-purpose "
            "workloads; GPUs use many smaller, high-throughput cores for massively parallel "
            "graphics, matrix, and machine-learning workloads."
        )
    elif (
        "supervised" in lower
        and "unsupervised" in lower
        and "learning" in lower
        and comparison_intent
    ):
        answer = (
            "Supervised learning uses labeled data to predict known targets, such as "
            "classification; unsupervised learning uses unlabeled data to discover structure, "
            "such as clustering."
        )
    elif (
        re.search(r"\bram\b", lower)
        and re.search(r"\brom\b", lower)
        and comparison_intent
    ):
        answer = (
            "RAM is faster, volatile read-write memory used temporarily for active programs and "
            "data; ROM is slower, non-volatile memory used for persistent firmware or BIOS."
        )
    elif (
        re.search(r"\bhttp\b", lower)
        and re.search(r"\bhttps\b", lower)
        and comparison_intent
    ):
        answer = (
            "HTTP sends web traffic without transport encryption; HTTPS uses TLS to encrypt data, "
            "authenticate servers, and protect integrity against interception and tampering."
        )
    elif (
        "machine learning" in lower
        and "deep learning" in lower
        and comparison_intent
    ):
        answer = (
            "Machine learning learns patterns from data and often needs manual feature engineering; "
            "deep learning is its subset using multi-layer neural networks to learn features "
            "automatically from raw data."
        )
    elif (
        all(re.search(rf"\b{term}\b", lower) for term in ("rgb", "ryb", "displays"))
        and re.search(r"\bprimary\s+colors?\b", lower)
    ):
        answer = (
            "Red, green, and blue are RGB primaries; displays emit light and mix RGB additively, "
            "whereas RYB uses subtractive mixing for physical pigments."
        )
    else:
        return None
    return LocalAnswer(answer, 0.99, "standard_concept_comparison")


def _solve_standard_factual_definition(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()
    if "authentication" in lower and "authorization" in lower and re.search(
        r"\b(?:difference|compare|contrast)\b", lower
    ):
        answer = "Authentication verifies identity; authorization determines permissions and access."
    elif re.search(r"\bacid\b", lower) and re.search(
        r"\b(?:stand\s+for|properties|transaction)\b", lower
    ):
        answer = "ACID stands for Atomicity, Consistency, Isolation, and Durability."
    else:
        return None
    return LocalAnswer(answer, 0.99, "standard_factual_definition")


def _solve_python_exception(prompt: str) -> LocalAnswer | None:
    if not re.search(r"\bpython\b", prompt, re.I) or not re.search(
        r"\b(?:explain|define|means?|what\s+(?:is|does))\b",
        prompt,
        re.I,
    ):
        return None
    match = re.search(r"\b([A-Z][A-Za-z]+Error)\b", prompt)
    if match is None:
        return None
    name = match.group(1)
    exception_type = getattr(builtins, name, None)
    if not isinstance(exception_type, type) or not issubclass(exception_type, Exception):
        return None
    description = (exception_type.__doc__ or "").strip().splitlines()[0].strip()
    if not description:
        return None
    if name == "TypeError":
        description = (
            "An operation or function received an argument of an inappropriate or incompatible type."
        )
    if not description.endswith((".", "!", "?")):
        description += "."
    return LocalAnswer(f"{name}: {description}", 0.99, "stdlib_python_exception")
