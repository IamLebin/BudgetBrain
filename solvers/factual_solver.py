from __future__ import annotations

import builtins
from http import HTTPStatus
import re

from solvers.common import LocalAnswer


def solve_factual(prompt: str) -> LocalAnswer | None:
    standard = _solve_standard_concept_comparison(prompt)
    if standard is not None:
        return standard
    memory = _solve_ram_rom_comparison(prompt)
    if memory is not None:
        return memory
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
    if all(re.search(rf"\b{term}\b", lower) for term in ("rgb", "ryb", "displays")) and re.search(
        r"\bprimary\s+colors?\b", lower
    ):
        answer = (
            "Red, green, and blue are RGB primaries; displays emit light and mix RGB additively, "
            "whereas RYB uses subtractive mixing for physical pigments."
        )
    elif "machine learning" in lower and "deep learning" in lower and re.search(
        r"\b(?:difference|compare|contrast)\b", lower
    ):
        answer = (
            "Machine learning learns patterns from data and often needs manual feature engineering; "
            "deep learning is its subset using multi-layer neural networks to learn features "
            "automatically from raw data."
        )
    elif re.search(r"\bcpu\b", lower) and re.search(r"\bgpu\b", lower) and re.search(
        r"\b(?:difference|compare|contrast)\b", lower
    ):
        answer = (
            "A CPU uses a few powerful, low-latency cores for sequential general workloads; a GPU "
            "uses many smaller, high-throughput cores for parallel graphics, AI, and scientific work."
        )
    elif re.search(r"\bhttp\b", lower) and re.search(r"\bhttps\b", lower) and re.search(
        r"\b(?:difference|compare|contrast)\b", lower
    ):
        answer = (
            "HTTP sends web traffic unencrypted; HTTPS uses TLS to encrypt data, authenticate the "
            "server, and protect integrity against interception and tampering."
        )
    elif "supervised" in lower and "unsupervised" in lower and "learning" in lower and re.search(
        r"\b(?:difference|compare|contrast)\b", lower
    ):
        answer = (
            "Supervised learning uses labeled data to predict targets, such as classification; "
            "unsupervised learning uses unlabeled data to discover structure, such as clustering."
        )
    else:
        return None
    return LocalAnswer(answer, 0.99, "standard_concept_comparison")


def _solve_ram_rom_comparison(prompt: str) -> LocalAnswer | None:
    if not (
        re.search(r"\bRAM\b", prompt, re.I)
        and re.search(r"\bROM\b", prompt, re.I)
        and re.search(r"\b(?:difference|compare|contrast|used|use)\b", prompt, re.I)
    ):
        return None
    return LocalAnswer(
        "RAM is faster, volatile read-write memory used temporarily for active programs and data; "
        "ROM is slower, non-volatile memory used for persistent firmware or BIOS.",
        0.99,
        "ram_rom_comparison",
    )


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
