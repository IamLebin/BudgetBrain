from __future__ import annotations

import builtins
from http import HTTPStatus
import re

from solvers.common import LocalAnswer


def solve_factual(prompt: str) -> LocalAnswer | None:
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
