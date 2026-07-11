from __future__ import annotations

from http import HTTPStatus
import re

from solvers.common import LocalAnswer


def solve_factual(prompt: str) -> LocalAnswer | None:
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
