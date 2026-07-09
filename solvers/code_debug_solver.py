from __future__ import annotations

import ast
import re

from solvers.common import LocalAnswer


def solve_code_debug(prompt: str) -> LocalAnswer | None:
    code = _extract_code(prompt)
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}" if exc.lineno else "the code"
        detail = exc.msg or "syntax error"
        return LocalAnswer(f"Syntax error at {location}: {detail}", 0.9, "ast_parse")
    return None


def _extract_code(prompt: str) -> str | None:
    fenced = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", prompt, flags=re.S)
    if fenced:
        return fenced.group(1)
    if re.search(r"\b(def|class|for|while|if|return|import)\b", prompt):
        return prompt
    return None
