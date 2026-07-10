from __future__ import annotations

import ast
import re

from solvers.common import LocalAnswer


def solve_code_debug(prompt: str) -> LocalAnswer | None:
    code = _extract_code(prompt)
    if not code:
        return None


    extremum = _repair_extremum_function(prompt, code)
    if extremum is not None:
        return LocalAnswer(extremum, 0.98, "extremum_repair")

    try:
        ast.parse(code)
    except SyntaxError:
        repaired = _repair_missing_colons(code)
        if repaired != code:
            try:
                ast.parse(repaired)
            except SyntaxError:
                return None
            return LocalAnswer(repaired, 0.97, "missing_colon_repair")
        return None
    return None


def _extract_code(prompt: str) -> str | None:
    fenced = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", prompt, flags=re.S)
    if fenced:
        return fenced.group(1).strip()

    inline_function = re.search(
        r"\b(def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:?.*?)"
        r"(?=\.\s*(?:Find|Fix|Identify|Explain|What|Why|Correct)\b|$)",
        prompt,
        flags=re.IGNORECASE | re.S,
    )
    if inline_function:
        return inline_function.group(1).strip().rstrip(".")
    return None


def _repair_extremum_function(prompt: str, code: str) -> str | None:
    intent_match = re.search(
        r"\b(?:return|find|compute|get)\s+(?:the\s+)?(max(?:imum)?|largest|min(?:imum)?|smallest)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    function_match = re.search(
        r"\bdef\s+([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*:",
        code,
    )
    if not intent_match or not function_match:
        return None

    function_name, parameter = function_match.groups()
    target = intent_match.group(1).lower()
    builtin = "min" if target in {"min", "minimum", "smallest"} else "max"

    # Only claim a free repair when the implementation clearly returns one fixed element.
    fixed_element = re.search(
        fr"\breturn\s+{re.escape(parameter)}\s*\[\s*-?\d+\s*\]",
        code,
    )
    if not fixed_element:
        return None
    return f"def {function_name}({parameter}):\n    return {builtin}({parameter})"


def _repair_missing_colons(code: str) -> str:
    repaired_lines: list[str] = []
    header = re.compile(r"^\s*(?:async\s+def|def|class|if|elif|else|for|while|try|except|finally|with)\b")
    for line in code.splitlines():
        stripped = line.rstrip()
        if header.search(stripped) and not stripped.endswith(":"):
            stripped += ":"
        repaired_lines.append(stripped)
    return "\n".join(repaired_lines)
