from __future__ import annotations

import ast
from fractions import Fraction
import math
import operator
import re

from solvers.common import LocalAnswer


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def solve_math(prompt: str) -> LocalAnswer | None:
    cleaned = _normalize_math_text(prompt)

    percent = _solve_percent(cleaned)
    if percent is not None:
        return percent

    word_problem = _solve_simple_word_arithmetic(cleaned)
    if word_problem is not None:
        return word_problem

    expression = _extract_expression(cleaned)
    if expression is None:
        return None
    try:
        value = _safe_eval(expression)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return LocalAnswer(_format_number(value), 0.95, "safe_eval")


def _normalize_math_text(text: str) -> str:
    return (
        text.replace("×", "*")
        .replace("÷", "/")
        .replace("^", "**")
        .replace(",", "")
        .strip()
    )


def _solve_percent(text: str) -> LocalAnswer | None:
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    pct = Fraction(match.group(1)) / 100
    base = Fraction(match.group(2))
    return LocalAnswer(_format_number(pct * base), 0.95, "percent")


def _solve_simple_word_arithmetic(text: str) -> LocalAnswer | None:
    number = r"(-?\d+(?:\.\d+)?)"
    patterns = [
        (fr"\bsum of {number} and {number}\b", lambda a, b: a + b),
        (fr"\badd {number} and {number}\b", lambda a, b: a + b),
        (fr"\b{number}\s+plus\s+{number}\b", lambda a, b: a + b),
        (fr"\bdifference between {number} and {number}\b", lambda a, b: a - b),
        (fr"\b{number}\s+minus\s+{number}\b", lambda a, b: a - b),
        (fr"\bproduct of {number} and {number}\b", lambda a, b: a * b),
        (fr"\bmultiply {number} by {number}\b", lambda a, b: a * b),
        (fr"\b{number}\s+times\s+{number}\b", lambda a, b: a * b),
        (fr"\bquotient of {number} and {number}\b", lambda a, b: a / b),
        (fr"\bdivide {number} by {number}\b", lambda a, b: a / b),
    ]
    for pattern, fn in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            a = Fraction(match.group(1))
            b = Fraction(match.group(2))
            return LocalAnswer(_format_number(fn(a, b)), 0.92, "word_arithmetic")
    return None


def _extract_expression(text: str) -> str | None:
    code_match = re.search(r"`([^`]+)`", text)
    if code_match:
        candidate = code_match.group(1)
    else:
        matches = re.findall(r"[-+*/().\d\s*]+", text)
        matches = [m.strip() for m in matches if re.search(r"\d", m) and re.search(r"[+\-*/]", m)]
        if not matches:
            return None
        candidate = max(matches, key=len)
    candidate = candidate.strip().rstrip("=?")
    if not re.fullmatch(r"[-+*/().\d\s]+", candidate):
        return None
    return candidate


def _safe_eval(expression: str) -> int | float | Fraction:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float | Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
            raise ValueError("power too large")
        return OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _format_number(value: int | float | Fraction) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        decimal = value.numerator / value.denominator
        if math.isclose(decimal, round(decimal), rel_tol=0, abs_tol=1e-12):
            return str(round(decimal))
        return f"{decimal:.10g}"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)
