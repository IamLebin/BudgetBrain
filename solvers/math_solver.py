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

    inventory = _solve_inventory_changes(cleaned)
    if inventory is not None:
        return inventory

    percent = _solve_percent(cleaned)
    if percent is not None:
        return percent

    projection = _solve_compound_projection(cleaned)
    if projection is not None:
        return projection

    percent_change = _solve_percent_change(cleaned)
    if percent_change is not None:
        return percent_change

    average = _solve_average(cleaned)
    if average is not None:
        return average

    equation = _solve_linear_equation(cleaned)
    if equation is not None:
        return equation

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
    normalized = (
        text.replace("×", "*")
        .replace("÷", "/")
        .replace("^", "**")
        .replace("−", "-")
        .strip()
    )
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)


def _solve_inventory_changes(text: str) -> LocalAnswer | None:
    number = r"(-?\d+(?:\.\d+)?)"
    initial_match = re.search(
        fr"\b(?:has|starts? with|begins? with|initially has)\s+{number}\s+"
        r"(?:items?|units?|products?|tickets?|books?|dollars?|people|employees)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not initial_match or not re.search(r"\b(remain|left|balance)\b", text, re.I):
        return None

    initial = Fraction(initial_match.group(1))
    percent_match = re.search(
        fr"\b(?:sells?|sold|uses?|used|loses?|lost|spends?|spent|removes?|removed)\s+"
        fr"{number}\s*(?:%|percent\b)",
        text,
        flags=re.IGNORECASE,
    )
    if not percent_match:
        return None

    remaining = initial * (1 - Fraction(percent_match.group(1)) / 100)
    tail = text[percent_match.end() :]
    if re.search(r"-?\d+(?:\.\d+)?\s*(?:%|percent\b)", tail, re.I):
        return None
    subtract_match = re.search(
        fr"\b(?:and|then)\s+(?:(?:it|they|the store)\s+)?"
        fr"(?:(?:sells?|sold|uses?|used|loses?|lost|spends?|spent|removes?|removed)\s+)?"
        fr"(?:another|an additional)?\s*{number}(?:\s+more)?\b",
        tail,
        flags=re.IGNORECASE,
    )
    if subtract_match:
        remaining -= Fraction(subtract_match.group(1))

    add_match = re.search(
        fr"\b(?:and|then)\s+(?:(?:it|they|the store)\s+)?"
        fr"(?:receives?|received|adds?|added|buys?|bought|gains?|gained)\s+"
        fr"(?:another|an additional)?\s*{number}\b",
        tail,
        flags=re.IGNORECASE,
    )
    if add_match:
        remaining += Fraction(add_match.group(1))

    return LocalAnswer(_format_number(remaining), 0.98, "inventory_percent_changes")


def _solve_percent(text: str) -> LocalAnswer | None:
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    if re.search(
        r"\b(?:then|plus|add|minus|subtract|increase|decrease|multiply|divide)\b",
        text[match.end() :],
        re.I,
    ):
        return None
    pct = Fraction(match.group(1)) / 100
    base = Fraction(match.group(2))
    return LocalAnswer(_format_number(pct * base), 0.95, "percent")


def _solve_percent_change(text: str) -> LocalAnswer | None:
    if len(re.findall(r"(?:%|\bpercent\b)", text, flags=re.IGNORECASE)) != 1:
        return None
    number = r"-?\d+(?:\.\d+)?"
    change = re.search(
        fr"\b(increase[sd]?|grow(?:s|n)?|rise[sd]?|decrease[sd]?|drop(?:s|ped)?|"
        fr"fall(?:s|en)?|reduc(?:e|ed))\s+by\s+(?P<pct>{number})\s*(?:%|percent)",
        text,
        flags=re.IGNORECASE,
    )
    if change:
        base = _last_non_year_number(text[: change.start()])
        if base is None:
            return None
        pct = Fraction(change.group("pct"))
        direction = change.group(1).lower()
        increasing = direction.startswith(("increase", "grow", "rise"))
        multiplier = 1 + pct / 100 if increasing else 1 - pct / 100
        return LocalAnswer(_format_number(base * multiplier), 0.95, "percent_change")

    adjustment = re.search(
        fr"(?P<pct>{number})\s*(?:%|percent)\s+(discount|off|tax|tip|markup)\b",
        text,
        flags=re.IGNORECASE,
    )
    if adjustment:
        base = _last_non_year_number(text[: adjustment.start()])
        if base is None:
            return None
        pct = Fraction(adjustment.group("pct"))
        kind = adjustment.group(2).lower()
        multiplier = 1 - pct / 100 if kind in {"discount", "off"} else 1 + pct / 100
        return LocalAnswer(_format_number(base * multiplier), 0.95, "percent_change")
    return None


def _last_non_year_number(text: str) -> Fraction | None:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    for match in reversed(matches):
        value = Fraction(match.group(0))
        nearby = text[max(0, match.start() - 8) : match.start()].lower()
        if value.denominator == 1 and 1900 <= value <= 2100 and re.search(r"\b(?:in|year)\s*$", nearby):
            continue
        return value
    return None


def _solve_compound_projection(text: str) -> LocalAnswer | None:
    match = re.search(
        r"(?:starts?|begins?|initial(?:ly)?|current(?:ly)?)\D{0,25}"
        r"(-?\d+(?:\.\d+)?)\D{0,60}"
        r"(?:grow(?:s|th)?|increase[sd]?|appreciate[sd]?)\s+(?:by\s+)?"
        r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:per|each)\s+year\D{0,40}"
        r"(?:for|after|over)\s+(\d+)\s+years?",
        text,
        flags=re.IGNORECASE | re.S,
    )
    if not match:
        return None
    initial = Fraction(match.group(1))
    rate = Fraction(match.group(2)) / 100
    years = int(match.group(3))
    if years > 100:
        return None
    value = initial * (1 + rate) ** years
    return LocalAnswer(_format_number(value), 0.95, "compound_projection")


def _solve_average(text: str) -> LocalAnswer | None:
    match = re.search(
        r"\b(?:average|mean)\s+(?:of\s+)?([^.?\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    values = [Fraction(value) for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(values) < 2 or not re.search(r",|\band\b", match.group(1), re.I):
        return None
    return LocalAnswer(_format_number(sum(values) / len(values)), 0.97, "average")


def _solve_linear_equation(text: str) -> LocalAnswer | None:
    match = re.search(
        r"(?<![A-Za-z0-9])([+-]?\s*\d*(?:\.\d+)?)\s*[*]?\s*x\s*"
        r"([+-]\s*\d+(?:\.\d+)?)?\s*=\s*([+-]?\s*\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    coefficient_text = match.group(1).replace(" ", "")
    if coefficient_text in {"", "+"}:
        coefficient = Fraction(1)
    elif coefficient_text == "-":
        coefficient = Fraction(-1)
    else:
        coefficient = Fraction(coefficient_text)
    if coefficient == 0:
        return None
    constant = Fraction(match.group(2).replace(" ", "")) if match.group(2) else Fraction(0)
    right = Fraction(match.group(3).replace(" ", ""))
    value = (right - constant) / coefficient
    return LocalAnswer(_format_number(value), 0.98, "linear_equation")


def _solve_simple_word_arithmetic(text: str) -> LocalAnswer | None:
    number = r"(-?\d+(?:\.\d+)?)"
    patterns = [
        (fr"\bsum of {number} and {number}\b", lambda a, b: a + b),
        (fr"\badd {number} and {number}\b", lambda a, b: a + b),
        (fr"\b{number}\s+plus\s+{number}\b", lambda a, b: a + b),
        (fr"\bdifference between {number} and {number}\b", lambda a, b: a - b),
        (fr"\bsubtract {number} from {number}\b", lambda a, b: b - a),
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
            if re.search(
                r"\b(?:then|plus|add|minus|subtract|multiply|divide)\b",
                text[match.end() :],
                re.I,
            ):
                return None
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
