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

    sequential_percent = _solve_sequential_percent_changes(cleaned)
    if sequential_percent is not None:
        return sequential_percent

    average_speed = _solve_average_speed(cleaned)
    if average_speed is not None:
        return average_speed

    ratio = _solve_ratio_share(cleaned)
    if ratio is not None:
        return ratio

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


def _solve_sequential_percent_changes(text: str) -> LocalAnswer | None:
    number = r"-?\d+(?:\.\d+)?"
    initial_patterns = (
        fr"\b(?:starts?|begins?)\s+(?:at|with)\s+\$?(?P<value>{number})\b",
        fr"\b(?:initial|original|starting)\s+(?:price|value|amount|cost)?\s*"
        fr"(?:is|was|of|:)\s*\$?(?P<value>{number})\b",
    )
    initial_match = next(
        (
            match
            for pattern in initial_patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE)) is not None
        ),
        None,
    )
    if initial_match is None:
        return None

    tail = text[initial_match.end() :]
    change_pattern = re.compile(
        fr"\b(?P<verb>increase[sd]?|rise[sd]?|grow(?:s|n)?|"
        fr"decrease[sd]?|drop(?:s|ped)?|fall(?:s|en)?|reduc(?:e[sd]?|ed))\s+"
        fr"(?:by\s+)?(?P<pct>{number})(?:\s*%|\s+percent\b)",
        flags=re.IGNORECASE,
    )
    changes = list(change_pattern.finditer(tail))
    percent_count = len(re.findall(r"(?:%|\bpercent\b)", tail, flags=re.IGNORECASE))
    if len(changes) < 2 or len(changes) != percent_count:
        return None
    percent_spans = [change.span("pct") for change in changes]
    for number_match in re.finditer(number, tail):
        if not any(
            number_match.start() >= start and number_match.end() <= end
            for start, end in percent_spans
        ):
            return None
    if not re.search(r"\b(?:then|after(?:wards)?|followed by|and)\b", tail, re.I):
        return None

    value = Fraction(initial_match.group("value"))
    for change in changes:
        pct = Fraction(change.group("pct"))
        verb = change.group("verb").lower()
        increasing = verb.startswith(("increase", "rise", "grow"))
        if pct < 0 or (not increasing and pct > 100):
            return None
        value *= 1 + pct / 100 if increasing else 1 - pct / 100
    return LocalAnswer(_format_number(value), 0.99, "sequential_percent_changes")


def _solve_average_speed(text: str) -> LocalAnswer | None:
    if not re.search(r"\baverage\s+speed\b", text, re.I):
        return None
    if re.search(r"\b(?:rests?|stops?|waits?|layovers?|breaks?)\b", text, re.I):
        return None
    number = r"\d+(?:\.\d+)?"
    leg_pattern = re.compile(
        fr"\b(?:(?:travels?|covers?|goes?)\s+)?(?P<distance>{number})\s*"
        r"(?P<distance_unit>km|kilometers?|kilometres?|mi|miles?)\s+"
        r"(?:at|with\s+(?:a\s+)?speed\s+of)\s+"
        fr"(?P<speed>{number})\s*"
        r"(?P<speed_unit>km/h|kph|kilometers?\s+per\s+hour|"
        r"kilometres?\s+per\s+hour|mph|miles?\s+per\s+hour)\b",
        flags=re.IGNORECASE,
    )
    legs = list(leg_pattern.finditer(text))
    speed_mentions = len(
        re.findall(
            r"\b(?:km/h|kph|kilometers?\s+per\s+hour|kilometres?\s+per\s+hour|"
            r"mph|miles?\s+per\s+hour)\b",
            text,
            re.I,
        )
    )
    if len(legs) < 2 or len(legs) != speed_mentions:
        return None

    unit_family: str | None = None
    total_distance = Fraction(0)
    total_time = Fraction(0)
    for leg in legs:
        distance = Fraction(leg.group("distance"))
        speed = Fraction(leg.group("speed"))
        if distance < 0 or speed <= 0:
            return None
        distance_unit = leg.group("distance_unit").lower()
        speed_unit = leg.group("speed_unit").lower()
        family = "metric" if distance_unit.startswith(("km", "kilo")) else "imperial"
        speed_family = "metric" if speed_unit.startswith(("km", "kilo")) else "imperial"
        if family != speed_family or (unit_family is not None and family != unit_family):
            return None
        unit_family = family
        total_distance += distance
        total_time += distance / speed
    if total_time == 0:
        return None
    unit = "km/h" if unit_family == "metric" else "mph"
    return LocalAnswer(
        f"{_format_number(total_distance / total_time)} {unit}",
        0.99,
        "weighted_average_speed",
    )


def _solve_ratio_share(text: str) -> LocalAnswer | None:
    ratio_patterns = (
        re.compile(
            r"\b(?P<first>[A-Za-z][A-Za-z'-]*)\s+and\s+"
            r"(?P<second>[A-Za-z][A-Za-z'-]*)"
            r"(?:\s+[A-Za-z][A-Za-z'-]*){0,3}\s+are\s+in\s+(?:a\s+)?"
            r"(?P<a>\d+)\s*:\s*(?P<b>\d+)\s+ratio\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bratio\s+of\s+(?P<first>[A-Za-z][A-Za-z'-]*)\s+to\s+"
            r"(?P<second>[A-Za-z][A-Za-z'-]*)\s+is\s+"
            r"(?P<a>\d+)\s*:\s*(?P<b>\d+)\b",
            flags=re.IGNORECASE,
        ),
    )
    ratio = next(
        (match for pattern in ratio_patterns if (match := pattern.search(text)) is not None),
        None,
    )
    if ratio is None:
        return None

    tail = text[ratio.end() :]
    if len(re.findall(r"\d+(?:\.\d+)?", tail)) != 1:
        return None
    total_match = re.search(
        r"\b(?:there\s+are|total(?:\s+of)?|altogether(?:\s+are|\s+is)?)\s+"
        r"(?P<total>\d+(?:\.\d+)?)\b",
        tail,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b(?P<total>\d+(?:\.\d+)?)\s+(?:[A-Za-z][A-Za-z'-]*\s+){0,3}"
        r"(?:in\s+)?total\b",
        tail,
        flags=re.IGNORECASE,
    )
    question = re.search(r"\bhow\s+many\b(?P<body>[^?]*)", tail, flags=re.IGNORECASE)
    if total_match is None or question is None:
        return None

    labels = (ratio.group("first"), ratio.group("second"))
    mentioned = [
        index
        for index, label in enumerate(labels)
        if re.search(fr"\b{re.escape(label)}\b", question.group("body"), re.I)
    ]
    if len(mentioned) != 1:
        return None
    parts = (int(ratio.group("a")), int(ratio.group("b")))
    if parts[0] <= 0 or parts[1] <= 0:
        return None
    total = Fraction(total_match.group("total"))
    result = total * parts[mentioned[0]] / sum(parts)
    return LocalAnswer(_format_number(result), 0.99, "ratio_share")


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
