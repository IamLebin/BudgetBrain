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

    recipe = _solve_recipe_scaling(cleaned)
    if recipe is not None:
        return recipe

    for deterministic_solver in (
        _solve_percentage_remainder,
        _solve_total_distance,
        _solve_discount_then_tax,
        _solve_team_days,
        _solve_periodic_doubling,
        _solve_rectangle_dimensions,
        _solve_compound_investment,
        _solve_fraction_remaining,
        _solve_linear_rental_cost,
        _solve_volume_cups,
    ):
        deterministic = deterministic_solver(cleaned)
        if deterministic is not None:
            return deterministic

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

    median = _solve_median(cleaned)
    if median is not None:
        return median

    symbolic = _solve_symbolic_equation(cleaned)
    if symbolic is not None:
        return symbolic

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
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
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
        r"(?:items?|units?|parts?|products?|tickets?|books?|dollars?|people|employees)\b",
        text,
        flags=re.IGNORECASE,
    )
    if not initial_match or not re.search(r"\b(remain|left|balance)\b", text, re.I):
        return None

    initial = Fraction(initial_match.group(1))
    percent_match = re.search(
        fr"\b(?:sells?|sold|ships?|shipped|uses?|used|loses?|lost|spends?|spent|removes?|removed)\s+"
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
    operation_tail = re.sub(r"\bQ[1-4]\b", "", tail, flags=re.I)
    operation_tail = re.sub(
        fr"\b(?:and|then)\s+({number})\s+more\b",
        r"sells \1",
        operation_tail,
        flags=re.I,
    )
    operation_pattern = re.compile(
        fr"\b(?P<verb>sells?|sold|ships?|shipped|uses?|used|loses?|lost|spends?|spent|removes?|removed|"
        fr"receives?|received|adds?|added|buys?|bought|gains?|gained|restocks?|restocked)\s+"
        fr"(?:another|an\s+additional)?\s*(?P<value>{number})(?:\s+more)?\b",
        flags=re.IGNORECASE,
    )
    operations = list(operation_pattern.finditer(operation_tail))
    value_spans = [operation.span("value") for operation in operations]
    for extra in re.finditer(r"-?\d+(?:\.\d+)?", operation_tail):
        if not any(extra.start() >= start and extra.end() <= end for start, end in value_spans):
            return None
    subtract_verbs = (
        "sell", "sold", "ship", "shipped", "use", "used", "lose", "lost", "spend", "spent", "remove"
    )
    for operation in operations:
        value = Fraction(operation.group("value"))
        verb = operation.group("verb").lower()
        remaining = remaining - value if verb.startswith(subtract_verbs) else remaining + value

    return LocalAnswer(_format_number(remaining), 0.98, "inventory_percent_changes")


def _solve_recipe_scaling(text: str) -> LocalAnswer | None:
    amount = re.search(
        r"\brequires?\s+(?P<amount>\d+\s*/\s*\d+|\d+(?:\.\d+)?)\s+cups?\b"
        r".{0,50}?\bfor\s+(?P<base>\d+)\s+(?P<item>[A-Za-z]+)\b",
        text,
        re.I,
    )
    if amount is None:
        return None
    target = re.search(
        fr"\b(?:for|needed\s+for)\s+(?P<target>\d+)\s+{re.escape(amount.group('item'))}\b",
        text[amount.end() :],
        re.I,
    )
    price = re.search(r"\bcosts?\s+\$?(?P<price>\d+(?:\.\d+)?)\s+per\s+cup\b", text, re.I)
    if target is None or price is None or text.count("?") < 2:
        return None
    amount_text = amount.group("amount").replace(" ", "")
    base_amount = (
        Fraction(*map(int, amount_text.split("/")))
        if "/" in amount_text
        else Fraction(amount_text)
    )
    scaled = base_amount * Fraction(int(target.group("target")), int(amount.group("base")))
    cost = scaled * Fraction(price.group("price"))
    return LocalAnswer(
        f"{_format_number(scaled)} cups; ${float(cost):.2f}",
        0.99,
        "recipe_scaling",
    )


def _solve_percentage_remainder(text: str) -> LocalAnswer | None:
    total = re.search(r"\b(?:has|contains|employs)\s+(\d+(?:\.\d+)?)\s+(?:employees|people|items|units)\b", text, re.I)
    if total is None or not re.search(r"\b(?:the\s+)?rest\s+(?:are|is|in|goes?)\b", text, re.I):
        return None
    percentages = [Fraction(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
    if len(percentages) < 2 or any(value < 0 for value in percentages) or sum(percentages) > 100:
        return None
    remainder = Fraction(total.group(1)) * (1 - sum(percentages) / 100)
    return LocalAnswer(_format_number(remainder), 0.99, "percentage_remainder")


def _solve_total_distance(text: str) -> LocalAnswer | None:
    if not re.search(r"\btotal\s+distance\b", text, re.I):
        return None
    legs = re.findall(
        r"(?:at\s+)?(\d+(?:\.\d+)?)\s*(km/h|kph|mph)\s+for\s+"
        r"(\d+(?:\.\d+)?)\s*hours?\b",
        text,
        re.I,
    )
    if len(legs) < 2 or len({unit.lower() for _, unit, _ in legs}) != 1:
        return None
    distance = sum(Fraction(speed) * Fraction(hours) for speed, _, hours in legs)
    unit = "km" if legs[0][1].lower() in {"km/h", "kph"} else "miles"
    return LocalAnswer(f"{_format_number(distance)} {unit}", 0.99, "total_distance")


def _solve_discount_then_tax(text: str) -> LocalAnswer | None:
    original = re.search(r"\boriginally\s+priced\s+at\s+\$?(\d+(?:\.\d+)?)\b", text, re.I)
    discount = re.search(r"(\d+(?:\.\d+)?)\s*%\s+discount\b", text, re.I)
    tax = re.search(r"(?:sales\s+)?tax\s+of\s+(\d+(?:\.\d+)?)\s*%", text, re.I)
    if original is None or discount is None or tax is None:
        return None
    value = Fraction(original.group(1)) * (1 - Fraction(discount.group(1)) / 100)
    value *= 1 + Fraction(tax.group(1)) / 100
    return LocalAnswer(f"${float(value):.2f}", 0.99, "discount_then_tax")


def _solve_team_days(text: str) -> LocalAnswer | None:
    total = re.search(r"\brequires?\s+(\d+(?:\.\d+)?)\s+hours?\s+of\s+work\b", text, re.I)
    schedule = re.search(
        r"\b(\d+)\s+people\s+work\s+on\s+it\s+for\s+(\d+(?:\.\d+)?)\s+hours?\s+per\s+day\b",
        text,
        re.I,
    )
    if total is None or schedule is None:
        return None
    daily = Fraction(schedule.group(1)) * Fraction(schedule.group(2))
    if daily <= 0:
        return None
    return LocalAnswer(f"{_format_number(Fraction(total.group(1)) / daily)} days", 0.99, "team_days")


def _solve_periodic_doubling(text: str) -> LocalAnswer | None:
    interval = re.search(r"\bdoubles?\s+every\s+(\d+)\s+hours?\b", text, re.I)
    initial = re.search(r"\bstarts?\s+with\s+(\d+)\b", text, re.I)
    elapsed = re.search(r"\bafter\s+(\d+)\s+hours?\b", text, re.I)
    if interval is None or initial is None or elapsed is None:
        return None
    interval_hours, elapsed_hours = int(interval.group(1)), int(elapsed.group(1))
    if interval_hours <= 0 or elapsed_hours % interval_hours:
        return None
    value = int(initial.group(1)) * 2 ** (elapsed_hours // interval_hours)
    return LocalAnswer(str(value), 0.99, "periodic_doubling")


def _solve_rectangle_dimensions(text: str) -> LocalAnswer | None:
    perimeter = re.search(r"\bperimeter\s+of\s+(\d+(?:\.\d+)?)\s*(meters?|metres?|m)\b", text, re.I)
    if perimeter is None or not re.search(r"\blength\s+(?:that\s+)?is\s+twice\s+(?:its\s+|the\s+)?width\b", text, re.I):
        return None
    width = Fraction(perimeter.group(1)) / 6
    length = width * 2
    return LocalAnswer(
        f"Length {_format_number(length)} m; Width {_format_number(width)} m",
        0.99,
        "rectangle_dimensions",
    )


def _solve_compound_investment(text: str) -> LocalAnswer | None:
    principal = re.search(r"\binvestment\s+of\s+\$?(\d+(?:\.\d+)?)\b", text, re.I)
    rate = re.search(r"\b(?:grows?|interest)\s+(?:at|of)\s+(\d+(?:\.\d+)?)\s*%", text, re.I)
    years = re.search(r"\bafter\s+(\d+)\s+years?\b", text, re.I)
    if principal is None or rate is None or years is None or "compound" not in text.lower():
        return None
    value = Fraction(principal.group(1)) * (1 + Fraction(rate.group(1)) / 100) ** int(years.group(1))
    return LocalAnswer(f"${float(value):.2f}", 0.99, "compound_investment")


def _solve_fraction_remaining(text: str) -> LocalAnswer | None:
    if not re.search(r"\b(?:fraction|portion)\b.{0,30}\b(?:left|remain)", text, re.I):
        return None
    fractions = re.findall(r"\b(\d+)\s*/\s*(\d+)\b", text)
    if len(fractions) != 2:
        return None
    remaining = Fraction(1) - sum(Fraction(int(a), int(b)) for a, b in fractions)
    if remaining < 0:
        return None
    return LocalAnswer(f"{remaining.numerator}/{remaining.denominator}", 0.99, "fraction_remaining")


def _solve_linear_rental_cost(text: str) -> LocalAnswer | None:
    rates = re.search(
        r"\bcharges?\s+\$?(\d+(?:\.\d+)?)\s+per\s+day\s+plus\s+"
        r"\$?(\d+(?:\.\d+)?)\s+per\s+mile",
        text,
        re.I,
    )
    usage = re.search(r"\b(?:rents?|rented)\s+for\s+(\d+)\s+days?\s+and\s+drives?\s+(\d+(?:\.\d+)?)\s+miles?", text, re.I)
    if rates is None or usage is None:
        return None
    cost = Fraction(rates.group(1)) * int(usage.group(1)) + Fraction(rates.group(2)) * Fraction(usage.group(2))
    return LocalAnswer(f"${float(cost):.2f}", 0.99, "linear_rental_cost")


def _solve_volume_cups(text: str) -> LocalAnswer | None:
    volume = re.search(r"\bholds?\s+(\d+(?:\.\d+)?)\s+lit(?:ers?|res?)\b", text, re.I)
    cup = re.search(r"\b(\d+(?:\.\d+)?)\s*[- ]?millilit(?:er|re)\s+cups?\b", text, re.I)
    if volume is None or cup is None or not re.search(r"\bhow\s+many\b", text, re.I):
        return None
    cup_ml = Fraction(cup.group(1))
    if cup_ml <= 0:
        return None
    count = Fraction(volume.group(1)) * 1000 / cup_ml
    return LocalAnswer(f"{_format_number(count)} cups", 0.99, "volume_cups")


def _solve_sequential_percent_changes(text: str) -> LocalAnswer | None:
    number = r"-?\d+(?:\.\d+)?"
    initial_patterns = (
        fr"\b(?:starts?|begins?)\s+(?:at|with)\s+\$?(?P<value>{number})\b",
        fr"\b(?:initial|original|starting)\s+(?:price|value|amount|cost)?\s*"
        fr"(?:is|was|of|:)\s*\$?(?P<value>{number})\b",
        fr"\b(?:price|value|amount|cost)\s+(?:is|was)\s+\$?(?P<value>{number})"
        r"(?:\s+(?:dollars?|usd))?\b",
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
        fr"decrease[sd]?|drop(?:s|ped)?|fall(?:s|en)?|reduc(?:e[sd]?|ed)|discount(?:ed|s)?)\s+"
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


def _solve_median(text: str) -> LocalAnswer | None:
    number = r"-?\d+(?:\.\d+)?"
    match = re.search(r"\bmedian\s+of\s+([^?\n]+)", text, re.I)
    if match is None:
        return None
    raw_values = match.group(1).strip().rstrip(".!")
    normalized = re.sub(r"\s*,?\s+and\s+", ",", raw_values, flags=re.I)
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) < 2 or any(re.fullmatch(number, part) is None for part in parts):
        return None
    values = sorted(Fraction(value) for value in parts)
    if len(values) < 2:
        return None
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return LocalAnswer(_format_number(median), 0.99, "median")


def _solve_symbolic_equation(text: str) -> LocalAnswer | None:
    request = re.search(r"\bsolve\s+for\s+([A-Za-z])\b", text, re.I)
    equation = re.search(
        r"(?<![A-Za-z0-9])([A-Za-z])\s*([+-])\s*([A-Za-z])\s*=\s*([A-Za-z])"
        r"(?![A-Za-z0-9])",
        text,
        re.I,
    )
    if request is None or equation is None:
        return None
    if len(re.findall(r"(?<![A-Za-z0-9])[A-Za-z]\s*[+-]\s*[A-Za-z]\s*=\s*[A-Za-z]", text)) != 1:
        return None
    target = request.group(1).lower()
    left, op, right, result = (
        equation.group(1).lower(),
        equation.group(2),
        equation.group(3).lower(),
        equation.group(4).lower(),
    )
    if len({left, right, result}) != 3 or target not in {left, right, result}:
        return None
    if target == result:
        isolated = f"{left} {op} {right}"
    elif target == left:
        inverse = "-" if op == "+" else "+"
        isolated = f"{result} {inverse} {right}"
    elif op == "+":
        isolated = f"{result} - {left}"
    else:
        isolated = f"{left} - {result}"
    return LocalAnswer(f"{target} = {isolated}", 0.99, "symbolic_isolation")


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
    if text.count("?") > 1:
        return None
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
