from __future__ import annotations

import ast
import re

from solvers.common import LocalAnswer


def solve_code_generation(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()
    if "sql" in lower:
        no_orders = _solve_customers_without_orders_sql(lower)
        if no_orders:
            return _answer(no_orders, "customers_without_orders_sql_generation", python=False)
        active_customers = _solve_active_customers_sql(lower)
        if active_customers:
            return _answer(active_customers, "active_customers_sql_generation", python=False)
        sql = _solve_grouped_average_sql(lower)
        return _answer(sql, "grouped_average_sql_generation", python=False) if sql else None
    if re.search(r"\b(?:javascript|typescript|java|c\+\+|rust)\b", lower):
        return None

    signature = _requested_signature(prompt)
    if re.search(r"\bunique[_ ]words\b", lower) and re.search(
        r"\blowercase\b", lower
    ) and re.search(r"\b(?:ignore|ignoring|remove)\w*\s+punctuation\b", lower):
        name, parameter = signature or ("unique_words", "text")
        return _answer(
            "import re\n\n"
            f"def {name}({parameter}):\n"
            f"    return set(re.findall(r\"[a-z0-9']+\", {parameter}.lower()))",
            "unique_words_generation",
        )
    merge_signature = re.search(
        r"\b(?:function|def)\s+([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)",
        prompt,
        re.I,
    )
    if merge_signature and re.search(r"\bmerge[_ ]sorted\b", lower) and re.search(
        r"\b(?:two|2)\s+(?:already\s+)?sorted\s+lists?\b", lower
    ):
        name, first, second = merge_signature.groups()
        return _answer(
            f"def {name}({first}, {second}):\n"
            "    merged = []\n"
            "    i = j = 0\n"
            f"    while i < len({first}) and j < len({second}):\n"
            f"        if {first}[i] <= {second}[j]:\n"
            f"            merged.append({first}[i])\n"
            "            i += 1\n"
            "        else:\n"
            f"            merged.append({second}[j])\n"
            "            j += 1\n"
            f"    return merged + {first}[i:] + {second}[j:]",
            "merge_sorted_generation",
        )
    if re.search(r"\bsquare\b", lower) and re.search(
        r"\b(?:multipl(?:y|ied)\s+by|times)\s+(?:itself|the\s+number)\b|\bn\s*\*\s*n\b",
        lower,
    ):
        name = signature[0] if signature else (_requested_function_name(prompt) or "square")
        parameter_match = re.search(
            r"\breturns?\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:multiplied\s+by|times)\s+itself\b",
            prompt,
            re.I,
        )
        parameter = signature[1] if signature else (parameter_match.group(1) if parameter_match else "n")
        return _answer(
            f"def {name}({parameter}):\n    return {parameter} * {parameter}",
            "square_generation",
        )
    if re.search(r"\brevers(?:e|es|ing)\s+(?:a\s+|the\s+)?list\b", lower) and not re.search(
        r"\bin[- ]place\b|\bwithout\s+(?:slicing|built-?ins?)\b",
        lower,
    ):
        name, parameter = signature or ("reverse_list", "items")
        return _answer(
            f"def {name}({parameter}):\n    return {parameter}[::-1]",
            "reverse_list_generation",
        )
    if re.search(r"\brevers(?:e|es|ing)\s+(?:a\s+|the\s+)?string\b", lower) and not re.search(
        r"\bwithout\s+slicing\b",
        lower,
    ):
        name, parameter = signature or ("reverse_string", "text")
        return _answer(
            f"def {name}({parameter}):\n    return {parameter}[::-1]",
            "reverse_string_generation",
        )
    if re.search(r"\b(?:is|whether)\s+[A-Za-z_][A-Za-z0-9_]*\s+is\s+even\b", lower):
        name, parameter = signature or ("is_even", "n")
        return _answer(
            f"def {name}({parameter}):\n    return {parameter} % 2 == 0",
            "is_even_generation",
        )
    if re.search(r"\b(?:returns?|calculates?)\s+the\s+(?:sum|total)\s+of\s+(?:a\s+|the\s+)?list\b", lower):
        name, parameter = signature or ("sum_list", "numbers")
        return _answer(
            f"def {name}({parameter}):\n    return sum({parameter})",
            "sum_list_generation",
        )
    if re.search(r"\bcounts?\s+(?:the\s+)?vowels\s+in\s+(?:a\s+|the\s+)?string\b", lower):
        name, parameter = signature or ("count_vowels", "text")
        return _answer(
            f"def {name}({parameter}):\n"
            f"    return sum(ch.lower() in 'aeiou' for ch in {parameter})",
            "count_vowels_generation",
        )
    if re.search(r"\bsecond[- ]largest\b", lower) and not re.search(
        r"\b(?:count|include|including|allow)\w*\s+duplicates?\b", lower
    ):
        name, parameter = signature or ("second_largest", "numbers")
        return _answer(
            f"def {name}({parameter}):\n    return sorted(set({parameter}), reverse=True)[1]",
            "second_largest_generation",
        )
    if "palindrome" in lower:
        name, parameter = signature or ("is_palindrome", "text")
        return _answer(
            f"def {name}({parameter}):\n"
            f"    cleaned = ''.join(ch.lower() for ch in {parameter} if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]",
            "palindrome_generation",
        )
    if ("balanced" in lower or re.search(r"\bmatching\b", lower)) and re.search(
        r"\b(?:brackets?|parentheses)\b|\(\)|\[\]|\{\}", lower
    ):
        name, parameter = signature or ("is_balanced", "text")
        return _answer(
            f"def {name}({parameter}):\n"
            "    stack = []\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            f"    for ch in {parameter}:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack",
            "balanced_brackets_generation",
        )
    if re.search(r"\bmerge\w*\s+(?:all\s+)?overlapping\s+intervals?\b", lower):
        name, parameter = signature or ("merge_intervals", "intervals")
        return _answer(
            f"def {name}({parameter}):\n"
            f"    ordered = sorted({parameter})\n"
            "    merged = []\n"
            "    for start, end in ordered:\n"
            "        if not merged or start > merged[-1][1]:\n"
            "            merged.append([start, end])\n"
            "        else:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "    return merged",
            "merge_intervals_generation",
        )
    return None


def _requested_signature(prompt: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:function|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
        prompt,
        re.I,
    )
    return (match.group(1), match.group(2)) if match else None


def _requested_function_name(prompt: str) -> str | None:
    match = re.search(
        r"\bfunction\s+(?:called|named)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        prompt,
        re.I,
    )
    return match.group(1) if match else None


def _solve_grouped_average_sql(lower: str) -> str | None:
    direct = re.search(
        r"(?:each\s+)?(?P<group>[a-z_][a-z0-9_]*)\s+and\s+(?:its\s+)?average\s+"
        r"(?P<metric>[a-z_][a-z0-9_]*)\s+from\s+(?P<table>[a-z_][a-z0-9_]*)",
        lower,
    )
    alternate = re.search(
        r"average\s+(?P<metric>[a-z_][a-z0-9_]*)\s+(?:per|by)\s+"
        r"(?P<group>[a-z_][a-z0-9_]*)\s+from\s+(?P<table>[a-z_][a-z0-9_]*)",
        lower,
    )
    match = direct or alternate
    if not match:
        return None
    group, metric, table = match.group("group"), match.group("metric"), match.group("table")
    alias = f"average_{metric}"
    direction = "DESC" if re.search(r"\b(?:desc|descending|highest|largest)\b", lower) else "ASC"
    return (
        f"SELECT {group}, AVG({metric}) AS {alias}\nFROM {table}\n"
        f"GROUP BY {group}\nORDER BY {alias} {direction};"
    )


def _solve_customers_without_orders_sql(lower: str) -> str | None:
    if not (
        re.search(r"\bcustomers?\b", lower)
        and re.search(r"\borders?\b", lower)
        and re.search(r"\b(?:no|without|never)\s+orders?\b|\bplaced\s+no\s+orders?\b", lower)
        and re.search(r"customers\s*\(\s*id\s*,\s*name\s*\)", lower)
        and re.search(r"orders\s*\(\s*customer_id\s*\)", lower)
    ):
        return None
    return (
        "SELECT c.name\nFROM customers AS c\n"
        "WHERE NOT EXISTS (\n"
        "    SELECT 1 FROM orders AS o WHERE o.customer_id = c.id\n"
        ");"
    )


def _solve_active_customers_sql(lower: str) -> str | None:
    if not re.search(r"\bactive\s+customers?\b", lower):
        return None
    if re.search(r"\b(?:join|orders?|group|average|count|sum|limit|inactive)\b", lower):
        return None
    selected = "name" if re.search(r"\b(?:names?|customer_name)\b", lower) else "*"
    return f"SELECT {selected} FROM customers WHERE active = TRUE;"


def _answer(code: str, method: str, *, python: bool = True) -> LocalAnswer:
    if python:
        ast.parse(code)
    return LocalAnswer(code, 0.99, method)
