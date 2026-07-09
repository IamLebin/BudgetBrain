from __future__ import annotations

import re

from solvers.common import LocalAnswer


def solve_logic(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()
    xor = _solve_exactly_one_truth(prompt, lower)
    if xor is not None:
        return xor
    return None


def _solve_exactly_one_truth(prompt: str, lower: str) -> LocalAnswer | None:
    if "exactly one" not in lower:
        return None
    statements = re.findall(
        r"([A-Z][A-Za-z]*)\s+says\s*[\"']([^\"']+)[\"']",
        prompt,
    )
    if len(statements) != 2:
        return None
    names = [name for name, _ in statements]
    if set(names) and len(set(names)) != len(names):
        return None

    truth_assignments: list[dict[str, bool]] = []
    for mask in range(1 << len(names)):
        world = {name: bool(mask & (1 << idx)) for idx, name in enumerate(names)}
        truth_values = [_statement_truth(text, world) for _, text in statements]
        if None in truth_values:
            return None
        if sum(bool(v) for v in truth_values) == 1:
            truth_assignments.append(world)

    if len(truth_assignments) != 1:
        return None
    true_names = [name for name, truth in truth_assignments[0].items() if truth]
    return LocalAnswer(", ".join(true_names) if true_names else "none", 0.88, "exactly_one_truth")


def _statement_truth(statement: str, world: dict[str, bool]) -> bool | None:
    lower = statement.lower()
    for name, value in world.items():
        if re.fullmatch(fr"{re.escape(name.lower())}\s+is\s+(?:telling the truth|truthful|true)", lower):
            return value
        if re.fullmatch(fr"{re.escape(name.lower())}\s+is\s+(?:lying|a liar|false)", lower):
            return not value
    return None
