from __future__ import annotations

from itertools import permutations
import re

from solvers.common import LocalAnswer


def solve_logic(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()
    xor = _solve_exactly_one_truth(prompt, lower)
    if xor is not None:
        return xor
    assignment = _solve_one_to_one_assignment(prompt)
    if assignment is not None:
        return assignment
    return None


def _solve_one_to_one_assignment(prompt: str) -> LocalAnswer | None:
    setup = re.search(
        r"\b(?:friends|people|students|colleagues|children|players|contestants)\s*,\s*"
        r"(.+?),\s*each\s+(?:owns?|has|gets?|chooses?|is assigned)\s+"
        r"(?:a\s+)?different\s+[^:.;]+:\s*([^.;]+)",
        prompt,
        flags=re.IGNORECASE | re.S,
    )
    if not setup:
        return None

    names = _split_list(setup.group(1))
    items = _split_list(setup.group(2))
    if not (2 <= len(names) <= 7) or len(names) != len(items):
        return None
    if len({name.lower() for name in names}) != len(names):
        return None
    if len({item.lower() for item in items}) != len(items):
        return None

    item_lookup = {item.lower(): item for item in items}
    item_pattern = "|".join(re.escape(item) for item in sorted(items, key=len, reverse=True))
    required: dict[str, str] = {}
    forbidden: set[tuple[str, str]] = set()

    for name in names:
        escaped_name = re.escape(name)
        negative_patterns = (
            fr"\b{escaped_name}\s+(?:does\s+not|doesn't|did\s+not|cannot|can't)\s+"
            fr"(?:own|have|get|choose|take)\s+(?:the\s+)?({item_pattern})\b",
            fr"\b{escaped_name}(?:'s)?\s+[^.;:]*?\s+is\s+not\s+(?:the\s+)?({item_pattern})\b",
        )
        for pattern in negative_patterns:
            for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
                forbidden.add((name, item_lookup[match.group(1).lower()]))

        positive_patterns = (
            fr"\b{escaped_name}\s+(?:owns?|has|gets?|chooses?|takes?)\s+"
            fr"(?:the\s+)?({item_pattern})\b",
            fr"\b{escaped_name}(?:'s)?\s+[^.;:]*?\s+is\s+(?:the\s+)?({item_pattern})\b",
        )
        for pattern in positive_patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                required[name] = item_lookup[match.group(1).lower()]
                break

    valid: list[dict[str, str]] = []
    for order in permutations(items):
        world = dict(zip(names, order))
        if any(world[name] != item for name, item in required.items()):
            continue
        if any(world[name] == item for name, item in forbidden):
            continue
        valid.append(world)

    if not valid:
        return None

    who_question = re.search(
        fr"\bwho\s+(?:owns?|has|gets?|chose|chooses|takes?)\s+(?:the\s+)?({item_pattern})\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if who_question:
        target = item_lookup[who_question.group(1).lower()]
        answers = {
            name
            for world in valid
            for name, item in world.items()
            if item == target
        }
        if len(answers) == 1:
            return LocalAnswer(next(iter(answers)), 0.97, "one_to_one_assignment")

    for name in names:
        what_question = re.search(
            fr"\bwhat\s+[^?]*?\s+does\s+{re.escape(name)}\s+"
            r"(?:own|have|get|choose|take)\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if what_question:
            answers = {world[name] for world in valid}
            if len(answers) == 1:
                return LocalAnswer(next(iter(answers)), 0.97, "one_to_one_assignment")
    return None


def _split_list(text: str) -> list[str]:
    cleaned = re.sub(r"\s+and\s+", ",", text.strip(), flags=re.IGNORECASE)
    return [part.strip(" ,") for part in cleaned.split(",") if part.strip(" ,")]


def _solve_exactly_one_truth(prompt: str, lower: str) -> LocalAnswer | None:
    if "exactly one" not in lower:
        return None
    statements = re.findall(
        r"([A-Z][A-Za-z]*)\s+says\s*[\"']([^\"']+)[\"']",
        prompt,
    )
    if len(statements) < 2:
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
        speakers_match_statements = all(
            world[speaker] == bool(truth_values[idx])
            for idx, (speaker, _) in enumerate(statements)
        )
        if speakers_match_statements and sum(bool(v) for v in truth_values) == 1:
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
