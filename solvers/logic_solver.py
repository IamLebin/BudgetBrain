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
    ordering = _solve_unique_ordering(prompt)
    if ordering is not None:
        return ordering
    conditional = _solve_conditional_deduction(prompt)
    if conditional is not None:
        return conditional
    universal = _solve_universal_application(prompt)
    if universal is not None:
        return universal
    return None


def _solve_unique_ordering(prompt: str) -> LocalAnswer | None:
    setup = re.search(
        r"\b(?P<names>[A-Z][A-Za-z'-]*(?:\s*,\s*[A-Z][A-Za-z'-]*)+"
        r"(?:\s*,?\s*(?:and|&)\s*[A-Z][A-Za-z'-]*)?)\s+"
        r"(?:sit|stand|line\s+up|arrive|finish|are\s+seated)\b",
        prompt,
    )
    if setup is None:
        return None
    names = _split_list(setup.group("names"))
    if not (2 <= len(names) <= 8) or len(set(names)) != len(names):
        return None

    name_lookup = {name.lower(): name for name in names}
    name_pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    before: set[tuple[str, str]] = set()
    relation_patterns = (
        (fr"\b({name_pattern})\s+(?:sits?\s+)?(?:to\s+the\s+)?left\s+of\s+({name_pattern})\b", False),
        (fr"\b({name_pattern})\s+(?:sits?\s+)?(?:to\s+the\s+)?right\s+of\s+({name_pattern})\b", True),
        (fr"\b({name_pattern})\s+(?:comes?|arrives?|finishes?)\s+before\s+({name_pattern})\b", False),
        (fr"\b({name_pattern})\s+(?:comes?|arrives?|finishes?)\s+after\s+({name_pattern})\b", True),
    )
    for pattern, reverse in relation_patterns:
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            first = name_lookup[match.group(1).lower()]
            second = name_lookup[match.group(2).lower()]
            before.add((second, first) if reverse else (first, second))
    if not before:
        return None

    valid = [
        order
        for order in permutations(names)
        if all(order.index(first) < order.index(second) for first, second in before)
    ]
    if len(valid) != 1:
        return None
    if not re.search(
        r"\b(?:what|which)\s+is\s+(?:the\s+)?(?:their\s+)?order\b|"
        r"\border\s+from\s+(?:left\s+to\s+right|first\s+to\s+last)\b",
        prompt,
        re.I,
    ):
        return None
    return LocalAnswer(", ".join(valid[0]), 0.99, "unique_ordering")


def _solve_conditional_deduction(prompt: str) -> LocalAnswer | None:
    conditional = re.search(
        r"\bif\s+(?P<antecedent>[^,.;]+)\s*,\s*(?:then\s+)?"
        r"(?P<consequent>[^.;]+)[.;]",
        prompt,
        flags=re.IGNORECASE,
    )
    if conditional is None:
        return None
    antecedent = _normalize_proposition(conditional.group("antecedent"))
    consequent = _normalize_proposition(conditional.group("consequent"))
    tail = prompt[conditional.end() :]
    question_match = re.search(r"([^.?]+)\?\s*$", tail.strip())
    if question_match is None:
        return None
    question_text = _normalize_proposition(question_match.group(1))

    facts_text = tail[: question_match.start()]
    facts = [
        _normalize_proposition(sentence)
        for sentence in re.split(r"[.;]", facts_text)
        if sentence.strip()
    ]
    positive_facts: set[str] = set()
    negative_facts: set[str] = set()
    for fact in facts:
        positive, is_negative = _remove_negation(fact)
        (negative_facts if is_negative else positive_facts).add(positive)

    if _question_matches_proposition(question_text, consequent) and antecedent in positive_facts:
        return LocalAnswer("Yes", 0.99, "modus_ponens")
    if _question_matches_proposition(question_text, antecedent) and consequent in negative_facts:
        return LocalAnswer("No", 0.99, "modus_tollens")
    return None


def _solve_universal_application(prompt: str) -> LocalAnswer | None:
    question = re.search(r"\bdoes\s+(?P<body>[^?]+)\?", prompt, re.I)
    if question is None:
        return None
    facts = re.finditer(
        r"(?:^|[.;]\s*)(?P<subject>[A-Z][A-Za-z0-9' -]*?)\s+is\s+"
        r"(?:an?\s+|the\s+)?(?P<property>[^.;]+)[.;]",
        prompt,
    )
    question_body = _normalize_proposition(question.group("body"))
    for fact in facts:
        subject = fact.group("subject").strip()
        subject_prefix = _normalize_proposition(subject) + " "
        if not question_body.startswith(subject_prefix):
            continue
        predicate = question_body[len(subject_prefix) :]
        property_text = _normalize_proposition(fact.group("property"))
        universal = re.search(
            r"\ball\s+(?P<class>[^.;]+?)\s+" + re.escape(predicate) + r"[.;]",
            prompt,
            re.I,
        )
        if universal is None:
            continue
        class_text = _normalize_proposition(universal.group("class"))
        property_words = set(property_text.split())
        class_words = set(class_text.split())
        subject_head = subject.split()[0].lower().rstrip("s")
        class_singulars = {word.rstrip("s") for word in class_words}
        if property_words and property_words <= class_words and subject_head in class_singulars:
            return LocalAnswer("Yes", 0.99, "universal_application")
    return None


def _normalize_proposition(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return normalized.strip(" .,:;\"'")


def _remove_negation(proposition: str) -> tuple[str, bool]:
    patterns = (
        (r"\bis\s+not\b", " is "),
        (r"\bare\s+not\b", " are "),
        (r"\bcannot\b", "can"),
        (r"\bdoes\s+not\b", ""),
    )
    for pattern, replacement in patterns:
        if re.search(pattern, proposition):
            positive = re.sub(pattern, replacement, proposition, count=1)
            return _normalize_proposition(positive), True
    return proposition, False


def _question_proposition(question: str) -> str | None:
    normalized = _normalize_proposition(question)
    can_be = re.fullmatch(r"can\s+(.+?)\s+be\s+(.+)", normalized)
    if can_be:
        return _normalize_proposition(f"{can_be.group(1)} is {can_be.group(2)}")
    is_question = re.fullmatch(r"is\s+(.+)", normalized)
    if is_question:
        return _normalize_proposition(is_question.group(1))
    does_question = re.fullmatch(r"does\s+(.+)", normalized)
    if does_question:
        return _normalize_proposition(does_question.group(1))
    return None


def _question_matches_proposition(question: str, proposition: str) -> bool:
    direct = _question_proposition(question)
    if direct == proposition or question == proposition:
        return True
    for copula in ("is", "are"):
        marker = f" {copula} "
        if marker not in proposition:
            continue
        subject, predicate = proposition.split(marker, maxsplit=1)
        if question == f"{copula} {subject} {predicate}":
            return True
    return False


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
    # Every supported statement is evaluated against every possible truth assignment, and we
    # return only when exactly one globally consistent world exists. This path is deterministic,
    # not a heuristic guess, so it is safe for the agent's high-confidence local gate.
    return LocalAnswer(", ".join(true_names) if true_names else "none", 0.99, "exactly_one_truth")


def _statement_truth(statement: str, world: dict[str, bool]) -> bool | None:
    lower = statement.lower()
    for name, value in world.items():
        if re.fullmatch(fr"{re.escape(name.lower())}\s+is\s+(?:telling the truth|truthful|true)", lower):
            return value
        if re.fullmatch(fr"{re.escape(name.lower())}\s+is\s+(?:lying|a liar|false)", lower):
            return not value
    return None
