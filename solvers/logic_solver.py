from __future__ import annotations

import re
from itertools import permutations

from solvers.common import LocalAnswer


def solve_logic(prompt: str) -> LocalAnswer | None:
    lower = prompt.lower()

    result = _solve_truth_teller(prompt, lower)
    if result is not None:
        return result

    result = _solve_ordering(prompt, lower)
    if result is not None:
        return result

    result = _solve_deduction(prompt, lower)
    if result is not None:
        return result

    return None


# ---------------------------------------------------------------------------
# 1. N-person truth-teller / liar puzzles
#    Pattern: "exactly one statement is true" (or N-1 are lying, etc.)
#    Each person says "<Name> is lying/truthful" or "<Name> is a liar/truth teller"
#    Brute-force all 2^N truth/lie assignments and check consistency.
# ---------------------------------------------------------------------------

def _solve_truth_teller(prompt: str, lower: str) -> LocalAnswer | None:
    # Determine how many truth-tellers are expected
    n_true = _expected_truth_count(lower)
    if n_true is None:
        return None

    # Extract statements: Name says "claim"
    statements = re.findall(
        r"([A-Z][A-Za-z]*)\s+says\s*[\"']([^\"']+)[\"']",
        prompt,
    )
    if len(statements) < 2:
        return None

    names = [name for name, _ in statements]
    if len(set(names)) != len(names):
        return None  # duplicate speakers — can't model cleanly

    # Try all 2^N assignments
    valid_worlds: list[dict[str, bool]] = []
    for mask in range(1 << len(names)):
        world = {name: bool(mask & (1 << idx)) for idx, name in enumerate(names)}

        # Each statement must be evaluable
        truth_values = [_statement_truth(text, world) for _, text in statements]
        if None in truth_values:
            return None  # statement we can't parse — bail

        # Check: speaker is truthful iff their statement is true
        consistent = True
        for idx, (speaker, _) in enumerate(statements):
            if world[speaker] != truth_values[idx]:
                consistent = False
                break
        if not consistent:
            continue

        # Check: number of truth-tellers matches the constraint
        if sum(1 for v in world.values() if v) == n_true:
            valid_worlds.append(world)

    if not valid_worlds:
        return None  # no consistent assignment found

    # Pick the first valid world (handles symmetric puzzles like "A says B lies, B says A lies")
    truth_tellers = [name for name, v in valid_worlds[0].items() if v]
    if truth_tellers:
        answer = ", ".join(truth_tellers)
    else:
        answer = "none"
    # Lower confidence if the puzzle has multiple valid solutions
    confidence = 0.90 if len(valid_worlds) == 1 else 0.85
    return LocalAnswer(answer, confidence, "truth_teller")


def _expected_truth_count(lower: str) -> int | None:
    """Determine how many truth-tellers the puzzle says there are."""
    if "exactly one" in lower and ("true" in lower or "truth" in lower or "statement" in lower):
        return 1
    if "exactly two" in lower and ("true" in lower or "truth" in lower or "statement" in lower):
        return 2
    # "one of them is telling the truth" patterns
    if re.search(r"\bone\b.*\btelling the truth\b", lower):
        return 1
    # "only one is honest"
    if re.search(r"\b(?:only|exactly)\s+one\b.*\bhonest\b", lower):
        return 1
    return None


def _statement_truth(statement: str, world: dict[str, bool]) -> bool | None:
    """Evaluate a claim like 'Bob is lying' against a truth-assignment world."""
    lower = statement.strip().lower()
    for name, value in world.items():
        name_lower = re.escape(name.lower())
        # "<Name> is telling the truth / truthful / honest"
        if re.fullmatch(
            rf"{name_lower}\s+is\s+(?:telling the truth|truthful|true|honest)",
            lower,
        ):
            return value
        # "<Name> is lying / a liar / dishonest"
        if re.fullmatch(
            rf"{name_lower}\s+is\s+(?:lying|a liar|false|dishonest)",
            lower,
        ):
            return not value
        # "I am telling the truth" — speaker self-reference handled upstream
    return None


# ---------------------------------------------------------------------------
# 2. Ordering / ranking puzzles
#    Pattern: "A is taller/older/faster than B" chains.
#    Extract pairwise > relations, topological-sort, answer superlative.
# ---------------------------------------------------------------------------

_COMPARATIVE_PATTERN = re.compile(
    r"([A-Z][A-Za-z]*)\s+is\s+\w+er\s+than\s+([A-Z][A-Za-z]*)",
    re.IGNORECASE,
)

# What superlative maps to which comparison direction
_SUPERLATIVE_MAP: dict[str, str] = {
    "tallest": "taller",
    "shortest": "shorter",
    "oldest": "older",
    "youngest": "younger",
    "fastest": "faster",
    "slowest": "slower",
    "heaviest": "heavier",
    "lightest": "lighter",
    "richest": "richer",
    "poorest": "poorer",
    "smartest": "smarter",
}


def _solve_ordering(prompt: str, lower: str) -> LocalAnswer | None:
    pairs = _COMPARATIVE_PATTERN.findall(prompt)
    if len(pairs) < 2:
        return None

    # Build a directed graph: greater -> lesser
    # Normalise names
    greater_than: dict[str, set[str]] = {}
    all_names: set[str] = set()
    for a, b in pairs:
        greater_than.setdefault(a, set()).add(b)
        all_names.update((a, b))

    # Topological sort (Kahn's algorithm)
    in_degree: dict[str, int] = {name: 0 for name in all_names}
    for node, children in greater_than.items():
        for child in children:
            in_degree[child] = in_degree.get(child, 0) + 1

    queue = [n for n in all_names if in_degree[n] == 0]
    order: list[str] = []
    while queue:
        if len(queue) > 1:
            return None  # ambiguous ordering
        node = queue.pop()
        order.append(node)
        for child in greater_than.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(all_names):
        return None  # cycle

    # Figure out what the question is asking
    # "Who is the tallest?" -> first in order
    # "Who is the shortest?" -> last in order
    superlative_match = re.search(
        r"who\s+is\s+(?:the\s+)?(\w+est)\b", lower,
    )
    if superlative_match:
        superlative = superlative_match.group(1)
        # "shortest", "youngest", "slowest", etc. are the opposite end
        if superlative in ("shortest", "youngest", "slowest", "lightest", "poorest"):
            answer = order[-1]
        else:
            answer = order[0]
        return LocalAnswer(answer, 0.92, "ordering")

    # "rank from tallest to shortest" / "put in order"
    if re.search(r"rank|order|sort", lower):
        return LocalAnswer(", ".join(order), 0.90, "ordering")

    return None


# ---------------------------------------------------------------------------
# 3. Simple deduction (modus tollens / ponens)
#    Pattern: "If P then Q. Q is false. Therefore P?"
# ---------------------------------------------------------------------------

_IF_THEN_PATTERN = re.compile(
    r"[Ii]f\s+(.+?)\s*,\s*(?:then\s+)?(.+?)\.",
)


def _solve_deduction(prompt: str, lower: str) -> LocalAnswer | None:
    rules = _IF_THEN_PATTERN.findall(prompt)
    if not rules:
        return None

    # Extract the question from the last sentence
    sentences = re.split(r'(?<=[.!?])\s+', prompt.strip())
    last_sentence = sentences[-1] if sentences else prompt
    question_match = re.search(
        r"(?:did|does|is|was|will|can|could|has)\s+(.+?)\??$",
        last_sentence.strip(),
        re.IGNORECASE,
    )
    if not question_match:
        return None

    question_subject = question_match.group(1).strip().lower().rstrip('?')

    for antecedent, consequent in rules:
        ant_lower = antecedent.strip().lower()
        con_lower = consequent.strip().lower()

        # Modus ponens: antecedent is stated true -> consequent is true
        if _fact_stated(prompt, antecedent):
            if _matches_question(con_lower, question_subject):
                return LocalAnswer("Yes", 0.88, "modus_ponens")

        # Modus tollens: consequent is denied -> antecedent is false
        if _fact_denied(prompt, consequent):
            if _matches_question(ant_lower, question_subject):
                return LocalAnswer("No", 0.88, "modus_tollens")

    return None


def _fact_stated(prompt: str, fact: str) -> bool:
    """Check if the prompt explicitly states this fact as true."""
    # Look for the fact stated as a standalone sentence
    escaped = re.escape(fact.strip().rstrip("."))
    return bool(re.search(rf"(?:^|\.\s+){escaped}\s*\.", prompt, re.IGNORECASE))


def _fact_denied(prompt: str, fact: str) -> bool:
    """Check if the prompt explicitly states this fact is false / not the case."""
    lower = prompt.lower()
    fact_lower = fact.strip().lower().rstrip(".")

    # "The ground is not wet" / "The ground is dry"
    # Check for "not <fact>" or "<subject> is not <predicate>"
    # Simple approach: look for negated version
    if f"not {fact_lower}" in lower:
        return True

    # "X is dry" as denial of "X is wet" — common opposites
    opposites = {
        "wet": "dry", "dry": "wet",
        "hot": "cold", "cold": "hot",
        "true": "false", "false": "true",
        "open": "closed", "closed": "open",
        "on": "off", "off": "on",
        "happy": "sad", "sad": "happy",
    }

    # Match pattern: "<subject> is <adjective>"
    fact_match = re.match(r"(?:the\s+)?(.+?)\s+is\s+(\w+)", fact_lower)
    if fact_match:
        subject = fact_match.group(1)
        adjective = fact_match.group(2)
        opposite = opposites.get(adjective)
        if opposite and re.search(
            rf"\b{re.escape(subject)}\s+is\s+{re.escape(opposite)}\b",
            lower,
        ):
            return True

    return False


def _matches_question(fact_lower: str, question_lower: str) -> bool:
    """Check if a fact roughly matches what the question is asking about."""
    # Strip common prefixes
    for prefix in ("it ", "the ", "that ", "this "):
        fact_lower = fact_lower.removeprefix(prefix)
        question_lower = question_lower.removeprefix(prefix)
    # Fuzzy: one contains the other or they share significant overlap
    return fact_lower in question_lower or question_lower in fact_lower
