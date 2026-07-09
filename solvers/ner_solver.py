from __future__ import annotations

import json
import re

from solvers.common import LocalAnswer


ORG_SUFFIXES = {
    "AI",
    "Corp",
    "Corporation",
    "Foundation",
    "Inc",
    "Institute",
    "Labs",
    "Ltd",
    "University",
}

LOCATION_PREPOSITIONS = {"at", "from", "in", "near", "to"}
MONTHS = {
    "Jan",
    "January",
    "Feb",
    "February",
    "Mar",
    "March",
    "Apr",
    "April",
    "May",
    "Jun",
    "June",
    "Jul",
    "July",
    "Aug",
    "August",
    "Sep",
    "Sept",
    "September",
    "Oct",
    "October",
    "Nov",
    "November",
    "Dec",
    "December",
}


def solve_ner(prompt: str) -> LocalAnswer | None:
    if not re.search(r"\b(extract|identify|find|named entities|entities)\b", prompt, re.I):
        return None
    text = _target_text(prompt)
    entities = _extract_entities(text)
    if not entities:
        return None
    return LocalAnswer(json.dumps(entities, ensure_ascii=False), 0.84, "regex_entities")


def _target_text(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    pieces = [a or b for a, b in quoted if a or b]
    if pieces:
        return pieces[-1]
    parts = re.split(r":\s*", prompt, maxsplit=1)
    return parts[-1]


def _extract_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    spans: list[tuple[int, int]] = []

    for match in re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        _add(entities, seen, spans, match.group(0), "EMAIL", match.span())
    for match in re.finditer(r"\bhttps?://[^\s,;]+", text):
        value = match.group(0).rstrip(".")
        _add(entities, seen, spans, value, "URL", (match.start(), match.start() + len(value)))
    for match in re.finditer(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        text,
    ):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())

    name_pattern = r"\b(?:[A-Z][A-Za-z]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z]*|[A-Z]{2,}))*\b"
    for match in re.finditer(name_pattern, text):
        if _overlaps(match.span(), spans):
            continue
        value = match.group(0)
        words = value.split()
        if value in {"I", "The", "A", "An"} or words[0] in MONTHS:
            continue
        label = _label_name(text, match.start(), value)
        if label is None:
            continue
        _add(entities, seen, spans, value, label, match.span())
    return entities


def _label_name(source: str, start: int, value: str) -> str | None:
    words = value.split()
    previous = re.findall(r"\b[A-Za-z]+\b", source[:start])
    previous_word = previous[-1].lower() if previous else ""
    if previous_word in LOCATION_PREPOSITIONS:
        return "LOCATION"
    if words[-1] in ORG_SUFFIXES or value.isupper() or (len(words) == 1 and _has_internal_capital(value)):
        return "ORG"
    if len(words) >= 2:
        return "PERSON"
    return None


def _has_internal_capital(value: str) -> bool:
    return any(char.isupper() for char in value[1:])


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < used_end and end > used_start for used_start, used_end in spans)


def _add(
    items: list[dict[str, str]],
    seen: set[tuple[str, str]],
    spans: list[tuple[int, int]],
    text: str,
    label: str,
    span: tuple[int, int],
) -> None:
    key = (text, label)
    if key not in seen:
        seen.add(key)
        spans.append(span)
        items.append({"text": text, "label": label})
