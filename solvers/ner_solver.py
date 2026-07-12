from __future__ import annotations

import json
import re

from solvers.common import LocalAnswer


ORG_SUFFIXES = {
    "AI",
    "Agency",
    "Association",
    "Authority",
    "Bank",
    "Center",
    "Centre",
    "College",
    "Company",
    "Commission",
    "Committee",
    "Corp",
    "Corporation",
    "Council",
    "Department",
    "Foundation",
    "Group",
    "Hospital",
    "Inc",
    "Institute",
    "Labs",
    "Ltd",
    "Ministry",
    "Museum",
    "Nations",
    "Network",
    "Organization",
    "Organisation",
    "School",
    "Systems",
    "Technologies",
    "Times",
    "Union",
    "University",
}

GEOGRAPHIC_NAMES = {
    "Buenos Aires",
    "Cape Town",
    "Costa Rica",
    "Hong Kong",
    "Kuala Lumpur",
    "Los Angeles",
    "New Delhi",
    "New York",
    "New Zealand",
    "North Korea",
    "Rio de Janeiro",
    "San Francisco",
    "Saudi Arabia",
    "South Korea",
    "United Kingdom",
    "United States",
}

LOCATION_SUFFIXES = {
    "Airport",
    "Bay",
    "Beach",
    "City",
    "County",
    "Island",
    "Lake",
    "Mountain",
    "Province",
    "River",
    "State",
    "Street",
    "Valley",
}

LOCATION_PREFIXES = {
    "Fort",
    "Lake",
    "Mount",
    "Port",
}

PERSON_TITLES = {
    "Dr",
    "Mr",
    "Mrs",
    "Ms",
    "President",
    "Professor",
    "Prof",
    "Senator",
}

WEEKDAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

DATE_RELATIVES = {
    "last",
    "next",
    "this",
}

AMBIGUOUS_ROLE_WORDS = {
    "CEO",
    "CFO",
    "CTO",
    "Director",
    "Founder",
    "Governor",
    "Minister",
    "Secretary",
}

LOCATION_PREPOSITIONS = {
    "across",
    "at",
    "from",
    "in",
    "inside",
    "near",
    "outside",
    "to",
    "toward",
    "towards",
}

PERSON_PRECEDING_VERBS = {
    "appointed",
    "called",
    "contacted",
    "hired",
    "invited",
    "met",
    "named",
    "thanked",
}

PERSON_FOLLOWING_VERBS = {
    "announced",
    "became",
    "climbed",
    "founded",
    "joined",
    "met",
    "said",
    "spoke",
    "traveled",
    "travelled",
    "visited",
    "works",
}

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

SKIP_CAPITALIZED = {
    "A",
    "An",
    "Extract",
    "Find",
    "I",
    "Identify",
    "Label",
    "The",
}


def solve_ner(prompt: str) -> LocalAnswer | None:
    if not re.search(r"\b(extract|identify|find|named entities|entities)\b", prompt, re.I):
        return None
    text = _target_text(prompt)
    entities, unresolved = _extract_entities(text)
    if not entities:
        return None
    role_words = "|".join(sorted(AMBIGUOUS_ROLE_WORDS, key=len, reverse=True))
    has_ambiguous_role = bool(re.search(fr"\b(?:{role_words})\b", text))
    confidence = 0.78 if unresolved or has_ambiguous_role else 0.9
    return LocalAnswer(json.dumps(entities, ensure_ascii=False), confidence, "regex_entities")


def _target_text(prompt: str) -> str:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    pieces = [a or b for a, b in quoted if a or b]
    if pieces:
        return pieces[-1]
    parts = re.split(r":\s*", prompt, maxsplit=1)
    return parts[-1]


def _extract_entities(text: str) -> tuple[list[dict[str, str]], bool]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    spans: list[tuple[int, int]] = []
    unresolved = False

    for match in re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        _add(entities, seen, spans, match.group(0), "EMAIL", match.span())
    for match in re.finditer(r"\bhttps?://[^\s,;]+", text):
        value = match.group(0).rstrip(".")
        _add(entities, seen, spans, value, "URL", (match.start(), match.start() + len(value)))
    for match in re.finditer(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        text,
        flags=re.IGNORECASE,
    ):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    for match in re.finditer(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?(?:\s+\d{1,2})?(?:,?\s+\d{4})\b",
        text,
        flags=re.IGNORECASE,
    ):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    date_words = "|".join(sorted(MONTHS | WEEKDAYS, key=len, reverse=True))
    for match in re.finditer(
        fr"\b(?:last|next|this)\s+(?:{date_words})\b",
        text,
        flags=re.IGNORECASE,
    ):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    for match in re.finditer(r"\b(?:today|tomorrow|yesterday|tonight)\b", text, flags=re.I):
        _add(entities, seen, spans, match.group(0), "DATE", match.span())
    for match in re.finditer(
        fr"\b(?:in|during|since|until|by|before|after)\s+({date_words})\b",
        text,
        flags=re.IGNORECASE,
    ):
        if not _overlaps(match.span(1), spans):
            _add(entities, seen, spans, match.group(1), "DATE", match.span(1))

    title_words = "|".join(sorted(PERSON_TITLES, key=len, reverse=True))
    for match in re.finditer(
        fr"\b(?:{title_words})\.?\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+)+)\b",
        text,
    ):
        full_span = match.span()
        _add(entities, seen, spans, match.group(1), "PERSON", full_span)

    org_suffixes = "|".join(sorted(ORG_SUFFIXES, key=len, reverse=True))
    for match in re.finditer(
        fr"\b(?:[A-Z][A-Za-z&.'-]*\s+)*[A-Z][A-Za-z&.'-]*\s+(?:{org_suffixes})\b",
        text,
    ):
        if not _overlaps(match.span(), spans):
            _add(entities, seen, spans, match.group(0), "ORG", match.span())

    name_pattern = (
        r"\b(?:[A-Z][A-Za-z'-]*|[A-Z]{2,})"
        r"(?:\s+(?:[A-Z][A-Za-z'-]*|[A-Z]{2,}))*\b"
    )
    for match in re.finditer(name_pattern, text):
        if _overlaps(match.span(), spans):
            continue
        value = match.group(0)
        words = value.split()
        if value in SKIP_CAPITALIZED or words[0] in MONTHS or words[0] in DATE_RELATIVES:
            continue
        label = _label_name(text, match.start(), match.end(), value)
        if label is None:
            unresolved = True
            continue
        _add(entities, seen, spans, value, label, match.span())
    return entities, unresolved


def _label_name(source: str, start: int, end: int, value: str) -> str | None:
    words = value.split()
    previous = re.findall(r"\b[A-Za-z]+\b", source[:start])
    previous_word = previous[-1].lower() if previous else ""
    following = re.findall(r"\b[A-Za-z]+\b", source[end:])
    following_word = following[0].lower() if following else ""
    if previous_word in LOCATION_PREPOSITIONS:
        return "LOCATION"
    if value in GEOGRAPHIC_NAMES:
        return "LOCATION"
    if words[-1] in ORG_SUFFIXES or value.isupper() or (len(words) == 1 and _has_internal_capital(value)):
        return "ORG"
    if words[0] in LOCATION_PREFIXES or words[-1] in LOCATION_SUFFIXES:
        return "LOCATION"
    if len(words) >= 2 and (
        previous_word in PERSON_PRECEDING_VERBS or following_word in PERSON_FOLLOWING_VERBS
    ):
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
