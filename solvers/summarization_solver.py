from __future__ import annotations

import re

from solvers.common import LocalAnswer


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def solve_summarization(prompt: str) -> LocalAnswer | None:
    if not re.search(r"\b(?:summari[sz]e|summary|condense|overview)\b", prompt, re.I):
        return None
    body = _summary_source(prompt)
    if body is None:
        return None

    bullet_request = re.search(
        r"\b(?:in|into)\s+(?:exactly\s+)?(?P<count>[1-5]|one|two|three|four|five)\s+"
        r"bullet\s+points?\b",
        prompt,
        re.I,
    )
    sentences = _sentences(body)
    word_limit = re.search(
        r"\b(?:no\s+more\s+than|at\s+most|maximum\s+of)\s+(\d+)\s+words?\b",
        prompt,
        re.I,
    )
    if word_limit is not None and len(sentences) == 1:
        words = re.findall(r"\b[\w'-]+\b", sentences[0])
        if 4 <= len(words) <= int(word_limit.group(1)):
            return LocalAnswer(sentences[0], 0.99, "within_word_limit_passthrough")
    if bullet_request is not None:
        requested = _count_value(bullet_request.group("count"))
        if requested != len(sentences) or not (2 <= requested <= 5):
            return None
        word_counts = [len(re.findall(r"\b[\w'-]+\b", sentence)) for sentence in sentences]
        per_bullet_limit = re.search(
            r"\beach\s+(?:bullet(?:\s+point)?\s+)?(?:no\s+longer\s+than|"
            r"no\s+more\s+than|at\s+most)\s+(\d+)\s+words?\b",
            prompt,
            re.I,
        )
        max_words = int(per_bullet_limit.group(1)) if per_bullet_limit else 20
        if any(count < 3 or count > max_words for count in word_counts):
            return None
        if per_bullet_limit is None and sum(word_counts) > 50:
            return None
        answer = "\n".join(f"- {_bullet_text(sentence)}" for sentence in sentences)
        return LocalAnswer(answer, 0.93, "short_bullet_extraction")

    one_sentence = re.search(
        r"\b(?:(?:in|as)\s+(?:exactly\s+)?one\s+sentence|one[- ]sentence\s+(?:summary|overview))\b",
        prompt,
        re.I,
    )
    if one_sentence is not None and len(sentences) == 1:
        word_count = len(re.findall(r"\b[\w'-]+\b", sentences[0]))
        if 5 <= word_count <= 30:
            return LocalAnswer(sentences[0], 0.92, "already_one_sentence")
    if one_sentence is not None and len(sentences) == 2:
        counts = [len(re.findall(r"\b[\w'-]+\b", sentence)) for sentence in sentences]
        if all(4 <= count <= 35 for count in counts) and sum(counts) <= 55:
            first = sentences[0].rstrip(".!?")
            second = sentences[1].strip()
            leading_word = re.match(r"[A-Za-z]+", second)
            if leading_word is not None and leading_word.group(0)[1:].islower():
                second = second[0].lower() + second[1:]
            return LocalAnswer(f"{first}; {second}", 0.96, "two_sentence_join")
    if bullet_request is None and len(sentences) == 1:
        word_count = len(re.findall(r"\b[\w'-]+\b", sentences[0]))
        if 4 <= word_count <= 30:
            return LocalAnswer(sentences[0], 0.99, "short_source_passthrough")
    return None


def _summary_source(prompt: str) -> str | None:
    parts = re.split(r":\s*", prompt, maxsplit=1)
    if len(parts) != 2 or not parts[1].strip():
        return None
    body = parts[1].strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in {"'", '"'}:
        body = body[1:-1].strip()
    return body


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _count_value(raw: str) -> int:
    return int(raw) if raw.isdigit() else NUMBER_WORDS[raw.lower()]


def _bullet_text(sentence: str) -> str:
    cleaned = sentence.strip().rstrip(".!?")
    cleaned = re.sub(r"^(?:it\s+also|additionally)\s+", "", cleaned, flags=re.I)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned + "."
