from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODELS = (
    "minimax-m3",
    "kimi-k2p7-code",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
    "gemma-4-31b-it",
)

MODEL_PREFERENCE = {
    "factual_qa": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "math": ("minimax-m3", "kimi-k2p7-code", "gemma-4-26b-a4b-it"),
    "sentiment": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "summarization": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "ner": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "code_debugging": ("kimi-k2p7-code", "minimax-m3", "gemma-4-31b-it-nvfp4"),
    "logic": ("minimax-m3", "kimi-k2p7-code", "gemma-4-26b-a4b-it"),
    "code_generation": ("kimi-k2p7-code", "minimax-m3", "gemma-4-31b-it-nvfp4"),
}

MAX_TOKENS = {
    "factual_qa": 256,
    "math": 192,
    "sentiment": 96,
    "summarization": 224,
    "ner": 224,
    "code_debugging": 384,
    "logic": 384,
    "code_generation": 512,
}

SYSTEM_PROMPTS = {
    "factual_qa": (
        "Answer every requested part using standard textbook terms and the requested format. "
        "Unless the user asks for detail, use one sentence of at most 35 words with no preamble."
    ),
    "math": "Solve carefully and verify the arithmetic. Follow the requested format and detail level.",
    "sentiment": (
        "Classify the full text using only allowed labels. When a reason is requested, explicitly "
        "mention every positive and negative aspect from the input instead of grouping or "
        "generalizing them; preserve concrete timing, quantity, delivery, damage, and resolution "
        "details, and obey the requested sentence count."
    ),
    "summarization": (
        "Preserve key facts and obey every length, count, and format constraint. Cover every "
        "explicitly listed capability, benefit, drawback, risk, cause, response, and named actor "
        "when the requested length allows it; compress wording instead of dropping an enumerated item."
    ),
    "ner": (
        "Extract all entities in the requested format using PERSON, ORG, LOCATION, and DATE "
        "unless labels are provided."
    ),
    "code_debugging": (
        "Briefly name the bug, then show the smallest runnable fix in a fenced block. Preserve "
        "the signature and structure; prefer built-ins and give one fix only. Omit the diagnosis "
        "only if code-only output is explicitly requested."
    ),
    "logic": "Satisfy every stated constraint, verify the conclusion, and use the requested format.",
    "code_generation": (
        "Return only the shortest clear runnable code satisfying the stated requirements. "
        "Do not add validation or edge-case behavior that was not requested."
    ),
}


def _system_prompt(category: str, prompt: str) -> str:
    base = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_qa"])
    if category == "factual_qa" and re.search(r"\b(?:define|what\s+is|explain\s+what)\b", prompt, re.I):
        base += " For technical definitions, state the core mechanism, not only the purpose or outcome."
    example_request = re.search(
        r"\b(?:provide|give|name)\s+(two|three|2|3)\b.{0,30}\bexamples?\b",
        prompt,
        re.I,
    )
    if category == "factual_qa" and example_request is not None:
        count = {"two": "2", "three": "3"}.get(
            example_request.group(1).lower(),
            example_request.group(1),
        )
        base += f" Provide exactly {count} examples labeled Example 1, Example 2, and so on."
    if category == "factual_qa" and _is_comparison_question(prompt):
        comparison = (
            base
            + " For comparisons, explicitly state hierarchy or subset relationships and contrast "
            "mechanism, feature handling, key properties such as volatility and relative speed, "
            "and uses whenever relevant. Do not omit a comparison dimension."
        )
        if _is_ml_deep_comparison(prompt):
            comparison += (
                " Explicitly contrast manual feature engineering in traditional machine learning "
                "with automatic feature learning in deep neural networks."
            )
        return comparison
    if category == "summarization" and re.search(r"\bbullet\s+points?\b", prompt, re.I):
        return (
            base
            + " Use each requested bullet for a different major theme. When present, cover benefits, "
            "drawbacks or risks, and responses; do not repeat one theme while omitting another."
        )
    if category == "ner":
        return base + " Copy the user's label names exactly; do not abbreviate ORGANIZATION as ORG."
    return base


def _is_comparison_question(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(?:difference\s+between|compare|comparison|contrast|versus|vs\.?|"
            r"how\s+does\s+.+\s+differ\s+from)\b",
            prompt,
            re.I,
        )
    )


def _is_ml_deep_comparison(prompt: str) -> bool:
    return bool(
        re.search(r"\bmachine\s+learning\b", prompt, re.I)
        and re.search(r"\b(?:deep\s+learning|deep\s+neural|neural\s+network)\b", prompt, re.I)
    )


class FireworksError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FireworksClient:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.allowed_models = allowed_models
        self.last_tokens_used: int | None = None
        self._reasoning_effort_supported = True

    @classmethod
    def from_env(cls) -> "FireworksClient":
        api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
        if not api_key:
            raise FireworksError("FIREWORKS_API_KEY is not set")
        base_url = os.getenv("FIREWORKS_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        allowed = parse_allowed_models(os.getenv("ALLOWED_MODELS", ""))
        if not allowed:
            raise FireworksError("ALLOWED_MODELS is not set")
        return cls(api_key=api_key, base_url=base_url, allowed_models=allowed)

    def solve(self, prompt: str, category: str) -> str:
        self.last_tokens_used = None
        errors: list[str] = []
        for model in self.candidate_models_for_prompt(category, prompt):
            api_model = model_id_for_request(model, self.base_url)
            payload = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": _system_prompt(category, prompt)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": MAX_TOKENS.get(category, 128),
            }
            reasoning_effort = _reasoning_effort(model, category, prompt)
            if reasoning_effort is not None and self._reasoning_effort_supported:
                payload["reasoning_effort"] = reasoning_effort
            try:
                response = self._post_completion(payload, api_model)
                self._record_usage(response.get("usage"))
                self._log_usage(category, api_model, response.get("usage"))
                content = _response_content(response)
                if _response_was_truncated(response):
                    answer = _recover_truncated_answer(content, category, prompt)
                    if answer is None:
                        raise FireworksError("model answer was truncated", status_code=502)
                    print(
                        f"recovered complete factual sentence from truncated answer: {api_model}",
                        file=sys.stderr,
                    )
                else:
                    answer = clean_model_answer(content, category, prompt)
                validate_model_answer(answer, category, prompt)
                return answer
            except (KeyError, IndexError, TypeError) as exc:
                wrapped = FireworksError(
                    f"unexpected Fireworks response shape: {exc}",
                    status_code=502,
                )
                errors.append(f"{api_model}: {wrapped}")
                print(f"invalid model response, trying next: {api_model} ({wrapped})", file=sys.stderr)
                continue
            except FireworksError as exc:
                if not _should_try_next_model(exc):
                    raise
                errors.append(f"{api_model}: {exc}")
                print(f"model failed, trying next: {api_model} ({exc})", file=sys.stderr)
        raise FireworksError("all candidate Fireworks models failed: " + "; ".join(errors))

    def _post_completion(self, payload: dict[str, Any], api_model: str) -> dict[str, Any]:
        try:
            return self._post_json("/chat/completions", payload)
        except FireworksError as exc:
            if exc.status_code != 400 or "reasoning_effort" not in payload:
                raise
            compatible_payload = dict(payload)
            compatible_payload.pop("reasoning_effort", None)
            print(
                f"proxy rejected reasoning_effort; retrying without it: {api_model}",
                file=sys.stderr,
            )
            response = self._post_json("/chat/completions", compatible_payload)
            self._reasoning_effort_supported = False
            return response

    def pick_model(self, category: str) -> str:
        candidates = self.candidate_models(category)
        if not candidates:
            raise FireworksError("ALLOWED_MODELS is empty")
        return candidates[0]

    def candidate_models(self, category: str) -> list[str]:
        preferred = MODEL_PREFERENCE.get(category, DEFAULT_MODELS)
        candidates: list[str] = []
        for preferred_name in preferred:
            for allowed_model in self.allowed_models:
                if canonical_model_name(allowed_model) == preferred_name and allowed_model not in candidates:
                    candidates.append(allowed_model)
        for model in self.allowed_models:
            if model not in candidates:
                candidates.append(model)
        return candidates

    def candidate_models_for_prompt(self, category: str, prompt: str) -> list[str]:
        candidates = self.candidate_models(category)
        if (
            category == "factual_qa" and _is_comparison_question(prompt)
        ) or (
            category == "factual_qa"
            and re.search(r"\bexplain\b", prompt, re.I)
            and re.search(r"\b(?:and|then)\s+(?:name|list|provide|give)\b", prompt, re.I)
        ) or (
            category == "summarization" and re.search(r"\bbullet\s+points?\b", prompt, re.I)
        ) or (
            category == "sentiment"
            and re.search(r"\b(?:reason|reasoning|justify|explain)\b", prompt, re.I)
        ):
            candidates.sort(key=lambda model: canonical_model_name(model) != "minimax-m3")
        return candidates

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "BudgetBrain-Track1/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=25) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FireworksError(f"HTTP {exc.code}: {detail[:400]}", status_code=exc.code) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise FireworksError(str(exc)) from exc

    @staticmethod
    def _log_usage(category: str, model: str, usage: Any) -> None:
        if isinstance(usage, dict):
            total = usage.get("total_tokens", "?")
            prompt = usage.get("prompt_tokens", "?")
            completion = usage.get("completion_tokens", "?")
            print(
                f"usage category={category} model={model} total={total} prompt={prompt} completion={completion}",
                file=sys.stderr,
            )

    def _record_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        total = usage.get("total_tokens")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            return
        self.last_tokens_used = (self.last_tokens_used or 0) + total


def parse_allowed_models(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    models = [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]
    return list(dict.fromkeys(models))


def normalize_model_id(model: str) -> str:
    if "/" in model:
        return model
    return f"accounts/fireworks/models/{model}"


def model_id_for_request(model: str, base_url: str) -> str:
    """Keep proxy model IDs exact; normalize shorthand only for direct Fireworks calls."""
    host = (urlparse(base_url).hostname or "").lower()
    if host == "api.fireworks.ai":
        return normalize_model_id(model)
    return model


def canonical_model_name(model: str) -> str:
    return model.rstrip("/").rsplit("/", 1)[-1]


def _response_content(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
        if content.strip():
            return content
        raise FireworksError("Fireworks returned an empty answer", status_code=502)
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {None, "text"}
        ]
        joined = "".join(parts)
        if joined.strip():
            return joined
    raise FireworksError("Fireworks returned unsupported answer content", status_code=502)


def _response_was_truncated(response: dict[str, Any]) -> bool:
    choices = response.get("choices")
    return bool(
        isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
        and choices[0].get("finish_reason") == "length"
    )


def _recover_truncated_answer(content: str, category: str, prompt: str) -> str | None:
    if category != "factual_qa":
        return None
    complete_sentence = re.match(r"^(.+?[.!?])(?:\s|$)", content.strip(), flags=re.S)
    if complete_sentence is None:
        return None
    answer = clean_model_answer(complete_sentence.group(1), category, prompt)
    if len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", answer)) < 3:
        return None
    return answer


def _reasoning_effort(model: str, category: str, prompt: str) -> str | None:
    model_name = canonical_model_name(model)
    if model_name == "minimax-m3":
        if category == "logic":
            complex_logic = (
                "logic puzzle",
                "exactly",
                "different",
                "unless",
                "either",
                "if and only if",
            )
            if any(signal in prompt.lower() for signal in complex_logic):
                return "low"
            return "none"
        if category in {"math", "code_debugging", "code_generation"}:
            return "low"
        return "none"
    if model_name == "kimi-k2p7-code":
        if category in {"code_debugging", "code_generation"}:
            hard_signals = (
                "dynamic programming",
                "concurrent",
                "thread-safe",
                "asynchronous",
                "graph",
                "parser",
            )
            if len(prompt.split()) > 160 or any(signal in prompt.lower() for signal in hard_signals):
                return "low"
        return "none"
    return None


def clean_model_answer(content: str, category: str, prompt: str = "") -> str:
    answer = content.strip()
    if category in {"code_generation", "code_debugging"}:
        fenced = re.fullmatch(
            r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?(.*?)\n?```",
            answer,
            flags=re.S,
        )
        if fenced:
            answer = fenced.group(1).strip()
    if category == "ner":
        fenced_json = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", answer, flags=re.I | re.S)
        if fenced_json:
            answer = fenced_json.group(1).strip()
        answer = re.sub(r"\bGPE\b", "LOCATION", answer, flags=re.I)
        instruction = prompt.split(":", maxsplit=1)[0]
        if re.search(r"\bORGANIZATIONS?\b|\bORGANISATIONS?\b", instruction, re.I):
            answer = re.sub(r"\bORG\b", "ORGANIZATION", answer)
    if category == "math" and not re.search(
        r"\b(explain|steps?|reasoning|derive)\b|\bshow\s+(?:your\s+)?work\b",
        prompt,
        re.I,
    ):
        first_line = answer.splitlines()[0].strip().strip("*_` ")
        labelled = re.match(
            r"^(?:the\s+)?(?:final\s+)?(?:answer|price|speed|value|result|total)\s*"
            r"(?:is|:)\s*(.+)$",
            first_line,
            re.I,
        )
        if labelled:
            first_line = labelled.group(1).strip().strip("*_` ")
        if re.match(r"^[+-]?(?:\d|\.\d)", first_line):
            return first_line.rstrip(".")
    if category == "sentiment" and not re.search(
        r"\b(justify|explain|reason(?:ing)?|why)\b", prompt, re.I
    ):
        label = re.search(
            r"\b(positive|negative|neutral|mixed|favorable|unfavorable|"
            r"favourable|unfavourable)\b",
            answer,
            re.I,
        )
        if label:
            return label.group(1).lower()
    if category == "sentiment" and re.search(
        r"\b(?:justify|explain|reason(?:ing)?|why)\b", prompt, re.I
    ):
        joined = re.sub(r"\s*\n+\s*", " ", answer).strip()
        labelled_reason = re.match(
            r"^\s*[*_`]*(positive|negative|neutral|mixed|favorable|unfavorable|"
            r"favourable|unfavourable)[*_`]*\s*[.:]\s*(.+)$",
            joined,
            re.I | re.S,
        )
        if labelled_reason:
            label = labelled_reason.group(1).capitalize()
            reason = labelled_reason.group(2).strip()
            if reason:
                reason = reason[0].lower() + reason[1:]
            answer = f"{label} — {reason}"
    if category == "logic" and not re.search(r"\b(explain|show|justify|why)\b", prompt, re.I):
        yes_no = re.match(r"^[^A-Za-z]*(yes|no)\b", answer, re.I)
        if yes_no:
            return yes_no.group(1).capitalize()
    return answer


def validate_model_answer(answer: str, category: str, prompt: str) -> None:
    if category == "code_generation" and _expects_python(prompt, answer):
        try:
            tree = ast.parse(answer)
        except SyntaxError as exc:
            raise FireworksError(f"model returned invalid Python: {exc.msg}", status_code=502) from exc
        _validate_requested_python_names(tree, prompt)

    if category == "code_generation" and re.search(r"\bsql\b", prompt, re.I):
        if not re.match(r"\s*(?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER)\b", answer, re.I):
            raise FireworksError("model returned invalid SQL-shaped output", status_code=502)

    if category == "code_debugging":
        if _requires_bug_diagnosis(prompt) and not _has_bug_diagnosis(answer):
            raise FireworksError(
                "model omitted the requested bug diagnosis",
                status_code=502,
            )
        python_candidate = _debug_python_candidate(prompt, answer)
        if _requires_bug_fix(prompt) and _contains_python_source(prompt) and python_candidate is None:
            raise FireworksError(
                "model omitted the requested Python fix",
                status_code=502,
            )
        if python_candidate is not None:
            try:
                tree = ast.parse(python_candidate)
            except SyntaxError as exc:
                raise FireworksError(
                    f"model returned invalid Python fix: {exc.msg}",
                    status_code=502,
                ) from exc
            _validate_requested_python_names(tree, prompt)

    if category == "sentiment":
        instruction = prompt.split(":", maxsplit=1)[0]
        allowed_labels = {
            label.lower()
            for label in re.findall(
                r"\b(?:positive|negative|neutral|mixed|favorable|unfavorable|"
                r"favourable|unfavourable)\b",
                instruction,
                re.I,
            )
        }
        returned_label = re.fullmatch(
            r"\s*(positive|negative|neutral|mixed|favorable|unfavorable|"
            r"favourable|unfavourable)\s*[.!]?\s*",
            answer,
            re.I,
        )
        if allowed_labels and returned_label and returned_label.group(1).lower() not in allowed_labels:
            raise FireworksError("model returned a sentiment label outside the allowed set", status_code=502)
        if re.search(r"\b(?:reason|reasoning|justify|explain)\b", instruction, re.I):
            if len(re.findall(r"\b[\w'-]+\b", answer)) < 8:
                raise FireworksError("model omitted the requested sentiment reason", status_code=502)
            if re.search(r"\b(?:exactly\s+)?one[- ]sentence\b", instruction, re.I):
                if _sentence_count(answer) != 1:
                    raise FireworksError("model violated one-sentence sentiment format", status_code=502)

    if category == "summarization":
        bullet_request = re.search(
            r"\b(?P<count>[1-5]|one|two|three|four|five)\s+bullet\s+points?\b",
            prompt,
            re.I,
        )
        if bullet_request is not None:
            expected = _small_number(bullet_request.group("count"))
            bullets = [
                line
                for line in answer.splitlines()
                if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)
            ]
            if len(bullets) != expected:
                raise FireworksError(
                    f"model returned {len(bullets)} bullets; expected {expected}",
                    status_code=502,
                )
            per_bullet_limit = re.search(
                r"\beach\s+(?:bullet(?:\s+point)?\s+)?(?:no\s+longer\s+than|"
                r"no\s+more\s+than|at\s+most)\s+(\d+)\s+words?\b",
                prompt,
                re.I,
            )
            if per_bullet_limit is not None:
                limit = int(per_bullet_limit.group(1))
                for bullet in bullets:
                    content = re.sub(r"^\s*(?:[-*â€¢]|\d+[.)])\s+", "", bullet)
                    if len(re.findall(r"\b[\w'-]+\b", content)) > limit:
                        raise FireworksError(
                            f"model exceeded {limit}-word per-bullet limit",
                            status_code=502,
                        )
        word_limit = re.search(
            r"\b(?:no\s+more\s+than|at\s+most|maximum\s+of)\s+(\d+)\s+words?\b",
            prompt,
            re.I,
        )
        if word_limit is not None:
            words = re.findall(r"\b[\w'-]+\b", answer)
            if len(words) > int(word_limit.group(1)):
                raise FireworksError(
                    f"model exceeded {word_limit.group(1)}-word limit",
                    status_code=502,
                )
        exact_words = re.search(r"\bexactly\s+(\d+)\s+words?\b", prompt, re.I)
        if exact_words is not None:
            words = re.findall(r"\b[\w'-]+\b", answer)
            if len(words) != int(exact_words.group(1)):
                raise FireworksError(
                    f"model returned {len(words)} words; expected exactly {exact_words.group(1)}",
                    status_code=502,
                )
        sentence_request = re.search(
            r"\b(?:in\s+)?(?:exactly\s+)?(?P<count>[1-5]|one|two|three|four|five)\s+sentences?\b",
            prompt,
            re.I,
        )
        if sentence_request is not None:
            expected = _small_number(sentence_request.group("count"))
            actual = _sentence_count(answer)
            if actual != expected:
                raise FireworksError(
                    f"model returned {actual} sentences; expected {expected}",
                    status_code=502,
                )

    if category == "ner" and re.search(r"\bjson\b", prompt, re.I):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise FireworksError("model returned invalid NER JSON", status_code=502) from exc
        if not isinstance(parsed, (list, dict)):
            raise FireworksError("model returned unsupported NER JSON", status_code=502)


def _expects_python(prompt: str, answer: str) -> bool:
    if re.search(
        r"\b(?:explain|explanation|identify|describe|justify|reason|why)\b|"
        r"\bwhat\s+is\s+wrong\b",
        prompt,
        re.I,
    ):
        return False
    if re.search(r"\bsql\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+\b", prompt, re.I):
        return False
    return bool(
        re.search(r"\bpython\b|```python|\bdef\s+[A-Za-z_]", prompt, re.I)
        or re.match(r"\s*(?:from\s+\S+\s+import|import\s+\S+|async\s+def|def|class)\b", answer)
    )


def _validate_requested_python_names(tree: ast.AST, prompt: str) -> None:
    requested = {
        match.group(1)
        for match in re.finditer(
            r"\b(?:function|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            prompt,
            re.I,
        )
    }
    if not requested:
        return
    returned = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = requested - returned
    if missing:
        raise FireworksError(
            "model omitted requested Python function: " + ", ".join(sorted(missing)),
            status_code=502,
        )


def _sentence_count(text: str) -> int:
    normalized = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|e\.g|i\.e)\.",
        lambda match: match.group(0).replace(".", ""),
        text,
        flags=re.I,
    )
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", normalized.strip()) if part.strip()]
    return len(sentences) if sentences else int(bool(normalized.strip()))


def _requires_bug_diagnosis(prompt: str) -> bool:
    if re.search(
        r"\b(?:only|just)\s+(?:return|output|provide|show)\b.{0,30}\b(?:code|fix)\b|"
        r"\b(?:code|fix)\s+only\b",
        prompt,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:debug|bug|error|issue|wrong|incorrect|broken|fails?|"
            r"find\s+and\s+fix|identify|explain|why|what\s+is\s+wrong)\b",
            prompt,
            re.I,
        )
    )


def _requires_bug_fix(prompt: str) -> bool:
    return bool(re.search(r"\b(?:fix|repair|correct|corrected|rewrite)\b", prompt, re.I))


def _contains_python_source(prompt: str) -> bool:
    if re.search(r"\b(?:sql|javascript|typescript|java|c\+\+|rust)\b", prompt, re.I):
        return False
    return bool(
        re.search(
            r"```(?:python|py)?\s*\n|\b(?:async\s+def|def|class)\s+[A-Za-z_]\w*\b",
            prompt,
            re.I,
        )
    )


def _has_bug_diagnosis(answer: str) -> bool:
    try:
        ast.parse(answer)
        return False
    except SyntaxError:
        pass
    prose = re.sub(
        r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?.*?\n?```",
        " ",
        answer,
        flags=re.S,
    )
    prose = re.sub(
        r"(?m)^\s*(?:corrected|fixed|updated)\s+(?:code|implementation)\s*:\s*$",
        " ",
        prose,
        flags=re.I,
    )
    prose = re.sub(
        r"(?ms)^\s*(?:from\s+\S+\s+import|import\s+\S+|async\s+def|def|class)\b.*\Z",
        " ",
        prose,
    )
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", prose)) >= 4


def _debug_python_candidate(prompt: str, answer: str) -> str | None:
    if re.search(r"\bsql\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+\b", prompt, re.I):
        return None
    fenced = re.search(
        r"```(?:python|py)?\s*\n?(.*?)\n?```",
        answer,
        flags=re.I | re.S,
    )
    if fenced:
        return fenced.group(1).strip()
    try:
        ast.parse(answer)
        return answer
    except SyntaxError:
        pass
    code_start = re.search(
        r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+|async\s+def|def|class)\b",
        answer,
    )
    if code_start:
        return answer[code_start.start():].strip()
    return None


def _small_number(raw: str) -> int:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return int(raw) if raw.isdigit() else words[raw.lower()]


def _should_try_next_model(exc: FireworksError) -> bool:
    # Network/timeout/response-decoding failures have no HTTP status and are transient. Trying
    # another allowed model is safer than turning an otherwise answerable task into an empty row.
    return exc.status_code is None or exc.status_code in {400, 404, 408, 429, 500, 502, 503, 504}
