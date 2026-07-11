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
    "math": ("minimax-m3", "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4"),
    "sentiment": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "summarization": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "ner": ("kimi-k2p7-code", "minimax-m3", "gemma-4-26b-a4b-it"),
    "code_debugging": ("kimi-k2p7-code", "minimax-m3", "gemma-4-31b-it-nvfp4"),
    "logic": ("minimax-m3", "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4"),
    "code_generation": ("kimi-k2p7-code", "minimax-m3", "gemma-4-31b-it-nvfp4"),
}

MAX_TOKENS = {
    "factual_qa": 96,
    "math": 64,
    "sentiment": 32,
    "summarization": 160,
    "ner": 160,
    "code_debugging": 240,
    "logic": 96,
    "code_generation": 320,
}

SYSTEM_PROMPTS = {
    "factual_qa": "Answer every part in one concise sentence using standard key terms.",
    "math": "Return only the final answer unless steps are requested.",
    "sentiment": "Return only the requested label unless a reason is requested.",
    "summarization": "Follow all constraints; output only the summary.",
    "ner": "Return all entities in the requested format.",
    "code_debugging": "Return only corrected code unless an explanation is requested.",
    "logic": "Return only the requested conclusion unless reasoning is requested.",
    "code_generation": "Return only minimal correct code; prefer built-ins when possible.",
}


class FireworksError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FireworksClient:
    def __init__(self, api_key: str, base_url: str, allowed_models: list[str]) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.allowed_models = allowed_models

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
        errors: list[str] = []
        for model in self.candidate_models(category):
            api_model = model_id_for_request(model, self.base_url)
            payload = {
                "model": api_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_qa"])},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": MAX_TOKENS.get(category, 128),
            }
            reasoning_effort = _reasoning_effort(model, category, prompt)
            if reasoning_effort is not None:
                payload["reasoning_effort"] = reasoning_effort
            try:
                response = self._post_json("/chat/completions", payload)
                self._log_usage(category, api_model, response.get("usage"))
                content = _response_content(response)
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
    if category == "ner" and re.search(r"\bjson\b", prompt, re.I):
        fenced_json = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", answer, flags=re.I | re.S)
        if fenced_json:
            answer = fenced_json.group(1).strip()
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
        r"\b(justify|explain|reason|why)\b", prompt, re.I
    ):
        label = re.search(r"\b(positive|negative|neutral|mixed)\b", answer, re.I)
        if label:
            return label.group(1).lower()
    if category == "logic" and not re.search(r"\b(explain|show|justify|why)\b", prompt, re.I):
        yes_no = re.match(r"^[^A-Za-z]*(yes|no)\b", answer, re.I)
        if yes_no:
            return yes_no.group(1).capitalize()
    return answer


def validate_model_answer(answer: str, category: str, prompt: str) -> None:
    if category in {"code_generation", "code_debugging"} and _expects_python(prompt, answer):
        try:
            ast.parse(answer)
        except SyntaxError as exc:
            raise FireworksError(f"model returned invalid Python: {exc.msg}", status_code=502) from exc

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

    if category == "ner" and re.search(r"\bjson\b", prompt, re.I):
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise FireworksError("model returned invalid NER JSON", status_code=502) from exc
        if not isinstance(parsed, (list, dict)):
            raise FireworksError("model returned unsupported NER JSON", status_code=502)


def _expects_python(prompt: str, answer: str) -> bool:
    if re.search(r"\b(?:explain|explanation|justify|reason|why)\b", prompt, re.I):
        return False
    if re.search(r"\bsql\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+\b", prompt, re.I):
        return False
    return bool(
        re.search(r"\bpython\b|```python|\bdef\s+[A-Za-z_]", prompt, re.I)
        or re.match(r"\s*(?:from\s+\S+\s+import|import\s+\S+|async\s+def|def|class)\b", answer)
    )


def _small_number(raw: str) -> int:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    return int(raw) if raw.isdigit() else words[raw.lower()]


def _should_try_next_model(exc: FireworksError) -> bool:
    return exc.status_code in {400, 404, 408, 429, 500, 502, 503, 504}
