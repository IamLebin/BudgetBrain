from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODELS = (
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
    "gemma-4-31b-it",
    "minimax-m3",
    "kimi-k2p7-code",
)

MODEL_PREFERENCE = {
    "factual_qa": ("gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "minimax-m3"),
    "math": ("gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "minimax-m3"),
    "sentiment": ("gemma-4-26b-a4b-it", "minimax-m3"),
    "summarization": ("gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "minimax-m3"),
    "ner": ("gemma-4-26b-a4b-it", "minimax-m3"),
    "code_debugging": ("kimi-k2p7-code", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it"),
    "logic": ("gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4", "minimax-m3"),
    "code_generation": ("kimi-k2p7-code", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it"),
}

MAX_TOKENS = {
    "factual_qa": 96,
    "math": 64,
    "sentiment": 8,
    "summarization": 180,
    "ner": 160,
    "code_debugging": 220,
    "logic": 96,
    "code_generation": 520,
}

SYSTEM_PROMPTS = {
    "factual_qa": "Return only the direct answer.",
    "math": "Return only the final numeric answer.",
    "sentiment": "Classify as positive, negative, or neutral. Return 1 word.",
    "summarization": "Summarize briefly. Return only the summary.",
    "ner": "Return only requested entities in requested format.",
    "code_debugging": "Return only the fix or corrected code.",
    "logic": "Return only the final answer.",
    "code_generation": "Return only code. No explanations.",
}


class FireworksError(RuntimeError):
    pass


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
        return cls(api_key=api_key, base_url=base_url, allowed_models=allowed)

    def solve(self, prompt: str, category: str) -> str:
        model = self.pick_model(category)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual_qa"])},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": MAX_TOKENS.get(category, 128),
        }
        response = self._post_json("/chat/completions", payload)
        self._log_usage(category, model, response.get("usage"))
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise FireworksError(f"unexpected Fireworks response shape: {exc}") from exc

    def pick_model(self, category: str) -> str:
        preferred = MODEL_PREFERENCE.get(category, DEFAULT_MODELS)
        for model in preferred:
            if model in self.allowed_models:
                return model
        if self.allowed_models:
            return self.allowed_models[0]
        raise FireworksError("ALLOWED_MODELS is empty")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FireworksError(f"HTTP {exc.code}: {detail[:400]}") from exc
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
    return models or list(DEFAULT_MODELS)
