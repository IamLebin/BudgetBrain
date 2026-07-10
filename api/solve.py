"""Vercel serverless function — POST /api/solve"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# Add project root to path so router/, solvers/, fireworks/, app/ are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set default ALLOWED_MODELS if not injected by the environment
if not os.environ.get("ALLOWED_MODELS"):
    os.environ.setdefault(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4,gemma-4-31b-it",
    )

from app.agent import solve_prompt          # noqa: E402
from router.classify import classify_prompt  # noqa: E402


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self) -> None:
        self._cors_headers(200)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = (body.get("prompt") or "").strip()
        except Exception:
            self._json({"error": "invalid request body"}, 400)
            return

        if not prompt:
            self._json({"error": "prompt is required"}, 400)
            return

        try:
            clf = classify_prompt(prompt)
            result = solve_prompt(prompt)
            is_local = result.source.startswith("local:")
            method = result.source.replace("local:", "") if is_local else None
            self._json({
                "answer": result.answer.strip(),
                "category": result.category,
                "source": result.source,
                "is_local": is_local,
                "method": method,
                "confidence": clf.confidence,
            })
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)

    # ------------------------------------------------------------------

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self._cors_headers(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress default access log noise
