"""Railway entrypoint — serves index.html at / and POST /api/solve."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Make sure project modules are importable
sys.path.insert(0, os.path.dirname(__file__))

if not os.environ.get("ALLOWED_MODELS"):
    os.environ.setdefault(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4,gemma-4-31b-it",
    )

from app.agent import solve_prompt          # noqa: E402
from router.classify import classify_prompt  # noqa: E402

INDEX_HTML = Path(__file__).parent / "index.html"


class RequestHandler(BaseHTTPRequestHandler):

    # ------------------------------------------------------------------
    # GET  /          → serve index.html
    # GET  /anything  → 404
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", ""):
            body = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------------
    # OPTIONS — CORS preflight
    # ------------------------------------------------------------------
    def do_OPTIONS(self) -> None:
        self._cors_headers(200)
        self.end_headers()

    # ------------------------------------------------------------------
    # POST /api/solve
    # ------------------------------------------------------------------
    def do_POST(self) -> None:
        if self.path != "/api/solve":
            self._json({"error": "not found"}, 404)
            return

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
                "tokens_used": result.tokens_used,
            })
        except Exception as exc:  # noqa: BLE001
            self._json({"error": str(exc)}, 500)

    # ------------------------------------------------------------------

    def _json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self._cors_headers(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _cors_headers(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"BudgetBrain server starting on port {port}", flush=True)
    HTTPServer(("0.0.0.0", port), RequestHandler).serve_forever()
