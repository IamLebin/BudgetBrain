"""Railway entrypoint — serves index.html at / and POST /api/solve."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full, LifoQueue

# Make sure project modules are importable
sys.path.insert(0, os.path.dirname(__file__))

if not os.environ.get("ALLOWED_MODELS"):
    os.environ.setdefault(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4,gemma-4-31b-it",
    )

from app.agent import solve_prompt          # noqa: E402
from fireworks.client import FireworksClient, FireworksError  # noqa: E402
from router.classify import classify_prompt  # noqa: E402

INDEX_HTML = Path(__file__).parent / "index.html"
MAX_REQUEST_BODY_BYTES = 64 * 1024
DEMO_CLIENT_POOL_SIZE = 4
DEMO_CLIENT_POOL: LifoQueue[FireworksClient] = LifoQueue(maxsize=DEMO_CLIENT_POOL_SIZE)


def _acquire_demo_client() -> FireworksClient | None:
    try:
        return DEMO_CLIENT_POOL.get_nowait()
    except Empty:
        try:
            return FireworksClient.from_env()
        except FireworksError:
            return None


def _release_demo_client(client: FireworksClient | None) -> None:
    if client is None:
        return
    try:
        DEMO_CLIENT_POOL.put_nowait(client)
    except Full:
        pass


class RequestHandler(BaseHTTPRequestHandler):

    # ------------------------------------------------------------------
    # GET  /          → serve index.html
    # GET  /anything  → 404
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path in ("/", ""):
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
            if length <= 0:
                raise ValueError("missing request body")
            if length > MAX_REQUEST_BODY_BYTES:
                self._json({"error": "request body too large"}, 413)
                return
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            prompt = (body.get("prompt") or "").strip()
        except Exception:
            self._json({"error": "invalid request body"}, 400)
            return

        if not prompt:
            self._json({"error": "prompt is required"}, 400)
            return

        client = _acquire_demo_client()
        try:
            clf = classify_prompt(prompt)
            result = solve_prompt(prompt, client=client)
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
            print(f"demo solve failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._json({"error": "unable to solve prompt right now"}, 500)
        finally:
            _release_demo_client(client)

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
    ThreadingHTTPServer(("0.0.0.0", port), RequestHandler).serve_forever()
