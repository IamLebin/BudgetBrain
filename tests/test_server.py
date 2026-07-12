from __future__ import annotations

import io
import json
import types
import unittest
from unittest.mock import patch

from app.agent import SolveResult
import server


class ServerApiTests(unittest.TestCase):
    def _handler(self, path: str = "/api/solve", body: bytes = b"") -> tuple[object, dict[str, object]]:
        handler = object.__new__(server.RequestHandler)
        handler.path = path
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        captured: dict[str, object] = {"status": None, "data": None}
        handler._json = types.MethodType(  # type: ignore[method-assign]
            lambda self, data, status=200: captured.update(status=status, data=data),
            handler,
        )
        return handler, captured

    def _post(
        self,
        result: SolveResult | None = None,
        *,
        error: Exception | None = None,
    ) -> tuple[int, dict[str, object]]:
        body = json.dumps({"prompt": "What is the capital of Japan?"}).encode()
        handler, captured = self._handler(body=body)
        solve_effect = error if error is not None else result
        with (
            patch.object(server, "solve_prompt", side_effect=solve_effect if error else None, return_value=result),
            patch.object(server, "classify_prompt", return_value=types.SimpleNamespace(confidence=0.55)),
            patch.object(server, "_acquire_demo_client", return_value=None),
        ):
            handler.do_POST()
        return captured["status"], captured["data"]  # type: ignore[return-value]

    def test_health_endpoint_returns_ok(self) -> None:
        handler, captured = self._handler(path="/health")
        handler.do_GET()
        self.assertEqual(captured, {"status": 200, "data": {"status": "ok"}})

    def test_oversized_request_is_rejected(self) -> None:
        handler, captured = self._handler(body=b"{}")
        handler.headers = {"Content-Length": str(server.MAX_REQUEST_BODY_BYTES + 1)}
        handler.do_POST()
        self.assertEqual(captured["status"], 413)
        self.assertEqual(captured["data"], {"error": "request body too large"})

    def test_solver_error_returns_stable_json(self) -> None:
        status, payload = self._post(error=RuntimeError("provider detail"))
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "unable to solve prompt right now"})

    def test_api_exposes_fireworks_token_usage(self) -> None:
        status, payload = self._post(
            SolveResult(
                answer="Tokyo",
                category="factual_qa",
                source="fireworks",
                tokens_used=17,
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens_used"], 17)

    def test_api_reports_zero_for_local_solver(self) -> None:
        status, payload = self._post(
            SolveResult(
                answer="42",
                category="math",
                source="local:safe_eval",
                tokens_used=0,
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["tokens_used"], 0)


if __name__ == "__main__":
    unittest.main()
