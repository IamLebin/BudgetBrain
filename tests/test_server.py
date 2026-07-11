from __future__ import annotations

import io
import json
import types
import unittest
from unittest.mock import patch

from app.agent import SolveResult
import server


class ServerApiTests(unittest.TestCase):
    def _post(self, result: SolveResult) -> dict[str, object]:
        handler = object.__new__(server.RequestHandler)
        body = json.dumps({"prompt": "What is the capital of Japan?"}).encode()
        handler.path = "/api/solve"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        captured: dict[str, object] = {}
        handler._json = types.MethodType(  # type: ignore[method-assign]
            lambda self, data, status=200: captured.update(data),
            handler,
        )
        with (
            patch.object(server, "solve_prompt", return_value=result),
            patch.object(server, "classify_prompt", return_value=types.SimpleNamespace(confidence=0.55)),
        ):
            handler.do_POST()
        return captured

    def test_api_exposes_fireworks_token_usage(self) -> None:
        payload = self._post(
            SolveResult(
                answer="Tokyo",
                category="factual_qa",
                source="fireworks",
                tokens_used=17,
            )
        )
        self.assertEqual(payload["tokens_used"], 17)

    def test_api_reports_zero_for_local_solver(self) -> None:
        payload = self._post(
            SolveResult(
                answer="42",
                category="math",
                source="local:safe_eval",
                tokens_used=0,
            )
        )
        self.assertEqual(payload["tokens_used"], 0)


if __name__ == "__main__":
    unittest.main()
