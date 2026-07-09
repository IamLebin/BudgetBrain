from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalAnswer:
    answer: str
    confidence: float
    method: str
