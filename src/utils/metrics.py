from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryMetrics:
    query_id: str
    start_time: float = field(default_factory=time.time)
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_step(self, step_name: str, duration_ms: float, metadata: dict[str, Any] | None = None) -> None:
        self.steps.append({
            "step": step_name,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {},
        })

    def elapsed_ms(self) -> float:
        return round((time.time() - self.start_time) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "total_duration_ms": self.elapsed_ms(),
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }
