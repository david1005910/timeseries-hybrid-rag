"""Chain-of-Thought Processor: step-by-step reasoning tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningStep:
    """단일 추론 단계."""

    step_number: int
    action: str  # analyze, retrieve, reason, verify
    description: str
    evidence_used: list[str] = field(default_factory=list)
    confidence: float | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """전체 추론 체인."""

    query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    overall_confidence: float = 0.0
    start_time: float = field(default_factory=time.time)

    def add_step(
        self,
        action: str,
        description: str,
        evidence_used: list[str] | None = None,
        confidence: float | None = None,
        duration_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ReasoningStep:
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            action=action,
            description=description,
            evidence_used=evidence_used or [],
            confidence=confidence,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def total_duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action,
                    "description": s.description,
                    "evidence_used": s.evidence_used,
                    "confidence": s.confidence,
                    "duration_ms": round(s.duration_ms, 2),
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "overall_confidence": self.overall_confidence,
            "total_duration_ms": round(self.total_duration_ms(), 2),
            "total_steps": len(self.steps),
        }


class CoTProcessor:
    """Chain-of-Thought 추론 추적 프로세서."""

    def __init__(self) -> None:
        self._chains: dict[str, ReasoningChain] = {}

    def start_chain(self, query_id: str, query: str) -> ReasoningChain:
        """새 추론 체인 시작."""
        chain = ReasoningChain(query=query)
        self._chains[query_id] = chain
        return chain

    def get_chain(self, query_id: str) -> ReasoningChain | None:
        return self._chains.get(query_id)

    def finalize_chain(
        self, query_id: str, answer: str, confidence: float
    ) -> ReasoningChain | None:
        """추론 체인 완료."""
        chain = self._chains.get(query_id)
        if chain:
            chain.final_answer = answer
            chain.overall_confidence = confidence
        return chain
