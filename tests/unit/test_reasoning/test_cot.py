"""Tests for Chain-of-Thought Processor."""
from __future__ import annotations

import pytest

from src.reasoning.cot.processor import CoTProcessor, ReasoningChain


class TestCoTProcessor:
    def test_start_chain(self) -> None:
        processor = CoTProcessor()
        chain = processor.start_chain("q1", "테스트 질문")

        assert chain.query == "테스트 질문"
        assert len(chain.steps) == 0

    def test_add_steps(self) -> None:
        processor = CoTProcessor()
        chain = processor.start_chain("q1", "질문")

        chain.add_step("retrieve", "문서 검색", evidence_used=["doc-1"])
        chain.add_step("reason", "추론 수행", confidence=0.85)

        assert len(chain.steps) == 2
        assert chain.steps[0].step_number == 1
        assert chain.steps[0].action == "retrieve"
        assert chain.steps[1].confidence == 0.85

    def test_finalize_chain(self) -> None:
        processor = CoTProcessor()
        processor.start_chain("q1", "질문")
        chain = processor.finalize_chain("q1", "최종 답변", 0.9)

        assert chain is not None
        assert chain.final_answer == "최종 답변"
        assert chain.overall_confidence == 0.9

    def test_to_dict(self) -> None:
        processor = CoTProcessor()
        chain = processor.start_chain("q1", "질문")
        chain.add_step("plan", "계획 수립")
        chain.add_step("retrieve", "검색")
        processor.finalize_chain("q1", "답변", 0.8)

        result = chain.to_dict()
        assert result["query"] == "질문"
        assert result["total_steps"] == 2
        assert result["final_answer"] == "답변"
        assert result["overall_confidence"] == 0.8
        assert result["total_duration_ms"] >= 0

    def test_get_nonexistent_chain(self) -> None:
        processor = CoTProcessor()
        assert processor.get_chain("nonexistent") is None
