"""Self-RAG Verifier: automated answer verification with reflection tokens."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.config.constants import ReflectionToken
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

VERIFY_PROMPT = """당신은 Self-RAG 검증 시스템입니다. 다음 단계를 순서대로 수행하세요.

[질문]: {query}
[생성된 답변]: {answer}
[사용된 증거]:
{evidence}

각 Reflection Token에 대해 판단하세요:

1. [Retrieve] 추가 검색 필요 여부: Yes/No + 이유
2. [IsREL] 증거-질문 관련성: Relevant/Irrelevant (각 증거별)
3. [IsSUP] 증거 뒷받침 수준: Fully/Partially/No
4. [IsUSE] 전체 유용성 점수: 1-5

JSON 형식으로 응답:
{{
    "retrieve": {{"decision": "Yes|No", "reason": "이유"}},
    "is_relevant": [{{"source_idx": 1, "decision": "Relevant|Irrelevant", "reason": "이유"}}],
    "is_supported": {{"level": "Fully|Partially|No", "supported_parts": ["뒷받침 부분"], "unsupported_parts": ["미뒷받침 부분"]}},
    "is_useful": {{"score": 1-5, "reason": "이유"}},
    "final_verdict": "pass|fail|needs_retry",
    "adjusted_answer": "수정된 답변 (필요시)"
}}

JSON만 응답하세요."""


@dataclass
class VerificationResult:
    """검증 결과."""

    passed: bool
    final_answer: str
    confidence: float
    retrieve_needed: bool = False
    relevance_scores: list[dict[str, Any]] = field(default_factory=list)
    support_level: str = "partially"
    usefulness_score: int = 3
    iteration: int = 0
    warnings: list[str] = field(default_factory=list)


class SelfRAGVerifier:
    """Self-RAG 검증기: Reflection Tokens을 사용한 자동 답변 검증.

    Workflow:
    Query → [Retrieve Decision] → Retrieval → [IsREL Check]
                                                  │
            Irrelevant: Re-retrieve ◄─────────────┤
                                                  │ Relevant
                                                  ▼
             Generate → [IsSUP Verify] → [IsUSE Score] → Final Output
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self.max_iterations = 3

    async def verify(
        self,
        query: str,
        answer: str,
        evidence: list[dict[str, Any]],
        current_confidence: float = 0.5,
    ) -> VerificationResult:
        """답변 검증 수행.

        Args:
            query: 원본 질의
            answer: 생성된 답변
            evidence: 사용된 증거 리스트
            current_confidence: 현재 신뢰도

        Returns:
            VerificationResult with adjusted confidence and warnings
        """
        formatted_evidence = self._format_evidence(evidence)

        prompt = VERIFY_PROMPT.format(
            query=query,
            answer=answer,
            evidence=formatted_evidence or "증거 없음",
        )

        response = await self._llm.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000,
        )

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                return VerificationResult(
                    passed=True,
                    final_answer=answer,
                    confidence=current_confidence * 0.8,
                    warnings=["검증 결과 파싱 실패"],
                )

        # Extract verification details
        retrieve_decision = result.get("retrieve", {})
        relevance_checks = result.get("is_relevant", [])
        support_check = result.get("is_supported", {})
        usefulness_check = result.get("is_useful", {})
        verdict = result.get("final_verdict", "pass")
        adjusted_answer = result.get("adjusted_answer")

        # Calculate adjusted confidence
        support_level = support_check.get("level", "partially")
        usefulness_score = usefulness_check.get("score", 3)
        relevant_count = sum(1 for r in relevance_checks if r.get("decision") == "Relevant")
        total_sources = max(len(relevance_checks), 1)

        relevance_ratio = relevant_count / total_sources
        support_multiplier = {"Fully": 1.0, "Partially": 0.7, "No": 0.2}.get(support_level, 0.5)
        usefulness_multiplier = usefulness_score / 5.0

        adjusted_confidence = current_confidence * relevance_ratio * support_multiplier * usefulness_multiplier
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        # Build warnings
        warnings: list[str] = []
        if retrieve_decision.get("decision") == "Yes":
            warnings.append(f"추가 검색 필요: {retrieve_decision.get('reason', '')}")
        if support_level == "No":
            warnings.append("답변이 증거로 뒷받침되지 않음")
        unsupported = support_check.get("unsupported_parts", [])
        if unsupported:
            warnings.append(f"미뒷받침 부분: {', '.join(unsupported[:3])}")

        passed = verdict == "pass" and adjusted_confidence >= 0.3
        final_answer = adjusted_answer if adjusted_answer and verdict != "pass" else answer

        logger.info(
            "selfrag_verification",
            verdict=verdict,
            support_level=support_level,
            usefulness=usefulness_score,
            original_confidence=current_confidence,
            adjusted_confidence=adjusted_confidence,
        )

        return VerificationResult(
            passed=passed,
            final_answer=final_answer,
            confidence=adjusted_confidence,
            retrieve_needed=retrieve_decision.get("decision") == "Yes",
            relevance_scores=relevance_checks,
            support_level=support_level,
            usefulness_score=usefulness_score,
            warnings=warnings,
        )

    @staticmethod
    def _format_evidence(evidence: list[dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(evidence[:10], 1):
            content = doc.get("content", "")[:500]
            source = doc.get("source", "unknown")
            parts.append(f"[증거-{i}] (소스: {source}) {content}")
        return "\n\n".join(parts)
