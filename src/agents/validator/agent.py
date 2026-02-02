"""Validator Agent: Self-RAG verification with reflection tokens."""
from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from src.config.constants import ReflectionToken
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALIDATION_SYSTEM_PROMPT = """당신은 AI 답변 검증 전문가입니다.
생성된 답변이 검색된 증거로 뒷받침되는지 검증하세요.

Self-RAG Reflection Tokens을 사용하여 단계별로 검증합니다:
- [Retrieve]: 추가 검색이 필요한가?
- [IsREL]: 검색된 문서가 질문과 관련이 있는가?
- [IsSUP]: 답변이 증거로 뒷받침되는가?
- [IsUSE]: 전체적으로 유용한 답변인가?"""

VALIDATION_PROMPT = """질문: {query}

생성된 답변: {answer}

사용된 증거:
{evidence}

다음 JSON 형식으로 검증 결과를 응답하세요:
{{
    "retrieve_needed": true/false,
    "retrieve_reason": "추가 검색 필요 이유 또는 불필요 이유",
    "relevance": {{
        "score": 0.0-1.0,
        "irrelevant_sources": ["관련 없는 소스 ID"],
        "explanation": "관련성 설명"
    }},
    "support": {{
        "level": "fully|partially|no",
        "supported_claims": ["뒷받침되는 주장"],
        "unsupported_claims": ["뒷받침되지 않는 주장"],
        "explanation": "뒷받침 설명"
    }},
    "usefulness": {{
        "score": 1-5,
        "explanation": "유용성 설명"
    }},
    "overall_confidence": 0.0-1.0,
    "hallucination_flags": ["환각으로 의심되는 부분"],
    "improvement_suggestions": ["개선 제안"]
}}

JSON만 응답하세요."""


class ValidatorAgent(BaseAgent):
    """Self-RAG 기반 답변 검증 에이전트.

    Reflection Tokens:
    - [Retrieve]: 추가 검색 필요 여부
    - [IsREL]: 문서 관련성
    - [IsSUP]: 증거 뒷받침
    - [IsUSE]: 전체 유용성
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="validator")
        self._llm = llm_client
        self.max_retries = 2

    async def execute(self, context: AgentContext) -> AgentResult:
        """답변 검증 수행."""
        query = context.query
        answer = context.previous_results.get("answer", "")
        documents = context.previous_results.get("documents", [])
        confidence = context.previous_results.get("confidence", 0.5)

        evidence = self._format_evidence(documents[:5])

        prompt = VALIDATION_PROMPT.format(
            query=query,
            answer=answer,
            evidence=evidence or "증거 없음",
        )

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=VALIDATION_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=2000,
        )

        try:
            validation = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                validation = json.loads(content[start:end])
            else:
                validation = self._default_validation()

        # Calculate final confidence
        overall_confidence = validation.get("overall_confidence", confidence)
        support_level = validation.get("support", {}).get("level", "partially")
        usefulness_score = validation.get("usefulness", {}).get("score", 3)
        hallucination_flags = validation.get("hallucination_flags", [])

        # Adjust confidence based on validation
        adjusted_confidence = self._adjust_confidence(
            overall_confidence, support_level, usefulness_score, len(hallucination_flags)
        )

        # Determine if re-retrieval or re-generation is needed
        needs_retry = (
            validation.get("retrieve_needed", False)
            or support_level == "no"
            or adjusted_confidence < 0.3
        )

        warnings: list[str] = []
        if hallucination_flags:
            warnings.append(f"환각 의심: {', '.join(hallucination_flags)}")
        if adjusted_confidence < 0.5:
            warnings.append(f"낮은 신뢰도: {adjusted_confidence:.2f}")
        if support_level == "partially":
            unsupported = validation.get("support", {}).get("unsupported_claims", [])
            if unsupported:
                warnings.append(f"미뒷받침 주장: {', '.join(unsupported[:3])}")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "is_valid": not needs_retry,
                "adjusted_confidence": adjusted_confidence,
                "support_level": support_level,
                "usefulness_score": usefulness_score,
                "hallucination_flags": hallucination_flags,
                "warnings": warnings,
                "needs_retry": needs_retry,
                "validation_detail": validation,
                "improvement_suggestions": validation.get("improvement_suggestions", []),
            },
            metadata={
                "tokens_used": response.total_tokens,
                "cost_usd": response.estimated_cost_usd,
            },
        )

    @staticmethod
    def _adjust_confidence(
        base_confidence: float,
        support_level: str,
        usefulness_score: int,
        hallucination_count: int,
    ) -> float:
        """검증 결과를 바탕으로 신뢰도 조정."""
        adjusted = base_confidence

        # Support level adjustment
        if support_level == "fully":
            adjusted *= 1.1
        elif support_level == "partially":
            adjusted *= 0.8
        elif support_level == "no":
            adjusted *= 0.3

        # Usefulness adjustment
        adjusted *= (usefulness_score / 5.0)

        # Hallucination penalty
        adjusted -= hallucination_count * 0.15

        return max(0.0, min(1.0, adjusted))

    @staticmethod
    def _format_evidence(docs: list[dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[증거-{i}] {doc.get('content', '')[:300]}")
        return "\n\n".join(parts)

    @staticmethod
    def _default_validation() -> dict[str, Any]:
        return {
            "retrieve_needed": False,
            "relevance": {"score": 0.5, "explanation": "검증 불가"},
            "support": {"level": "partially", "explanation": "검증 불가"},
            "usefulness": {"score": 3, "explanation": "검증 불가"},
            "overall_confidence": 0.5,
            "hallucination_flags": [],
            "improvement_suggestions": [],
        }
