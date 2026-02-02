"""Reasoner Agent: Chain-of-Thought reasoning with graph integration."""
from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

REASONING_SYSTEM_PROMPT = """당신은 시계열 데이터 분석 전문가입니다.
검색된 증거를 바탕으로 단계별로 추론하여 정확한 답변을 생성하세요.

추론 규칙:
1. 각 단계를 명확히 구분하여 설명
2. 증거가 부족한 경우 명시적으로 표시
3. 인과관계와 상관관계를 구분
4. 시간적 선후관계를 고려
5. 신뢰도 점수를 0.0-1.0 사이로 산정"""

REASONING_PROMPT = """질문: {query}

검색된 증거:
{evidence}

그래프 경로 (인과관계):
{graph_paths}

시계열 데이터:
{timeseries_data}

위 증거를 바탕으로 단계별(Chain-of-Thought) 추론을 수행하세요.

다음 JSON 형식으로 응답하세요:
{{
    "reasoning_steps": [
        {{"step": 1, "action": "분석/추론/검증", "description": "설명", "evidence_used": ["증거 ID"]}}
    ],
    "answer": "최종 답변",
    "confidence": 0.0-1.0,
    "causal_chain": ["원인1 → 결과1", "결과1 → 결과2"],
    "uncertainties": ["불확실한 부분"]
}}

JSON만 응답하세요."""


class ReasonerAgent(BaseAgent):
    """검색된 정보를 바탕으로 Chain-of-Thought 추론을 수행하는 에이전트."""

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="reasoner")
        self._llm = llm_client

    async def execute(self, context: AgentContext) -> AgentResult:
        """CoT 추론 수행."""
        query = context.query
        documents = context.previous_results.get("documents", [])

        # Separate evidence by source type
        vector_docs = [d for d in documents if d.get("source_type") == "vector"]
        graph_docs = [d for d in documents if d.get("source_type") == "graph"]
        ts_docs = [d for d in documents if d.get("source_type") == "timeseries"]

        evidence = self._format_evidence(vector_docs[:5])
        graph_paths = self._format_graph_paths(graph_docs[:5])
        timeseries_data = self._format_timeseries(ts_docs[:10])

        prompt = REASONING_PROMPT.format(
            query=query,
            evidence=evidence or "검색된 문서 없음",
            graph_paths=graph_paths or "그래프 경로 없음",
            timeseries_data=timeseries_data or "시계열 데이터 없음",
        )

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=REASONING_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096,
        )

        try:
            reasoning = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                reasoning = json.loads(content[start:end])
            else:
                reasoning = {
                    "reasoning_steps": [],
                    "answer": response.content,
                    "confidence": 0.5,
                    "causal_chain": [],
                    "uncertainties": ["LLM 응답 파싱 실패"],
                }

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "answer": reasoning.get("answer", ""),
                "confidence": reasoning.get("confidence", 0.5),
                "reasoning_steps": reasoning.get("reasoning_steps", []),
                "causal_chain": reasoning.get("causal_chain", []),
                "uncertainties": reasoning.get("uncertainties", []),
                "sources_used": len(documents),
            },
            metadata={
                "tokens_used": response.total_tokens,
                "cost_usd": response.estimated_cost_usd,
                "llm_elapsed_ms": response.elapsed_ms,
            },
        )

    @staticmethod
    def _format_evidence(docs: list[dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] (score: {doc.get('relevance_score', 0):.2f}) {doc.get('content', '')[:500]}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_graph_paths(docs: list[dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"경로 {i}: {doc.get('content', '')}")
        return "\n".join(parts)

    @staticmethod
    def _format_timeseries(docs: list[dict[str, Any]]) -> str:
        parts = []
        for doc in docs[:10]:
            parts.append(doc.get("content", ""))
        return "\n".join(parts)
