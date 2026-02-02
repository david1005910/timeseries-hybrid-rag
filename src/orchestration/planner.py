"""Planner Agent: query analysis and execution plan generation."""
from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """당신은 시계열 데이터 분석 시스템의 플래너입니다.
사용자 질의를 분석하여 최적의 실행 계획을 수립하세요.

가용 에이전트:
- retriever: 시계열, 벡터, 그래프 데이터 검색
- extractor: 텍스트에서 엔티티/관계 추출
- reasoner: Chain-of-Thought 추론
- validator: Self-RAG 검증

가용 데이터 소스:
- timeseries (InfluxDB): 시계열 메트릭 데이터
- vector (Milvus): 문서 임베딩
- graph (Neo4j): 지식 그래프"""

PLANNER_PROMPT = """사용자 질의: {query}

대화 맥락:
{conversation_context}

다음 JSON 형식으로 실행 계획을 수립하세요:
{{
    "intent": "질의 의도 (분석/탐색/예측/비교/설명)",
    "complexity": "simple|medium|complex",
    "language": "ko|en",
    "steps": [
        {{
            "agent": "에이전트 이름",
            "action": "수행할 작업",
            "params": {{}},
            "depends_on": []
        }}
    ],
    "data_sources": ["필요한 데이터 소스"],
    "estimated_hops": 0-5,
    "clarification_needed": null 또는 "명확화 필요한 질문"
}}

JSON만 응답하세요."""


class PlannerAgent(BaseAgent):
    """사용자 질의를 분석하여 실행 계획을 수립하는 에이전트.

    ReAct 패턴:
    1. 질의 의도 분석
    2. 필요한 에이전트 결정
    3. DAG 형태의 태스크 그래프 생성
    """

    def __init__(self, llm_client: LLMClient) -> None:
        super().__init__(name="planner")
        self._llm = llm_client

    async def execute(self, context: AgentContext) -> AgentResult:
        """실행 계획 수립."""
        query = context.query
        conversation = context.conversation_history

        # Format conversation context
        conv_text = ""
        if conversation:
            conv_text = "\n".join(
                f"{msg['role']}: {msg['content'][:200]}" for msg in conversation[-5:]
            )

        prompt = PLANNER_PROMPT.format(
            query=query,
            conversation_context=conv_text or "이전 대화 없음",
        )

        response = await self._llm.generate(
            prompt=prompt,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=1500,
        )

        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                plan = json.loads(content[start:end])
            else:
                plan = self._default_plan(query)

        # Validate and enrich plan
        plan = self._validate_plan(plan, query)

        logger.info(
            "plan_created",
            intent=plan.get("intent"),
            complexity=plan.get("complexity"),
            step_count=len(plan.get("steps", [])),
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={"plan": plan},
            metadata={"tokens_used": response.total_tokens},
        )

    @staticmethod
    def _default_plan(query: str) -> dict[str, Any]:
        """기본 실행 계획 (LLM 실패 시)."""
        return {
            "intent": "분석",
            "complexity": "medium",
            "language": "ko",
            "steps": [
                {"agent": "retriever", "action": "search", "params": {"sources": ["vector"]}, "depends_on": []},
                {"agent": "reasoner", "action": "reason", "params": {}, "depends_on": ["retriever"]},
                {"agent": "validator", "action": "validate", "params": {}, "depends_on": ["reasoner"]},
            ],
            "data_sources": ["vector"],
            "estimated_hops": 0,
            "clarification_needed": None,
        }

    @staticmethod
    def _validate_plan(plan: dict[str, Any], query: str) -> dict[str, Any]:
        """실행 계획 유효성 검증 및 보완."""
        valid_agents = {"retriever", "extractor", "reasoner", "validator"}
        steps = plan.get("steps", [])

        # Ensure all agents are valid
        validated_steps = [s for s in steps if s.get("agent") in valid_agents]

        # Ensure at least retriever -> reasoner -> validator pipeline
        if not validated_steps:
            validated_steps = [
                {"agent": "retriever", "action": "search", "params": {"sources": ["vector"]}, "depends_on": []},
                {"agent": "reasoner", "action": "reason", "params": {}, "depends_on": ["retriever"]},
                {"agent": "validator", "action": "validate", "params": {}, "depends_on": ["reasoner"]},
            ]

        plan["steps"] = validated_steps
        return plan
