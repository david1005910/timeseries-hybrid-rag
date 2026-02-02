"""Tests for PlannerAgent (orchestration/planner)."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.llm.client import LLMClient, LLMResponse
from src.orchestration.planner import PlannerAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM client returning a valid execution plan."""
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps({
                "intent": "분석",
                "complexity": "medium",
                "language": "ko",
                "steps": [
                    {"agent": "retriever", "action": "search", "params": {"sources": ["vector", "timeseries"]}, "depends_on": []},
                    {"agent": "reasoner", "action": "reason", "params": {}, "depends_on": ["retriever"]},
                    {"agent": "validator", "action": "validate", "params": {}, "depends_on": ["reasoner"]},
                ],
                "data_sources": ["vector", "timeseries"],
                "estimated_hops": 2,
                "clarification_needed": None,
            }),
            provider="mock",
            model="mock-model",
            input_tokens=200,
            output_tokens=300,
            elapsed_ms=400.0,
        )
    )
    return client


@pytest.fixture
def planner(mock_llm: MagicMock) -> PlannerAgent:
    return PlannerAgent(llm_client=mock_llm)


@pytest.fixture
def basic_context() -> AgentContext:
    return AgentContext(
        query="어제 CPU 급증 원인을 분석해줘",
        session_id="test-session",
        user_id="test-user",
    )


@pytest.fixture
def context_with_history() -> AgentContext:
    return AgentContext(
        query="그 문제의 후속 영향은?",
        session_id="test-session",
        user_id="test-user",
        conversation_history=[
            {"role": "user", "content": "어제 CPU 급증 원인을 분석해줘"},
            {"role": "assistant", "content": "배치 작업이 원인으로 확인되었습니다."},
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlannerAgent:
    async def test_execute_creates_valid_plan(
        self, planner: PlannerAgent, basic_context: AgentContext, mock_llm: MagicMock,
    ) -> None:
        """execute() should produce SUCCESS result with a well-formed plan."""
        result = await planner.execute(basic_context)

        assert result.status == AgentStatus.SUCCESS
        assert result.agent_name == "planner"

        plan = result.data["plan"]
        assert plan["intent"] == "분석"
        assert plan["complexity"] == "medium"
        assert len(plan["steps"]) == 3

        # Verify LLM was called with query content
        mock_llm.generate.assert_awaited_once()
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "CPU 급증" in call_kwargs["prompt"]

    async def test_execute_with_conversation_history(
        self, planner: PlannerAgent, context_with_history: AgentContext, mock_llm: MagicMock,
    ) -> None:
        """execute() should include conversation history in the prompt."""
        result = await planner.execute(context_with_history)

        assert result.status == AgentStatus.SUCCESS

        call_kwargs = mock_llm.generate.call_args.kwargs
        prompt = call_kwargs["prompt"]
        # The conversation history should be formatted into the prompt
        assert "배치 작업" in prompt or "CPU 급증" in prompt

    async def test_execute_without_conversation_history(
        self, planner: PlannerAgent, mock_llm: MagicMock,
    ) -> None:
        """execute() without history should include the fallback text."""
        context = AgentContext(query="테스트 질의", conversation_history=[])
        result = await planner.execute(context)

        assert result.status == AgentStatus.SUCCESS
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "이전 대화 없음" in call_kwargs["prompt"]

    async def test_execute_llm_returns_malformed_json_with_braces(
        self, planner: PlannerAgent, mock_llm: MagicMock, basic_context: AgentContext,
    ) -> None:
        """execute() should extract JSON from text that wraps it in extra content."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content='여기 실행 계획입니다: {"intent":"탐색","complexity":"simple","language":"ko","steps":[{"agent":"retriever","action":"search","params":{},"depends_on":[]}],"data_sources":["vector"],"estimated_hops":0,"clarification_needed":null}',
                provider="mock", model="mock", input_tokens=50,
                output_tokens=100, elapsed_ms=150.0,
            )
        )

        result = await planner.execute(basic_context)

        assert result.status == AgentStatus.SUCCESS
        plan = result.data["plan"]
        assert plan["intent"] == "탐색"

    async def test_execute_llm_returns_completely_invalid(
        self, planner: PlannerAgent, mock_llm: MagicMock, basic_context: AgentContext,
    ) -> None:
        """execute() should fall back to default plan on completely invalid LLM output."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="파싱 불가능한 텍스트만 있습니다",
                provider="mock", model="mock", input_tokens=10,
                output_tokens=20, elapsed_ms=50.0,
            )
        )

        result = await planner.execute(basic_context)

        assert result.status == AgentStatus.SUCCESS
        plan = result.data["plan"]
        # Default plan always contains retriever -> reasoner -> validator
        agent_names = [s["agent"] for s in plan["steps"]]
        assert "retriever" in agent_names
        assert "reasoner" in agent_names
        assert "validator" in agent_names

    async def test_execute_records_token_usage(
        self, planner: PlannerAgent, basic_context: AgentContext,
    ) -> None:
        """execute() metadata should include tokens_used from LLM response."""
        result = await planner.execute(basic_context)

        assert result.metadata["tokens_used"] == 500  # 200 input + 300 output


class TestDefaultPlan:
    def test_default_plan_structure(self) -> None:
        """_default_plan should return a complete plan with required keys."""
        plan = PlannerAgent._default_plan("테스트 질의")

        assert plan["intent"] == "분석"
        assert plan["complexity"] == "medium"
        assert plan["language"] == "ko"
        assert plan["estimated_hops"] == 0
        assert plan["clarification_needed"] is None

        # Pipeline: retriever -> reasoner -> validator
        steps = plan["steps"]
        assert len(steps) == 3
        assert steps[0]["agent"] == "retriever"
        assert steps[1]["agent"] == "reasoner"
        assert steps[2]["agent"] == "validator"

        # Dependency chain is valid
        assert steps[0]["depends_on"] == []
        assert "retriever" in steps[1]["depends_on"]
        assert "reasoner" in steps[2]["depends_on"]

    def test_default_plan_has_vector_source(self) -> None:
        """_default_plan should default to 'vector' data source."""
        plan = PlannerAgent._default_plan("아무 질의")
        assert "vector" in plan["data_sources"]


class TestValidatePlan:
    def test_validate_plan_with_valid_agents(self) -> None:
        """_validate_plan should keep steps with valid agent names."""
        plan: dict[str, Any] = {
            "intent": "분석",
            "steps": [
                {"agent": "retriever", "action": "search", "params": {}, "depends_on": []},
                {"agent": "extractor", "action": "extract", "params": {}, "depends_on": ["retriever"]},
                {"agent": "reasoner", "action": "reason", "params": {}, "depends_on": ["extractor"]},
                {"agent": "validator", "action": "validate", "params": {}, "depends_on": ["reasoner"]},
            ],
        }

        validated = PlannerAgent._validate_plan(plan, "질의")

        assert len(validated["steps"]) == 4
        agent_names = {s["agent"] for s in validated["steps"]}
        assert agent_names == {"retriever", "extractor", "reasoner", "validator"}

    def test_validate_plan_removes_invalid_agents(self) -> None:
        """_validate_plan should filter out unknown agent names."""
        plan: dict[str, Any] = {
            "steps": [
                {"agent": "retriever", "action": "search", "params": {}, "depends_on": []},
                {"agent": "unknown_agent", "action": "magic", "params": {}, "depends_on": []},
                {"agent": "reasoner", "action": "reason", "params": {}, "depends_on": []},
            ],
        }

        validated = PlannerAgent._validate_plan(plan, "질의")

        agent_names = [s["agent"] for s in validated["steps"]]
        assert "unknown_agent" not in agent_names
        assert "retriever" in agent_names
        assert "reasoner" in agent_names

    def test_validate_plan_all_invalid_falls_back(self) -> None:
        """_validate_plan should produce a default pipeline when all agents are invalid."""
        plan: dict[str, Any] = {
            "steps": [
                {"agent": "bad1", "action": "x", "params": {}, "depends_on": []},
                {"agent": "bad2", "action": "y", "params": {}, "depends_on": []},
            ],
        }

        validated = PlannerAgent._validate_plan(plan, "질의")

        agent_names = [s["agent"] for s in validated["steps"]]
        assert agent_names == ["retriever", "reasoner", "validator"]

    def test_validate_plan_empty_steps_falls_back(self) -> None:
        """_validate_plan should produce a default pipeline when steps list is empty."""
        plan: dict[str, Any] = {"steps": []}

        validated = PlannerAgent._validate_plan(plan, "질의")

        assert len(validated["steps"]) == 3
        assert validated["steps"][0]["agent"] == "retriever"
