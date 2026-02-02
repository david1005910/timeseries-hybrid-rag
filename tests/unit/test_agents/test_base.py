"""Tests for BaseAgent."""
from __future__ import annotations

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent


class ConcreteAgent(BaseAgent):
    """테스트용 구체적 에이전트."""

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={"message": f"Processed: {context.query}"},
        )


class FailingAgent(BaseAgent):
    """테스트용 실패 에이전트."""

    async def execute(self, context: AgentContext) -> AgentResult:
        raise ValueError("Test error")


class TimeoutAgent(BaseAgent):
    """테스트용 타임아웃 에이전트."""

    async def execute(self, context: AgentContext) -> AgentResult:
        raise TimeoutError("Timeout")


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(query="테스트 질의")


class TestBaseAgent:
    async def test_successful_execution(self, context: AgentContext) -> None:
        agent = ConcreteAgent(name="test-agent")
        result = await agent.run(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.agent_name == "test-agent"
        assert result.data["message"] == "Processed: 테스트 질의"
        assert result.duration_ms > 0

    async def test_failed_execution(self, context: AgentContext) -> None:
        agent = FailingAgent(name="failing-agent")
        result = await agent.run(context)

        assert result.status == AgentStatus.FAILED
        assert result.error == "Test error"
        assert result.duration_ms > 0
        assert agent.status == AgentStatus.FAILED

    async def test_timeout_execution(self, context: AgentContext) -> None:
        agent = TimeoutAgent(name="timeout-agent")
        result = await agent.run(context)

        assert result.status == AgentStatus.TIMEOUT
        assert "timed out" in result.error
        assert agent.status == AgentStatus.TIMEOUT

    async def test_agent_initial_status(self) -> None:
        agent = ConcreteAgent(name="test")
        assert agent.status == AgentStatus.IDLE

    async def test_agent_context_defaults(self) -> None:
        ctx = AgentContext(query="test")
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.options == {}
        assert ctx.previous_results == {}
        assert ctx.conversation_history == []
