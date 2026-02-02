"""BaseAgent abstract class and shared agent types."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentContext:
    """에이전트 실행에 필요한 컨텍스트."""

    query: str
    session_id: str | None = None
    user_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    previous_results: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class AgentResult:
    """에이전트 실행 결과."""

    agent_name: str
    status: AgentStatus
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """모든 에이전트의 기본 추상 클래스.

    SOLID 원칙:
    - SRP: 각 에이전트는 하나의 책임만 가짐
    - OCP: 새로운 에이전트 타입 추가 시 기존 코드 수정 불필요
    - LSP: 모든 에이전트는 BaseAgent 인터페이스 준수
    - ISP: 필요한 메서드만 노출
    - DIP: 추상화에 의존
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = AgentStatus.IDLE

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """에이전트의 핵심 실행 로직.

        Args:
            context: 실행 컨텍스트 (질의, 옵션, 이전 결과)

        Returns:
            AgentResult with status, data, and metadata
        """
        ...

    async def run(self, context: AgentContext) -> AgentResult:
        """실행 + 에러 핸들링 + 메트릭 수집 래퍼."""
        self.status = AgentStatus.RUNNING
        t0 = time.time()

        try:
            result = await self.execute(context)
            self.status = result.status
            result.duration_ms = (time.time() - t0) * 1000
            return result
        except TimeoutError:
            self.status = AgentStatus.TIMEOUT
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.TIMEOUT,
                error="Agent execution timed out",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                error=str(e),
                duration_ms=(time.time() - t0) * 1000,
            )
