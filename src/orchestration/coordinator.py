"""Agent Coordinator: orchestrates multi-agent execution pipeline."""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.agents.extractor.agent import ExtractorAgent
from src.agents.reasoner.agent import ReasonerAgent
from src.agents.retriever.agent import RetrieverAgent
from src.agents.validator.agent import ValidatorAgent
from src.data.repositories.graph import GraphRepository
from src.data.repositories.session import SessionRepository
from src.data.repositories.timeseries import TimeseriesRepository
from src.data.repositories.vector import VectorRepository
from src.data.models.schemas import QueryRequest, QueryResponse, ReasoningStep, SourceReference
from src.llm.client import LLMClient
from src.llm.embeddings import EmbeddingService
from src.orchestration.planner import PlannerAgent
from src.reasoning.cot.processor import CoTProcessor
from src.reasoning.selfrag.verifier import SelfRAGVerifier
from src.utils.logging import get_logger
from src.utils.metrics import QueryMetrics

logger = get_logger(__name__)


class AgentCoordinator:
    """에이전트 코디네이터: 전체 파이프라인 조율.

    Pipeline:
    1. Planner → 실행 계획 수립
    2. Retriever → 다중 소스 검색
    3. Extractor → 엔티티/관계 추출 (선택)
    4. Reasoner → CoT 추론
    5. Validator → Self-RAG 검증
    6. 검증 실패 시 → 재검색/재생성 (최대 2회)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_service: EmbeddingService,
        timeseries_repo: TimeseriesRepository,
        vector_repo: VectorRepository,
        graph_repo: GraphRepository,
        session_repo: SessionRepository,
    ) -> None:
        self._llm = llm_client
        self._embedding = embedding_service

        # Agents
        self._planner = PlannerAgent(llm_client)
        self._retriever = RetrieverAgent(timeseries_repo, vector_repo, graph_repo, embedding_service)
        self._extractor = ExtractorAgent(llm_client, graph_repo)
        self._reasoner = ReasonerAgent(llm_client)
        self._validator = ValidatorAgent(llm_client)

        # Reasoning
        self._cot = CoTProcessor()
        self._verifier = SelfRAGVerifier(llm_client)

        # Session
        self._session_repo = session_repo

    async def process_query(self, request: QueryRequest, user_id: str | None = None) -> QueryResponse:
        """질의 처리 전체 파이프라인."""
        query_id = str(uuid4())
        metrics = QueryMetrics(query_id=query_id)
        chain = self._cot.start_chain(query_id, request.query)
        t0 = time.time()

        options = {}
        if request.options:
            options = {
                "max_hops": request.options.max_hops,
                "include_reasoning": request.options.include_reasoning,
                "language": request.options.language,
            }

        # Build context
        conversation_history: list[dict[str, str]] = []
        if request.session_id:
            messages = await self._session_repo.get_messages(request.session_id)
            conversation_history = [{"role": m.role, "content": m.content} for m in messages[-10:]]

        context = AgentContext(
            query=request.query,
            session_id=request.session_id,
            user_id=user_id,
            options=options,
            conversation_history=conversation_history,
        )

        # Step 1: Plan
        plan_result = await self._planner.run(context)
        chain.add_step("plan", "실행 계획 수립", duration_ms=plan_result.duration_ms)
        metrics.add_step("plan", plan_result.duration_ms)

        plan = plan_result.data.get("plan", {})
        steps = plan.get("steps", [])

        # Step 2: Retrieve
        retrieval_params = self._get_retrieval_params(steps)
        context.options.update(retrieval_params)
        retrieve_result = await self._retriever.run(context)
        chain.add_step(
            "retrieve",
            f"검색 완료: {retrieve_result.data.get('total_count', 0)}건",
            duration_ms=retrieve_result.duration_ms,
        )
        metrics.add_step("retrieve", retrieve_result.duration_ms)

        documents = retrieve_result.data.get("documents", [])

        # Step 3: Extract (if in plan)
        if any(s.get("agent") == "extractor" for s in steps):
            context.previous_results = {"documents": documents}
            extract_result = await self._extractor.run(context)
            chain.add_step(
                "extract",
                f"엔티티 {extract_result.data.get('entity_count', 0)}개 추출",
                duration_ms=extract_result.duration_ms,
            )
            metrics.add_step("extract", extract_result.duration_ms)

        # Step 4: Reason
        context.previous_results = {"documents": documents}
        reason_result = await self._reasoner.run(context)
        chain.add_step(
            "reason",
            "추론 완료",
            confidence=reason_result.data.get("confidence"),
            duration_ms=reason_result.duration_ms,
        )
        metrics.add_step("reason", reason_result.duration_ms)
        metrics.total_tokens += reason_result.metadata.get("tokens_used", 0)
        metrics.total_cost_usd += reason_result.metadata.get("cost_usd", 0.0)

        answer = reason_result.data.get("answer", "")
        confidence = reason_result.data.get("confidence", 0.5)

        # Step 5: Validate (Self-RAG)
        verification = await self._verifier.verify(
            query=request.query,
            answer=answer,
            evidence=documents[:5],
            current_confidence=confidence,
        )

        chain.add_step(
            "validate",
            f"검증: {'통과' if verification.passed else '재시도 필요'} (신뢰도: {verification.confidence:.2f})",
            confidence=verification.confidence,
        )

        # Step 6: Retry if needed
        if not verification.passed and verification.retrieve_needed:
            # Re-retrieve and re-reason (1 retry)
            retrieve_result_2 = await self._retriever.run(context)
            documents = retrieve_result_2.data.get("documents", [])
            context.previous_results = {"documents": documents}
            reason_result_2 = await self._reasoner.run(context)
            answer = reason_result_2.data.get("answer", answer)
            confidence = reason_result_2.data.get("confidence", confidence)

            # Re-verify
            verification = await self._verifier.verify(
                query=request.query,
                answer=answer,
                evidence=documents[:5],
                current_confidence=confidence,
            )
            chain.add_step("retry", "재검색 + 재추론 + 재검증")

        # Finalize
        final_answer = verification.final_answer
        final_confidence = verification.confidence
        self._cot.finalize_chain(query_id, final_answer, final_confidence)

        # Build sources
        sources = [
            SourceReference(
                source_type=doc.get("source_type", "unknown"),
                source_id=doc.get("id", ""),
                content=doc.get("content", "")[:200],
                relevance_score=doc.get("relevance_score", 0),
                metadata=doc.get("metadata", {}),
            )
            for doc in documents[:5]
        ]

        # Build reasoning steps
        reasoning_chain = [
            ReasoningStep(
                step=s.step_number,
                action=s.action,
                agent=s.action,
                result=s.description,
                duration_ms=s.duration_ms,
                confidence=s.confidence,
            )
            for s in chain.steps
        ]

        processing_time = (time.time() - t0) * 1000

        # Save to session
        session_id = request.session_id
        if session_id and user_id:
            await self._session_repo.add_message(session_id, "user", request.query)
            await self._session_repo.add_message(
                session_id,
                "assistant",
                final_answer,
                sources=[s.model_dump() for s in sources],
                confidence=final_confidence,
                processing_time_ms=processing_time,
            )

        warnings = verification.warnings if hasattr(verification, "warnings") else []

        return QueryResponse(
            id=query_id,
            answer=final_answer,
            confidence=final_confidence,
            reasoning_chain=reasoning_chain,
            sources=sources,
            graph_path=reason_result.data.get("causal_chain", []),
            processing_time_ms=round(processing_time, 2),
            session_id=session_id,
            warnings=warnings,
        )

    @staticmethod
    def _get_retrieval_params(steps: list[dict[str, Any]]) -> dict[str, Any]:
        """실행 계획에서 검색 파라미터 추출."""
        for step in steps:
            if step.get("agent") == "retriever":
                params = step.get("params", {})
                return {"retrieval_plan": params}
        return {"retrieval_plan": {"sources": ["vector"]}}
