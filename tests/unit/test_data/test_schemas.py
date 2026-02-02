"""Tests for Pydantic schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.models.schemas import (
    FeedbackRequest,
    MetricsQuery,
    QueryOptions,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
    SourceReference,
)


class TestQueryRequest:
    def test_valid_request(self) -> None:
        req = QueryRequest(query="CPU 급증 원인 분석")
        assert req.query == "CPU 급증 원인 분석"
        assert req.session_id is None
        assert req.options is None

    def test_with_options(self) -> None:
        req = QueryRequest(
            query="테스트",
            session_id="sess-1",
            options=QueryOptions(max_hops=3, include_reasoning=False),
        )
        assert req.options.max_hops == 3
        assert req.options.include_reasoning is False

    def test_empty_query_fails(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_max_hops_validation(self) -> None:
        with pytest.raises(ValidationError):
            QueryOptions(max_hops=0)  # min is 1

        with pytest.raises(ValidationError):
            QueryOptions(max_hops=11)  # max is 10


class TestQueryResponse:
    def test_valid_response(self) -> None:
        resp = QueryResponse(
            answer="테스트 답변",
            confidence=0.85,
            processing_time_ms=1234.5,
        )
        assert resp.answer == "테스트 답변"
        assert resp.confidence == 0.85
        assert resp.id is not None  # auto-generated

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryResponse(answer="test", confidence=1.5, processing_time_ms=100)

        with pytest.raises(ValidationError):
            QueryResponse(answer="test", confidence=-0.1, processing_time_ms=100)


class TestFeedbackRequest:
    def test_valid_feedback(self) -> None:
        fb = FeedbackRequest(query_id="q1", rating=4, comment="좋은 답변")
        assert fb.rating == 4

    def test_rating_bounds(self) -> None:
        with pytest.raises(ValidationError):
            FeedbackRequest(query_id="q1", rating=0)

        with pytest.raises(ValidationError):
            FeedbackRequest(query_id="q1", rating=6)


class TestMetricsQuery:
    def test_defaults(self) -> None:
        mq = MetricsQuery(measurement="cpu")
        assert mq.start == "-1h"
        assert mq.stop == "now()"
        assert mq.aggregation is None
