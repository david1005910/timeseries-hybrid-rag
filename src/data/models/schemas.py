"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# --- Query API ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="자연어 질의", min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="세션 ID (대화 맥락 유지)")
    options: QueryOptions | None = None


class QueryOptions(BaseModel):
    max_hops: int = Field(default=5, ge=1, le=10)
    include_reasoning: bool = Field(default=True)
    stream: bool = Field(default=False)
    language: str = Field(default="auto", description="ko, en, auto")


class ReasoningStep(BaseModel):
    step: int
    action: str
    agent: str
    result: str
    duration_ms: float
    confidence: float | None = None


class SourceReference(BaseModel):
    source_type: str  # timeseries, graph, vector, document
    source_id: str
    content: str
    relevance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_chain: list[ReasoningStep] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    graph_path: list[dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float
    session_id: str | None = None
    warnings: list[str] = Field(default_factory=list)


# --- Session API ---
class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageResponse(BaseModel):
    role: str  # user, assistant
    content: str
    sources: list[SourceReference] = Field(default_factory=list)
    created_at: datetime


# --- Graph API ---
class EntityResponse(BaseModel):
    id: str
    type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationshipResponse(BaseModel):
    source_id: str
    target_id: str
    type: str
    confidence: float
    properties: dict[str, Any] = Field(default_factory=dict)


# --- Feedback API ---
class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    correction: str | None = None


# --- Health API ---
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    services: dict[str, str] = Field(default_factory=dict)


# --- Metrics API ---
class MetricsQuery(BaseModel):
    measurement: str
    tags: dict[str, str] = Field(default_factory=dict)
    fields: list[str] = Field(default_factory=list)
    start: str = "-1h"
    stop: str = "now()"
    aggregation: str | None = None  # mean, max, min, sum
    group_by: str | None = None  # time interval like "5m"


class MetricsResponse(BaseModel):
    measurement: str
    records: list[dict[str, Any]]
    count: int
    query_time_ms: float
