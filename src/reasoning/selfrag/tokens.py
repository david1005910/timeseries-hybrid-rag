"""Reflection Token definitions for Self-RAG."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RetrieveDecision(str, Enum):
    YES = "Yes"
    NO = "No"


class RelevanceDecision(str, Enum):
    RELEVANT = "Relevant"
    IRRELEVANT = "Irrelevant"


class SupportLevel(str, Enum):
    FULLY = "Fully"
    PARTIALLY = "Partially"
    NO = "No"


@dataclass
class RetrieveToken:
    """[Retrieve] 검색 필요 여부 판단."""

    decision: RetrieveDecision
    reason: str

    @property
    def needs_retrieval(self) -> bool:
        return self.decision == RetrieveDecision.YES


@dataclass
class RelevanceToken:
    """[IsREL] 문서-질문 관련성 판단."""

    source_idx: int
    decision: RelevanceDecision
    reason: str

    @property
    def is_relevant(self) -> bool:
        return self.decision == RelevanceDecision.RELEVANT


@dataclass
class SupportToken:
    """[IsSUP] 증거 뒷받침 수준 판단."""

    level: SupportLevel
    supported_parts: list[str]
    unsupported_parts: list[str]

    @property
    def is_supported(self) -> bool:
        return self.level in (SupportLevel.FULLY, SupportLevel.PARTIALLY)


@dataclass
class UsefulnessToken:
    """[IsUSE] 전체 유용성 점수."""

    score: int  # 1-5
    reason: str

    @property
    def is_useful(self) -> bool:
        return self.score >= 3
