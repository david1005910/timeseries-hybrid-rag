from __future__ import annotations

from enum import Enum


# Agent types
class AgentType(str, Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    EXTRACTOR = "extractor"
    REASONER = "reasoner"
    VALIDATOR = "validator"


# Graph node types
class NodeType(str, Enum):
    METRIC = "metric"
    EVENT = "event"
    ENTITY = "entity"
    DOCUMENT = "document"
    CONCEPT = "concept"


# Graph relationship types
class RelationshipType(str, Enum):
    CAUSES = "causes"
    CORRELATES_WITH = "correlates"
    BELONGS_TO = "belongs_to"
    PRECEDES = "precedes"


# Self-RAG reflection tokens
class ReflectionToken(str, Enum):
    RETRIEVE = "[Retrieve]"
    IS_RELEVANT = "[IsREL]"
    IS_SUPPORTED = "[IsSUP]"
    IS_USEFUL = "[IsUSE]"


# Performance limits
MAX_HOPS = 5
MAX_QUERY_TIMEOUT_SECONDS = 30
SIMPLE_QUERY_TIMEOUT_SECONDS = 5
MAX_CONCURRENT_AGENTS = 10
DEFAULT_TOP_K = 10
EMBEDDING_BATCH_SIZE = 100
TEXT_RECORD_BATCH_SIZE = 96
