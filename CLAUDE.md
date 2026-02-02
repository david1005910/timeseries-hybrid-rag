# Hybrid RAG System - Development Guide

## Project Overview
GraphRAG + Self-RAG + Multi-Agent hybrid reasoning RAG system for time-series infrastructure monitoring.

## Tech Stack
- **Backend**: Python 3.11+ / FastAPI
- **Time-series DB**: InfluxDB 3.0 (Flux queries)
- **Graph DB**: Neo4j 5.x (async driver)
- **Vector DB**: Milvus 2.3+ (pymilvus)
- **RDBMS**: PostgreSQL 15+ (SQLAlchemy async + asyncpg)
- **Cache/Queue**: Redis 7.0+
- **LLM**: Anthropic Claude (primary) / OpenAI GPT-4 (fallback)

## Architecture (5-Layer)
1. **Presentation**: FastAPI REST API, WebSocket streaming
2. **Orchestration**: PlannerAgent, AgentCoordinator, SessionManager
3. **Agent**: RetrieverAgent, ExtractorAgent, ReasonerAgent, ValidatorAgent
4. **Reasoning**: GraphRAG Engine, Self-RAG Verifier, CoT Processor
5. **Data**: InfluxDB, Neo4j, Milvus, PostgreSQL repositories

## Key Commands
```bash
# Run tests
poetry run pytest tests/ -v

# Run specific test category
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v

# Run app
poetry run uvicorn src.main:app --reload --port 8000

# Setup databases
poetry run python scripts/setup_db.py

# Seed sample data
poetry run python scripts/seed_data.py
```

## Code Conventions
- Type hints required on all functions
- Black formatter (line-length=100) + isort
- asyncio_mode = "auto" for pytest
- Structured logging via structlog
- Korean docstrings for domain-specific modules

## Testing
- 153 tests total (unit + integration)
- All external services mocked (no Docker needed for tests)
- pytest-asyncio for async test support
- aiosqlite for in-memory DB tests (SessionRepository)

## Important Patterns
- `BaseAgent.execute()` abstract method + `run()` wrapper with error handling
- `async with await self._get_session()` pattern for Neo4j sessions
- LLM fallback: Anthropic -> OpenAI with tenacity retry
- Self-RAG reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE]
- Graph traversal max 5 hops with confidence filtering
