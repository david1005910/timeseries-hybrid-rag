# Hybrid RAG System

GraphRAG + Self-RAG + Multi-Agent 하이브리드 추론 강화 RAG 시스템

시계열 데이터의 복잡한 패턴과 인과관계를 이해하고, 신뢰할 수 있는 분석 결과를 자연어로 제공하는 지능형 추론 시스템입니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│       REST API │ WebSocket │ GraphQL (Optional)              │
├─────────────────────────────────────────────────────────────┤
│                   Orchestration Layer                        │
│      Planner Agent │ Coordinator │ Session Manager           │
├─────────────────────────────────────────────────────────────┤
│                      Agent Layer                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Retriever │ │Extractor │ │Reasoner  │ │Validator │       │
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
├─────────────────────────────────────────────────────────────┤
│                     Reasoning Layer                          │
│    GraphRAG Engine │ Self-RAG Verifier │ CoT Processor       │
├─────────────────────────────────────────────────────────────┤
│                       Data Layer                             │
│   InfluxDB │ Neo4j (Graph) │ Milvus (Vector) │ PostgreSQL   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry

### Setup

```bash
# Clone
git clone https://github.com/david1005910/timeseries-hybrid-rag.git
cd timeseries-hybrid-rag

# Install dependencies
poetry install

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker-compose -f docker/docker-compose.yml up -d

# Run the server
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

After starting the server, visit: http://localhost:8000/docs

## Key Features

- **다중 홉 추론**: GraphRAG 기반 최대 5홉 인과관계 탐색
- **자기 검증**: Self-RAG Reflection Tokens으로 자동 답변 검증
- **병렬 처리**: Multi-Agent 아키텍처를 통한 병렬 검색/추론
- **설명 가능성**: Chain-of-Thought 추론 과정 투명 제공

## Testing

```bash
poetry run pytest                           # All tests
poetry run pytest tests/unit/               # Unit tests only
poetry run pytest --cov=src --cov-report=term-missing  # With coverage
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Python 3.11 |
| 시계열 DB | InfluxDB 3.0 |
| 그래프 DB | Neo4j 5.x |
| 벡터 DB | Milvus 2.3+ |
| 관계형 DB | PostgreSQL 15+ |
| LLM | Anthropic Claude / GPT-4 |
| Message Queue | Redis Streams 7.0+ |
