# Hybrid RAG System

GraphRAG + Self-RAG + Multi-Agent 하이브리드 추론 강화 RAG 시스템

시계열 인프라 모니터링 데이터의 복잡한 패턴과 인과관계를 이해하고, 신뢰할 수 있는 분석 결과를 자연어로 제공하는 지능형 추론 시스템입니다.

## Architecture

5-Layer 아키텍처로 설계되어 있으며, 각 계층은 독립적으로 확장 가능합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                     Presentation Layer                        │
│    REST API (7 endpoints)  │  WebSocket Streaming             │
│    Auth / Rate Limit / Request Logging Middleware             │
├──────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                        │
│       PlannerAgent  │  AgentCoordinator  │  SessionManager   │
├──────────────────────────────────────────────────────────────┤
│                        Agent Layer                            │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│   │ Retriever │ │ Extractor │ │ Reasoner  │ │ Validator │   │
│   │   Agent   │ │   Agent   │ │   Agent   │ │   Agent   │   │
│   └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
├──────────────────────────────────────────────────────────────┤
│                      Reasoning Layer                         │
│     GraphRAG Engine  │  Self-RAG Verifier  │  CoT Processor  │
├──────────────────────────────────────────────────────────────┤
│                        Data Layer                            │
│   InfluxDB  │  Neo4j (Graph)  │  Milvus (Vector)  │  PgSQL  │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **GraphRAG 다중 홉 추론**: Neo4j 지식 그래프에서 최대 5홉까지 인과관계를 탐색하고, Leiden 커뮤니티 감지로 관련 엔티티를 그룹화
- **Self-RAG 자기 검증**: `[Retrieve]`, `[IsREL]`, `[IsSUP]`, `[IsUSE]` Reflection Tokens으로 답변의 정확성을 자동 검증하고 환각(hallucination) 감지
- **Multi-Agent 병렬 처리**: 4개 에이전트가 검색/추출/추론/검증을 병렬 수행, Planner가 DAG 기반 실행 계획 수립
- **Chain-of-Thought**: 단계별 추론 과정을 투명하게 제공하여 설명 가능한 AI 구현
- **LLM Fallback**: Anthropic Claude (primary) -> OpenAI GPT-4 (fallback) + 자동 재시도
- **다중 데이터 소스 통합**: 시계열(InfluxDB) + 벡터(Milvus) + 그래프(Neo4j) 병렬 검색 후 병합/랭킹

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry

### Installation

```bash
# Clone
git clone https://github.com/david1005910/timeseries-hybrid-rag.git
cd timeseries-hybrid-rag

# Install dependencies
poetry install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
```

### Start Infrastructure

```bash
# Start all databases (PostgreSQL, InfluxDB, Neo4j, Milvus, Redis)
docker-compose -f docker/docker-compose.yml up -d

# Initialize database schemas
poetry run python scripts/setup_db.py

# Seed sample data (optional)
poetry run python scripts/seed_data.py
```

### Run the Server

```bash
poetry run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

API 문서: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | 서비스 상태 확인 |
| `POST` | `/api/v1/query` | 자연어 질의 처리 (전체 RAG 파이프라인) |
| `POST` | `/api/v1/query/{id}/feedback` | 질의 결과 피드백 제출 |
| `GET` | `/api/v1/sessions` | 세션 목록 조회 |
| `POST` | `/api/v1/sessions` | 새 세션 생성 |
| `GET` | `/api/v1/sessions/{id}` | 세션 메시지 조회 |
| `DELETE` | `/api/v1/sessions/{id}` | 세션 삭제 |
| `GET` | `/api/v1/graph/entities` | 지식 그래프 엔티티 조회 |
| `GET` | `/api/v1/graph/entities/{id}/traverse` | 다중 홉 그래프 탐색 |
| `POST` | `/api/v1/metrics/query` | 시계열 메트릭 조회 |
| `WS` | `/ws/query` | WebSocket 스트리밍 질의 |

### Query API Example

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "어제 CPU 급증 원인을 분석해줘",
    "options": {
      "max_hops": 5,
      "include_reasoning": true
    }
  }'
```

**Response:**
```json
{
  "id": "query-uuid",
  "answer": "CPU 급증은 14:30에 시작된 배치 작업(batch-7842)이 원인입니다...",
  "confidence": 0.87,
  "reasoning_chain": [
    {"step": 1, "action": "retrieve", "agent": "retriever", "result": "..."},
    {"step": 2, "action": "reason", "agent": "reasoner", "result": "..."},
    {"step": 3, "action": "validate", "agent": "validator", "result": "..."}
  ],
  "sources": [...],
  "graph_path": [...],
  "processing_time_ms": 2340
}
```

## Project Structure

```
src/
├── api/                    # Presentation Layer
│   ├── routes/             #   REST endpoints (health, query, sessions, graph, metrics)
│   ├── websocket/          #   WebSocket streaming handler
│   └── middleware/          #   Auth, rate limiting, request logging
├── orchestration/          # Orchestration Layer
│   ├── planner.py          #   LLM 기반 실행 계획 수립
│   ├── coordinator.py      #   Agent 실행 조율
│   └── session_manager.py  #   대화 세션 관리
├── agents/                 # Agent Layer
│   ├── base.py             #   BaseAgent ABC (execute + run wrapper)
│   ├── retriever/          #   다중 소스 병렬 검색
│   ├── extractor/          #   엔티티/관계 추출
│   ├── reasoner/           #   Chain-of-Thought 추론
│   └── validator/          #   Self-RAG 검증
├── reasoning/              # Reasoning Layer
│   ├── graphrag/           #   GraphRAG 엔진 (탐색, 커뮤니티 감지)
│   ├── selfrag/            #   Self-RAG Verifier + Reflection Tokens
│   └── cot/                #   Chain-of-Thought Processor
├── data/                   # Data Layer
│   ├── repositories/       #   InfluxDB, Neo4j, Milvus, PostgreSQL
│   └── models/             #   Pydantic schemas, graph entities
├── llm/                    # LLM Integration
│   ├── client.py           #   Anthropic/OpenAI 듀얼 클라이언트
│   ├── embeddings.py       #   임베딩 생성 + 캐싱
│   └── prompts/            #   Jinja2 프롬프트 템플릿
├── config/                 # Configuration
└── utils/                  # Logging, metrics utilities
```

## Testing

182개 테스트 전체 통과. 모든 외부 서비스는 mock 처리되어 Docker 없이 실행 가능합니다.

```bash
# 전체 테스트
poetry run pytest tests/ -v

# 카테고리별 실행
poetry run pytest tests/unit/               # 단위 테스트
poetry run pytest tests/integration/        # 통합 테스트
poetry run pytest tests/contract/           # API 계약 테스트

# 커버리지 리포트
poetry run pytest --cov=src --cov-report=term-missing
```

| Category | Tests | Coverage |
|----------|-------|----------|
| Data Layer (InfluxDB, Neo4j, Milvus, PostgreSQL) | 45 | Repositories |
| LLM / Embedding | 18 | Client fallback, streaming, caching |
| Agents (Retriever, Extractor, Reasoner, Validator, Planner) | 48 | Agent logic |
| Reasoning (GraphRAG, Self-RAG, CoT) | 22 | Reasoning engines |
| Integration API | 13 | HTTP endpoints |
| Contract / OpenAPI | 29 | Spec compliance |
| **Total** | **182** | |

## Deployment

### Docker

```bash
# 전체 스택 실행 (앱 + 인프라)
docker-compose -f docker/docker-compose.yml up -d

# 앱만 빌드
docker build -f docker/Dockerfile -t hybrid-rag-system .
```

### Kubernetes

```bash
# 네임스페이스 생성
kubectl create namespace hybrid-rag

# Secrets 생성 (API 키 등)
kubectl create secret generic hybrid-rag-secrets \
  --from-env-file=.env \
  -n hybrid-rag

# 배포
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml    # Auto-scaling: 2-8 pods, CPU 70%
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI + Python 3.11 | 비동기 REST API |
| Time-series DB | InfluxDB 2.7 | 시스템 메트릭 저장/조회 |
| Graph DB | Neo4j 5.17 | 인과관계 지식 그래프 |
| Vector DB | Milvus 2.4 | 문서 임베딩 유사도 검색 |
| RDBMS | PostgreSQL 15 | 사용자/세션/피드백 |
| Cache | Redis 7.2 | 캐싱, 임베딩 캐시 |
| LLM | Anthropic Claude / OpenAI GPT-4 | 추론, 생성, 검증 |
| Embedding | OpenAI text-embedding-3-large | 1536차원 벡터 생성 |

## Seed Data

`scripts/seed_data.py`는 개발/테스트용 샘플 데이터를 생성합니다:

- **InfluxDB**: 1,000개 시계열 포인트 (CPU, Memory, Disk, Network) - ~500분 전 CPU spike 패턴 포함
- **Neo4j**: 지식 그래프 (9 entities, 5 metrics, 6 events, 3 concepts, 20+ relationships)
  - 인과 체인: Traffic spike -> Connection pool exhaustion -> CPU spike -> Service degradation
- **Milvus**: 8개 문서 임베딩 (장애 보고서, 분석 리포트, 인프라 문서)
- **PostgreSQL**: 테스트 사용자 (admin@example.com)

## License

MIT
