# Technical Implementation Plan

## 001-hybrid-rag-system

**Branch:** 001-hybrid-rag-system  
**Date:** 2025-02-02  
**Spec:** [spec.md](./spec.md)  
**Status:** Draft

---

## 1. Technical Overview

### 1.1 Solution Summary
GraphRAG, Self-RAG, Multi-Agent 아키텍처를 결합한 하이브리드 추론 강화 RAG 시스템을 Python + FastAPI 기반으로 구현합니다.

### 1.2 Architecture Diagram

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

---

## 2. Technology Stack

### 2.1 Core Technologies

| 계층 | 기술 | 버전 | 선정 이유 |
|------|------|------|----------|
| **Backend Framework** | FastAPI | 0.109+ | 비동기 지원, 자동 OpenAPI 문서 |
| **Runtime** | Python | 3.11+ | AI/ML 생태계, 타입 힌트 개선 |
| **시계열 DB** | InfluxDB | 3.0 | 고성능 시계열 쿼리, SQL 지원 |
| **그래프 DB** | Neo4j | 5.x | 성숙한 생태계, Cypher 쿼리 |
| **벡터 DB** | Milvus | 2.3+ | 대규모 벡터 검색, 확장성 |
| **관계형 DB** | PostgreSQL | 15+ | 세션/사용자 데이터 저장 |
| **Message Queue** | Redis Streams | 7.0+ | 낮은 지연시간, 에이전트 통신 |
| **LLM Provider** | Anthropic Claude | 3.5 Sonnet | 강력한 추론 능력 |
| **Embedding Model** | OpenAI | text-embedding-3-large | 1536차원, 고품질 |

### 2.2 Development Tools

| 도구 | 용도 |
|------|------|
| Poetry | 의존성 관리 |
| Black + isort | 코드 포맷팅 |
| Pylint + mypy | 린트 + 타입 체크 |
| pytest | 테스트 프레임워크 |
| Docker Compose | 로컬 개발 환경 |
| Kubernetes | 프로덕션 배포 |

### 2.3 Monitoring & Observability

| 도구 | 용도 |
|------|------|
| Prometheus | 메트릭 수집 |
| Grafana | 대시보드 |
| Fluent Bit | 로그 수집 |
| Elasticsearch | 로그 저장/검색 |
| Jaeger | 분산 추적 |

---

## 3. Project Structure

```
hybrid-rag-system/
├── src/
│   ├── __init__.py
│   ├── main.py                      # FastAPI 앱 진입점
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py              # 설정 관리 (Pydantic)
│   │   └── constants.py             # 상수 정의
│   │
│   ├── api/                         # Presentation Layer
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── query.py             # /api/v1/query
│   │   │   ├── sessions.py          # /api/v1/sessions
│   │   │   ├── graph.py             # /api/v1/graph
│   │   │   ├── metrics.py           # /api/v1/metrics
│   │   │   └── health.py            # /api/v1/health
│   │   ├── websocket/
│   │   │   ├── __init__.py
│   │   │   └── handlers.py          # WebSocket 핸들러
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py              # 인증 미들웨어
│   │   │   ├── rate_limit.py        # Rate limiting
│   │   │   └── logging.py           # 요청 로깅
│   │   └── dependencies.py          # FastAPI 의존성
│   │
│   ├── orchestration/               # Orchestration Layer
│   │   ├── __init__.py
│   │   ├── planner.py               # Planner Agent
│   │   ├── coordinator.py           # Agent Coordinator
│   │   └── session_manager.py       # Session Manager
│   │
│   ├── agents/                      # Agent Layer
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseAgent 추상 클래스
│   │   ├── retriever/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py             # Retriever Agent
│   │   │   ├── timeseries.py        # 시계열 검색
│   │   │   ├── vector.py            # 벡터 검색
│   │   │   └── graph.py             # 그래프 검색
│   │   ├── extractor/
│   │   │   ├── __init__.py
│   │   │   └── agent.py             # Extractor Agent
│   │   ├── reasoner/
│   │   │   ├── __init__.py
│   │   │   └── agent.py             # Reasoner Agent
│   │   └── validator/
│   │       ├── __init__.py
│   │       └── agent.py             # Validator Agent (Self-RAG)
│   │
│   ├── reasoning/                   # Reasoning Layer
│   │   ├── __init__.py
│   │   ├── graphrag/
│   │   │   ├── __init__.py
│   │   │   ├── engine.py            # GraphRAG Engine
│   │   │   ├── traversal.py         # 그래프 탐색
│   │   │   └── community.py         # 커뮤니티 감지
│   │   ├── selfrag/
│   │   │   ├── __init__.py
│   │   │   ├── verifier.py          # Self-RAG Verifier
│   │   │   └── tokens.py            # Reflection Tokens
│   │   └── cot/
│   │       ├── __init__.py
│   │       └── processor.py         # Chain-of-Thought
│   │
│   ├── data/                        # Data Layer
│   │   ├── __init__.py
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── timeseries.py        # InfluxDB Repository
│   │   │   ├── graph.py             # Neo4j Repository
│   │   │   ├── vector.py            # Milvus Repository
│   │   │   └── session.py           # PostgreSQL Repository
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── entities.py          # 그래프 엔티티
│   │       ├── relationships.py     # 그래프 관계
│   │       └── schemas.py           # Pydantic 스키마
│   │
│   ├── llm/                         # LLM Integration
│   │   ├── __init__.py
│   │   ├── client.py                # LLM 클라이언트
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   ├── planner.py           # 플래너 프롬프트
│   │   │   ├── retriever.py         # 리트리버 프롬프트
│   │   │   ├── reasoner.py          # 리즈너 프롬프트
│   │   │   └── validator.py         # 검증자 프롬프트
│   │   └── embeddings.py            # 임베딩 생성
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py               # 로깅 유틸
│       └── metrics.py               # 메트릭 유틸
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_agents/
│   │   ├── test_reasoning/
│   │   └── test_data/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   └── test_agents/
│   └── contract/
│       ├── __init__.py
│       └── test_api_spec.py
│
├── scripts/
│   ├── setup_db.py                  # DB 초기화
│   └── seed_data.py                 # 테스트 데이터
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml           # 개발 환경
│   └── docker-compose.prod.yml      # 프로덕션
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
│
├── docs/
│   ├── api/
│   │   └── openapi.yaml             # API 스펙
│   └── architecture/
│       └── decisions/               # ADR
│
├── .specify/                        # Spec-Kit
│   ├── memory/
│   │   └── constitution.md
│   ├── specs/
│   │   └── 001-hybrid-rag-system/
│   │       ├── spec.md
│   │       ├── plan.md
│   │       └── tasks.md
│   └── templates/
│
├── pyproject.toml                   # Poetry 설정
├── README.md
└── .env.example
```

---

## 4. Component Design

### 4.1 Multi-Agent Orchestrator

#### 4.1.1 Planner Agent
```python
class PlannerAgent(BaseAgent):
    """사용자 질의를 분석하여 실행 계획 수립"""
    
    async def plan(self, query: str, context: Context) -> ExecutionPlan:
        # 1. 질의 의도 분석
        # 2. 필요한 에이전트 결정
        # 3. DAG 형태의 태스크 그래프 생성
        # 4. ReAct 패턴으로 계획 수립
        pass
```

#### 4.1.2 Retriever Agent
```python
class RetrieverAgent(BaseAgent):
    """다중 데이터 소스에서 관련 정보 검색"""
    
    sources = ["timeseries", "vector", "graph"]
    
    async def retrieve(self, plan: RetrievalPlan) -> List[Document]:
        # 병렬 검색 수행
        # 결과 캐싱
        # 증분 검색 지원
        pass
```

#### 4.1.3 Reasoner Agent
```python
class ReasonerAgent(BaseAgent):
    """검색된 정보를 바탕으로 추론 수행"""
    
    async def reason(self, documents: List[Document]) -> ReasoningResult:
        # Chain-of-Thought 추론
        # 그래프 탐색 결합
        # 신뢰도 점수 계산
        pass
```

#### 4.1.4 Validator Agent (Self-RAG)
```python
class ValidatorAgent(BaseAgent):
    """생성된 결과의 정확성과 일관성 검증"""
    
    reflection_tokens = ["[Retrieve]", "[IsREL]", "[IsSUP]", "[IsUSE]"]
    
    async def validate(self, result: ReasoningResult) -> ValidationResult:
        # 증거 뒷받침 여부 확인
        # 논리적 일관성 검증
        # 사실 정확성 체크
        pass
```

### 4.2 GraphRAG Engine

#### 4.2.1 Node Types
```python
class NodeType(Enum):
    METRIC = "metric"       # 시계열 메트릭 (CPU, Memory)
    EVENT = "event"         # 시스템 이벤트 (장애, 배포)
    ENTITY = "entity"       # 시스템 구성 요소 (서버, 서비스)
    DOCUMENT = "document"   # 관련 문서 (로그, 매뉴얼)
    CONCEPT = "concept"     # 추상적 개념 (패턴, 원인)
```

#### 4.2.2 Relationship Types
```python
class RelationshipType(Enum):
    CAUSES = "causes"               # 인과 관계
    CORRELATES_WITH = "correlates"  # 상관 관계
    BELONGS_TO = "belongs_to"       # 소속 관계
    PRECEDES = "precedes"           # 시간적 선후
```

#### 4.2.3 Graph Pipeline
```python
class GraphPipeline:
    """그래프 구축 파이프라인"""
    
    async def build(self, source_data):
        # 1. 데이터 수집
        # 2. 엔티티 추출 (LLM 사용)
        # 3. 관계 추출 (LLM 사용)
        # 4. Neo4j 저장
        # 5. Leiden 알고리즘으로 커뮤니티 감지
        # 6. 커뮤니티별 요약 생성
        pass
```

### 4.3 Self-RAG Verification

#### 4.3.1 Reflection Tokens
| 토큰 | 역할 | 출력 값 | 트리거 |
|------|------|---------|--------|
| `[Retrieve]` | 검색 필요 여부 | Yes / No | 질의 분석 후 |
| `[IsREL]` | 문서 관련성 | Relevant / Irrelevant | 검색 결과 후 |
| `[IsSUP]` | 증거 뒷받침 | Fully / Partially / No | 답변 생성 후 |
| `[IsUSE]` | 전체 유용성 | Score 1-5 | 최종 검증 |

#### 4.3.2 Verification Workflow
```
Query → [Retrieve Decision] → Retrieval → [IsREL Check]
                                              │
        Irrelevant: Re-retrieve ◄─────────────┤
                                              │ Relevant
                                              ▼
         Generate → [IsSUP Verify] → [IsUSE Score] → Final Output
```

---

## 5. Data Models

### 5.1 시계열 데이터 (InfluxDB)
```
measurement: metrics
tags: host, service, region
fields: cpu_usage, memory_usage, disk_io, network_throughput
timestamp: nanosecond precision
```

### 5.2 벡터 저장소 (Milvus)
```python
collection_schema = {
    "name": "documents",
    "fields": [
        {"name": "id", "type": "int64", "is_primary": True},
        {"name": "embedding", "type": "float_vector", "dim": 1536},
        {"name": "content", "type": "varchar", "max_length": 65535},
        {"name": "source", "type": "varchar", "max_length": 256},
        {"name": "metadata", "type": "json"},
    ],
    "index": {"type": "IVF_FLAT", "metric": "IP"}
}
```

### 5.3 그래프 데이터 (Neo4j)
```cypher
// 노드 예시
CREATE (m:Metric {name: "cpu_usage", unit: "%", source: "prometheus"})
CREATE (e:Event {timestamp: datetime(), severity: "critical"})
CREATE (s:Entity {name: "server-01", type: "server", status: "running"})

// 관계 예시
CREATE (m)-[:CAUSES {confidence: 0.85}]->(e)
CREATE (s)-[:BELONGS_TO]->(c:Entity {name: "cluster-1", type: "cluster"})
```

---

## 6. API Specification

### 6.1 REST Endpoints

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/query` | 자연어 질의 처리 |
| GET | `/api/v1/query/{id}` | 특정 질의 결과 조회 |
| POST | `/api/v1/query/{id}/feedback` | 피드백 제출 |
| GET | `/api/v1/sessions` | 세션 목록 조회 |
| GET | `/api/v1/graph/entities` | 엔티티 조회 |
| GET | `/api/v1/metrics` | 메트릭 데이터 조회 |
| GET | `/api/v1/health` | 서비스 상태 확인 |

### 6.2 Query API Request/Response

**Request:**
```json
{
  "query": "어제 CPU 급증 원인을 분석해줘",
  "session_id": "uuid-optional",
  "options": {
    "max_hops": 5,
    "include_reasoning": true,
    "stream": false
  }
}
```

**Response:**
```json
{
  "id": "query-uuid",
  "answer": "분석 결과...",
  "confidence": 0.87,
  "reasoning_chain": [
    {"step": 1, "action": "retrieve", "result": "..."},
    {"step": 2, "action": "reason", "result": "..."}
  ],
  "sources": [...],
  "graph_path": [...],
  "processing_time_ms": 2340
}
```

---

## 7. Deployment Architecture

### 7.1 Kubernetes Resources

| 컴포넌트 | 리소스 | 스케일링 |
|---------|--------|---------|
| API Gateway | 2 CPU, 4GB RAM | HPA: CPU 70% |
| Agent Workers | 4 CPU, 8GB RAM | HPA: Queue depth |
| InfluxDB | 8 CPU, 32GB RAM | 수직 확장 + Sharding |
| Neo4j | 8 CPU, 32GB RAM | Read Replicas |
| Milvus | 8 CPU, 32GB RAM | 수평 확장 (Segments) |

### 7.2 Environment Configuration

| 환경 | 목적 | 특징 |
|------|------|------|
| Development | 개발 및 단위 테스트 | 단일 인스턴스, 로컬 DB |
| Staging | 통합 테스트, QA | 프로덕션 유사, 축소된 리소스 |
| Production | 실서비스 | 고가용성, 자동 스케일링 |

---

## 8. Quality Gates

### 8.1 Pre-Implementation
- [ ] 스펙 문서 리뷰 완료
- [ ] 아키텍처 결정 문서화
- [ ] 데이터 모델 검증
- [ ] API 스펙 승인

### 8.2 Implementation
- [ ] 코드 리뷰 통과
- [ ] 단위 테스트 80%+
- [ ] 통합 테스트 통과
- [ ] 성능 벤치마크 충족

### 8.3 Pre-Deployment
- [ ] 보안 리뷰 완료
- [ ] 문서 업데이트
- [ ] 배포 체크리스트 완료
- [ ] Rollback 계획 확인

---

## 9. Risk Mitigation

| 위험 요소 | 영향도 | 대응 방안 |
|----------|--------|----------|
| LLM API 비용 초과 | 높음 | 캐싱 전략, 로컬 모델 병행 |
| 지식 그래프 구축 복잡도 | 중간 | 단계적 도입, 핵심 도메인 우선 |
| Multi-Agent 조율 실패 | 높음 | 단일 에이전트 폴백, 충분한 테스트 |
| 데이터 품질 이슈 | 중간 | 검증 파이프라인, 품질 모니터링 |

---

## 10. Implementation Phases

| Phase | 기간 | 주요 산출물 |
|-------|------|------------|
| Phase 1 | 1-4주 | 기본 RAG 파이프라인 + 시계열 DB 연동 |
| Phase 2 | 5-8주 | Self-RAG 자기 검증 메커니즘 구현 |
| Phase 3 | 9-14주 | GraphRAG 지식 그래프 통합 |
| Phase 4 | 15-20주 | Multi-Agent 오케스트레이션 |
| Phase 5 | 21-24주 | 통합 테스트 + 최적화 + 베타 출시 |

---

## Appendix A: Related Files

- `data-model.md` - 상세 데이터 모델
- `contracts/api-spec.json` - OpenAPI 스펙
- `research.md` - 기술 조사 결과
- `quickstart.md` - 빠른 시작 가이드
