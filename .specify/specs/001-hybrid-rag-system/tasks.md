# Task Breakdown

## 001-hybrid-rag-system

**Branch:** 001-hybrid-rag-system  
**Date:** 2025-02-02  
**Plan:** [plan.md](./plan.md)  
**Status:** Ready for Implementation

---

## Legend

- `[P]` - Parallelizable (병렬 실행 가능)
- `[ ]` - Not started
- `[x]` - Completed
- `depends-on: T###` - 의존성

---

## Phase 1: Project Setup (Week 1-2)

### T001: 프로젝트 초기화
**Priority:** P0 | **Est:** 4h | **Files:** `pyproject.toml`, `.env.example`

- [ ] T001.1: Poetry 프로젝트 초기화
- [ ] T001.2: 핵심 의존성 추가 (FastAPI, Pydantic, httpx, anthropic)
- [ ] T001.3: 개발 의존성 추가 (pytest, black, isort, pylint, mypy)
- [ ] T001.4: `.env.example` 환경 변수 템플릿
- [ ] T001.5: README.md 초기 문서

**Done:** `poetry install` 성공, `poetry run pytest` 실행 가능

---

### T002: [P] 디렉토리 구조 생성
**Priority:** P0 | **Est:** 2h | **Depends:** T001

- [ ] T002.1: `src/api/`, `src/agents/`, `src/reasoning/`, `src/data/`, `src/llm/` 생성
- [ ] T002.2: 모든 `__init__.py` 파일 생성
- [ ] T002.3: `tests/unit/`, `tests/integration/` 구조

---

### T003: [P] 설정 관리 구현
**Priority:** P0 | **Est:** 3h | **Files:** `src/config/settings.py`

- [ ] T003.1: Pydantic Settings 클래스 (DB URLs, API Keys, App Config)
- [ ] T003.2: 환경별 설정 분리
- [ ] T003.3: 설정 유효성 검증 테스트

---

### T004: Docker 환경 구성
**Priority:** P0 | **Est:** 4h | **Files:** `docker/docker-compose.yml`

- [ ] T004.1: InfluxDB, Neo4j, Milvus, PostgreSQL, Redis 컨테이너
- [ ] T004.2: 볼륨 및 네트워크 설정
- [ ] T004.3: 헬스체크 설정

**Done:** `docker-compose up` 전체 환경 기동

---

### T005: [P] 로깅 시스템
**Priority:** P1 | **Est:** 2h | **Files:** `src/utils/logging.py`

- [ ] T005.1: 구조화된 JSON 로깅
- [ ] T005.2: Correlation ID 지원

---

## Phase 2: Data Layer (Week 2-4)

### T006: InfluxDB Repository
**Priority:** P0 | **Est:** 6h | **Depends:** T003, T004

- [ ] T006.1: InfluxDB 클라이언트 초기화
- [ ] T006.2: `query_metrics()` - 시계열 쿼리
- [ ] T006.3: 집계 함수 (mean, max, min)
- [ ] T006.4: 연결 풀 관리
- [ ] T006.5: 단위 테스트

**Done:** 1초 이내 1000 포인트 조회

---

### T007: [P] Neo4j Repository
**Priority:** P0 | **Est:** 8h | **Depends:** T003, T004

- [ ] T007.1: Neo4j 드라이버 설정
- [ ] T007.2: 노드 CRUD (Metric, Event, Entity, Document, Concept)
- [ ] T007.3: 관계 CRUD (CAUSES, CORRELATES, BELONGS_TO, PRECEDES)
- [ ] T007.4: `traverse()` - 다중 홉 탐색 (max 5홉)
- [ ] T007.5: Cypher 쿼리 빌더
- [ ] T007.6: 단위 테스트

---

### T008: [P] Milvus Repository
**Priority:** P0 | **Est:** 6h | **Depends:** T003, T004

- [ ] T008.1: Milvus 클라이언트 초기화
- [ ] T008.2: 컬렉션 생성 (1536차원)
- [ ] T008.3: `insert()` - 벡터 삽입
- [ ] T008.4: `search()` - Top-K 유사도 검색
- [ ] T008.5: 단위 테스트

---

### T009: [P] PostgreSQL Repository
**Priority:** P1 | **Est:** 4h | **Depends:** T003, T004

- [ ] T009.1: SQLAlchemy async 설정
- [ ] T009.2: Session, Feedback 모델
- [ ] T009.3: 세션 CRUD
- [ ] T009.4: 마이그레이션 스크립트

---

## Phase 3: LLM Integration (Week 3-5)

### T010: LLM 클라이언트
**Priority:** P0 | **Est:** 6h | **Depends:** T003

- [ ] T010.1: Anthropic Claude 클라이언트
- [ ] T010.2: OpenAI 클라이언트 (fallback)
- [ ] T010.3: 재시도 로직 (exponential backoff)
- [ ] T010.4: Rate limiting
- [ ] T010.5: 토큰 사용량 추적
- [ ] T010.6: 스트리밍 응답 지원

---

### T011: [P] 임베딩 생성기
**Priority:** P0 | **Est:** 4h | **Depends:** T010

- [ ] T011.1: OpenAI text-embedding-3-large 통합
- [ ] T011.2: 배치 임베딩 생성
- [ ] T011.3: 캐싱 레이어

---

### T012: [P] 프롬프트 템플릿
**Priority:** P1 | **Est:** 4h | **Files:** `src/llm/prompts/`

- [ ] T012.1: 플래너/리트리버/리즈너/검증자 프롬프트
- [ ] T012.2: Jinja2 템플릿 시스템

---

## Phase 4: Agent Layer (Week 5-8)

### T013: BaseAgent 추상 클래스
**Priority:** P0 | **Est:** 4h | **Files:** `src/agents/base.py`

```python
class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        pass
```

- [ ] T013.1: BaseAgent, AgentContext, AgentResult 정의
- [ ] T013.2: 에러 핸들링/메트릭 수집 믹스인

---

### T014: Retriever Agent
**Priority:** P0 | **Est:** 8h | **Depends:** T006, T007, T008, T013

- [ ] T014.1: RetrieverAgent 클래스
- [ ] T014.2: 시계열/벡터/그래프 검색 전략
- [ ] T014.3: 병렬 검색 조율
- [ ] T014.4: 결과 병합 및 랭킹
- [ ] T014.5: 통합 테스트

---

### T015: [P] Extractor Agent
**Priority:** P1 | **Est:** 6h | **Depends:** T010, T013

- [ ] T015.1: 엔티티/관계 추출 로직
- [ ] T015.2: 정보 정규화

---

### T016: Reasoner Agent
**Priority:** P0 | **Est:** 8h | **Depends:** T010, T013, T014

- [ ] T016.1: ReasonerAgent 클래스
- [ ] T016.2: Chain-of-Thought 추론
- [ ] T016.3: 그래프 탐색 통합
- [ ] T016.4: 신뢰도 점수 계산
- [ ] T016.5: 통합 테스트

---

### T017: Validator Agent (Self-RAG)
**Priority:** P0 | **Est:** 10h | **Depends:** T010, T013, T016

- [ ] T017.1: ValidatorAgent 클래스
- [ ] T017.2: Reflection Tokens 구현
  - `[Retrieve]` - 검색 필요 여부
  - `[IsREL]` - 문서 관련성
  - `[IsSUP]` - 증거 뒷받침
  - `[IsUSE]` - 전체 유용성
- [ ] T017.3: 자동 재검색/재생성 로직
- [ ] T017.4: 통합 테스트

**Done:** 환각 발생률 10% 이하

---

## Phase 5: Reasoning Layer (Week 6-10)

### T018: GraphRAG Engine
**Priority:** P0 | **Est:** 12h | **Depends:** T007, T011

- [ ] T018.1: GraphRAG Engine 클래스
- [ ] T018.2: 그래프 탐색 알고리즘 (BFS/DFS)
- [ ] T018.3: Leiden 커뮤니티 감지
- [ ] T018.4: 커뮤니티 요약 생성
- [ ] T018.5: 다중 홉 추론 (max 5홉)
- [ ] T018.6: 경로 설명 생성
- [ ] T018.7: 통합 테스트

**Done:** 5홉 탐색 정확도 85%

---

### T019: [P] Self-RAG Verifier
**Priority:** P0 | **Est:** 8h | **Depends:** T017

- [ ] T019.1: 검증 워크플로우
- [ ] T019.2: 증거 매핑 로직
- [ ] T019.3: 신뢰도 스코어링
- [ ] T019.4: 검증 실패 시 재시도

---

### T020: [P] CoT Processor
**Priority:** P1 | **Est:** 6h | **Depends:** T010

- [ ] T020.1: 단계별 추론 생성
- [ ] T020.2: 중간 결과 추적
- [ ] T020.3: 추론 체인 시각화 데이터

---

*Continued in tasks-part2.md*
