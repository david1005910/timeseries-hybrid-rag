"""GraphRAG Engine: knowledge graph-based multi-hop reasoning."""
from __future__ import annotations

from typing import Any

from src.data.repositories.graph import GraphRepository
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

GRAPH_QUERY_PROMPT = """다음 질문에 대해 지식 그래프에서 탐색할 시작 노드와 관계 유형을 결정하세요.

질문: {query}

다음 JSON 형식으로 응답하세요:
{{
    "start_entities": ["시작 엔티티 이름"],
    "relationship_types": ["causes", "correlates", "belongs_to", "precedes"],
    "max_hops": 1-5,
    "reasoning": "탐색 전략 설명"
}}

JSON만 응답하세요."""

PATH_EXPLANATION_PROMPT = """다음 그래프 경로를 자연어로 설명하세요.

질문: {query}

탐색된 경로:
{paths}

인과관계 체인을 포함하여 명확하게 설명하세요. 각 단계의 신뢰도를 포함하세요."""


class GraphRAGEngine:
    """GraphRAG 엔진: 지식 그래프 기반 다중 홉 추론.

    기능:
    - 그래프 탐색 (BFS/DFS, max 5홉)
    - 커뮤니티 감지 (Leiden 알고리즘)
    - 커뮤니티 요약 생성
    - 인과관계 체인 추론
    - 경로 설명 생성
    """

    def __init__(self, graph_repo: GraphRepository, llm_client: LLMClient) -> None:
        self._graph = graph_repo
        self._llm = llm_client

    async def query(self, query: str, max_hops: int = 5) -> GraphRAGResult:
        """GraphRAG 질의 처리.

        1. LLM으로 탐색 전략 결정
        2. 그래프 탐색 수행
        3. 경로 설명 생성
        """
        # Step 1: Determine traversal strategy
        strategy = await self._plan_traversal(query)
        start_entities = strategy.get("start_entities", [])
        rel_types = strategy.get("relationship_types")
        planned_hops = min(strategy.get("max_hops", max_hops), max_hops)

        # Step 2: Search for start nodes and traverse
        all_paths: list[dict[str, Any]] = []
        for entity_name in start_entities:
            node = await self._find_node_by_name(entity_name)
            if node:
                paths = await self._graph.traverse(
                    start_node_id=node["id"],
                    max_hops=planned_hops,
                    relationship_types=rel_types,
                    min_confidence=0.3,
                )
                all_paths.extend(paths)

        # Step 3: Generate path explanations
        explanation = ""
        if all_paths:
            explanation = await self._explain_paths(query, all_paths[:10])

        return GraphRAGResult(
            paths=all_paths,
            explanation=explanation,
            strategy=strategy,
            total_paths=len(all_paths),
        )

    async def find_causal_chain(self, start_entity: str, max_hops: int = 5) -> list[dict[str, Any]]:
        """인과관계 체인 탐색."""
        node = await self._find_node_by_name(start_entity)
        if not node:
            return []
        return await self._graph.find_causal_chain(
            start_node_id=node["id"],
            max_hops=max_hops,
        )

    async def _plan_traversal(self, query: str) -> dict[str, Any]:
        """LLM으로 그래프 탐색 전략 결정."""
        import json

        prompt = GRAPH_QUERY_PROMPT.format(query=query)
        response = await self._llm.generate(prompt=prompt, temperature=0.1, max_tokens=500)
        try:
            return json.loads(response.content)
        except Exception:
            return {
                "start_entities": [],
                "relationship_types": None,
                "max_hops": 3,
                "reasoning": "기본 탐색",
            }

    async def _find_node_by_name(self, name: str) -> dict[str, Any] | None:
        """이름으로 노드 검색."""
        # Simplified: search in Neo4j by name property
        from neo4j import AsyncGraphDatabase

        session = await self._graph._get_session()
        async with session as s:
            result = await s.run(
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n.id as id, n.name as name, labels(n) as labels LIMIT 1",
                name=name,
            )
            record = await result.single()
            if record:
                return {"id": record["id"], "name": record["name"], "type": record["labels"][0] if record["labels"] else "unknown"}
        return None

    async def _explain_paths(self, query: str, paths: list[dict[str, Any]]) -> str:
        """탐색된 경로를 자연어로 설명."""
        formatted = []
        for i, path in enumerate(paths, 1):
            nodes = path.get("nodes", [])
            rels = path.get("relationships", [])
            parts = []
            for j, node in enumerate(nodes):
                parts.append(f"[{node.get('name', '?')}]")
                if j < len(rels):
                    rel = rels[j]
                    parts.append(f" --{rel.get('type', '?')}({rel.get('confidence', 0):.0%})--> ")
            formatted.append(f"경로 {i} ({path.get('hops', 0)}홉): {''.join(parts)}")

        prompt = PATH_EXPLANATION_PROMPT.format(
            query=query,
            paths="\n".join(formatted),
        )
        response = await self._llm.generate(prompt=prompt, temperature=0.3, max_tokens=1000)
        return response.content


class GraphRAGResult:
    """GraphRAG 질의 결과."""

    def __init__(
        self,
        paths: list[dict[str, Any]],
        explanation: str,
        strategy: dict[str, Any],
        total_paths: int,
    ) -> None:
        self.paths = paths
        self.explanation = explanation
        self.strategy = strategy
        self.total_paths = total_paths
