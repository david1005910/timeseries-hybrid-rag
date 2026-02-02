"""Neo4j Repository for knowledge graph operations."""
from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncSession

from src.config.constants import MAX_HOPS, NodeType, RelationshipType
from src.config.settings import get_settings
from src.data.models.entities import GraphNode, GraphRelationship
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRepository:
    """지식 그래프를 위한 Neo4j Repository."""

    def __init__(self) -> None:
        settings = get_settings()
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    async def _get_session(self) -> AsyncSession:
        return self._driver.session()

    # --- Node CRUD ---
    async def create_node(self, node: GraphNode) -> GraphNode:
        """노드 생성."""
        async with await self._get_session() as session:
            query = f"""
            CREATE (n:{node.type} {{
                id: $id, name: $name, created_at: datetime()
            }})
            SET n += $properties
            RETURN n
            """
            await session.run(
                query,
                id=node.id,
                name=node.name,
                properties=node.properties,
            )
            logger.info("graph_node_created", node_type=node.type, node_id=node.id)
            return node

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """노드 조회."""
        async with await self._get_session() as session:
            result = await session.run(
                "MATCH (n {id: $id}) RETURN n, labels(n) as labels",
                id=node_id,
            )
            record = await result.single()
            if record:
                node_data = dict(record["n"])
                node_data["type"] = record["labels"][0] if record["labels"] else "unknown"
                return node_data
            return None

    async def delete_node(self, node_id: str) -> bool:
        """노드 및 관련 관계 삭제."""
        async with await self._get_session() as session:
            result = await session.run(
                "MATCH (n {id: $id}) DETACH DELETE n RETURN count(n) as deleted",
                id=node_id,
            )
            record = await result.single()
            return record is not None and record["deleted"] > 0

    # --- Relationship CRUD ---
    async def create_relationship(self, rel: GraphRelationship) -> GraphRelationship:
        """관계 생성."""
        async with await self._get_session() as session:
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[r:{rel.type.upper()} {{
                confidence: $confidence,
                created_at: datetime()
            }}]->(b)
            SET r += $properties
            RETURN r
            """
            await session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                confidence=rel.confidence,
                properties=rel.properties,
            )
            logger.info("graph_rel_created", rel_type=rel.type, source=rel.source_id, target=rel.target_id)
            return rel

    # --- Multi-hop Traversal ---
    async def traverse(
        self,
        start_node_id: str,
        max_hops: int = MAX_HOPS,
        relationship_types: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """다중 홉 그래프 탐색.

        Args:
            start_node_id: 시작 노드 ID
            max_hops: 최대 탐색 깊이 (기본 5)
            relationship_types: 탐색할 관계 유형 (None이면 전부)
            min_confidence: 최소 신뢰도 필터

        Returns:
            탐색된 경로 리스트
        """
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(r.upper() for r in relationship_types)
            rel_filter = f":{rel_types}"

        async with await self._get_session() as session:
            query = f"""
            MATCH path = (start {{id: $start_id}})-[r{rel_filter}*1..{max_hops}]->(end)
            WHERE ALL(rel IN relationships(path) WHERE rel.confidence >= $min_confidence)
            RETURN path,
                   [n IN nodes(path) | {{id: n.id, name: n.name, labels: labels(n)}}] as nodes,
                   [r IN relationships(path) | {{type: type(r), confidence: r.confidence}}] as rels,
                   length(path) as hops
            ORDER BY hops ASC
            LIMIT 50
            """
            result = await session.run(
                query,
                start_id=start_node_id,
                min_confidence=min_confidence,
            )
            paths = []
            async for record in result:
                paths.append({
                    "nodes": record["nodes"],
                    "relationships": record["rels"],
                    "hops": record["hops"],
                })
            logger.info("graph_traverse", start=start_node_id, max_hops=max_hops, paths_found=len(paths))
            return paths

    async def find_causal_chain(
        self,
        start_node_id: str,
        end_node_id: str | None = None,
        max_hops: int = MAX_HOPS,
    ) -> list[dict[str, Any]]:
        """인과관계 체인 탐색 (CAUSES 관계만)."""
        return await self.traverse(
            start_node_id=start_node_id,
            max_hops=max_hops,
            relationship_types=["causes"],
            min_confidence=0.5,
        )

    async def get_community_summary(self, community_id: str) -> dict[str, Any] | None:
        """커뮤니티 요약 조회."""
        async with await self._get_session() as session:
            result = await session.run(
                """
                MATCH (n) WHERE n.community_id = $community_id
                RETURN collect({id: n.id, name: n.name, labels: labels(n)}) as members,
                       count(n) as size
                """,
                community_id=community_id,
            )
            record = await result.single()
            if record:
                return {"community_id": community_id, "members": record["members"], "size": record["size"]}
            return None

    async def close(self) -> None:
        await self._driver.close()
