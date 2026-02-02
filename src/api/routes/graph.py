"""Graph API routes for knowledge graph operations."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_graph_repo
from src.data.repositories.graph import GraphRepository

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


@router.get("/entities")
async def list_entities(
    node_type: str | None = Query(None, description="Filter by node type"),
    limit: int = Query(50, ge=1, le=200),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> list[dict[str, Any]]:
    """그래프 엔티티 조회."""
    # Simplified: list nodes from Neo4j
    from neo4j import AsyncGraphDatabase

    session = await graph_repo._get_session()
    async with session as s:
        query = "MATCH (n) "
        if node_type:
            query += f"WHERE '{node_type}' IN labels(n) "
        query += "RETURN n.id as id, n.name as name, labels(n) as labels LIMIT $limit"
        result = await s.run(query, limit=limit)
        entities = []
        async for record in result:
            entities.append({
                "id": record["id"],
                "name": record["name"],
                "type": record["labels"][0] if record["labels"] else "unknown",
            })
    return entities


@router.get("/entities/{node_id}/traverse")
async def traverse_entity(
    node_id: str,
    max_hops: int = Query(3, ge=1, le=5),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    graph_repo: GraphRepository = Depends(get_graph_repo),
) -> list[dict[str, Any]]:
    """엔티티에서 다중 홉 탐색."""
    paths = await graph_repo.traverse(
        start_node_id=node_id,
        max_hops=max_hops,
        min_confidence=min_confidence,
    )
    return paths
