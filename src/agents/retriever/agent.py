"""Retriever Agent: multi-source parallel retrieval."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from src.data.repositories.graph import GraphRepository
from src.data.repositories.timeseries import TimeseriesRepository
from src.data.repositories.vector import VectorRepository
from src.llm.embeddings import EmbeddingService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RetrieverAgent(BaseAgent):
    """다중 데이터 소스에서 관련 정보를 병렬 검색하는 에이전트.

    Sources: 시계열(InfluxDB), 벡터(Milvus), 그래프(Neo4j)
    """

    def __init__(
        self,
        timeseries_repo: TimeseriesRepository,
        vector_repo: VectorRepository,
        graph_repo: GraphRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        super().__init__(name="retriever")
        self._timeseries = timeseries_repo
        self._vector = vector_repo
        self._graph = graph_repo
        self._embedding = embedding_service

    async def execute(self, context: AgentContext) -> AgentResult:
        """다중 소스 병렬 검색 수행."""
        query = context.query
        options = context.options
        retrieval_plan = options.get("retrieval_plan", {})

        # Determine which sources to search
        sources_to_search = retrieval_plan.get("sources", ["vector", "graph", "timeseries"])
        top_k = options.get("top_k", 10)

        tasks: dict[str, asyncio.Task[Any]] = {}

        if "vector" in sources_to_search:
            tasks["vector"] = asyncio.create_task(self._search_vector(query, top_k))

        if "graph" in sources_to_search:
            node_id = retrieval_plan.get("start_node_id")
            max_hops = options.get("max_hops", 5)
            tasks["graph"] = asyncio.create_task(self._search_graph(query, node_id, max_hops))

        if "timeseries" in sources_to_search:
            ts_params = retrieval_plan.get("timeseries_params", {})
            tasks["timeseries"] = asyncio.create_task(self._search_timeseries(ts_params))

        # Await all tasks
        results: dict[str, Any] = {}
        for source_name, task in tasks.items():
            try:
                results[source_name] = await task
            except Exception as e:
                logger.warning("retrieval_source_failed", source=source_name, error=str(e))
                results[source_name] = {"error": str(e), "items": []}

        # Merge and rank results
        merged = self._merge_results(results)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "documents": merged,
                "source_counts": {k: len(v.get("items", [])) for k, v in results.items()},
                "total_count": len(merged),
            },
        )

    async def _search_vector(self, query: str, top_k: int) -> dict[str, Any]:
        """벡터 유사도 검색."""
        embedding = await self._embedding.embed(query)
        hits = await self._vector.search(query_embedding=embedding, top_k=top_k)
        items = [
            {
                "id": hit["id"],
                "content": hit["content"],
                "source": hit["source"],
                "source_type": "vector",
                "relevance_score": hit["score"],
                "metadata": json.loads(hit.get("metadata_json", "{}")) if hit.get("metadata_json") else {},
            }
            for hit in hits
        ]
        return {"items": items}

    async def _search_graph(
        self, query: str, start_node_id: str | None, max_hops: int
    ) -> dict[str, Any]:
        """그래프 탐색 검색."""
        if not start_node_id:
            return {"items": []}

        paths = await self._graph.traverse(
            start_node_id=start_node_id,
            max_hops=max_hops,
        )
        items = [
            {
                "id": f"path-{i}",
                "content": self._format_path(path),
                "source": "knowledge_graph",
                "source_type": "graph",
                "relevance_score": 0.8,
                "metadata": {"hops": path["hops"], "nodes": path["nodes"]},
            }
            for i, path in enumerate(paths)
        ]
        return {"items": items}

    async def _search_timeseries(self, params: dict[str, Any]) -> dict[str, Any]:
        """시계열 데이터 검색."""
        if not params.get("measurement"):
            return {"items": []}

        records = await self._timeseries.query_metrics(
            measurement=params["measurement"],
            tags=params.get("tags"),
            fields=params.get("fields"),
            start=params.get("start", "-1h"),
            stop=params.get("stop", "now()"),
            aggregation=params.get("aggregation"),
            group_by=params.get("group_by"),
        )
        items = [
            {
                "id": f"ts-{i}",
                "content": json.dumps(record, ensure_ascii=False),
                "source": params["measurement"],
                "source_type": "timeseries",
                "relevance_score": 0.7,
                "metadata": record,
            }
            for i, record in enumerate(records)
        ]
        return {"items": items}

    def _merge_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        """다중 소스 결과 병합 및 랭킹."""
        all_items: list[dict[str, Any]] = []
        for source_data in results.values():
            if isinstance(source_data, dict) and "items" in source_data:
                all_items.extend(source_data["items"])

        # Sort by relevance score
        all_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_items

    @staticmethod
    def _format_path(path: dict[str, Any]) -> str:
        """그래프 경로를 자연어 설명으로 변환."""
        nodes = path.get("nodes", [])
        rels = path.get("relationships", [])
        parts = []
        for i, node in enumerate(nodes):
            parts.append(f"[{node.get('name', 'unknown')}]")
            if i < len(rels):
                rel = rels[i]
                parts.append(f" --{rel.get('type', '?')}--> ")
        return "".join(parts)
