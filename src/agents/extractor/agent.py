"""Extractor Agent: entity and relationship extraction from text."""
from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentContext, AgentResult, AgentStatus, BaseAgent
from src.data.models.entities import GraphNode, GraphRelationship
from src.data.repositories.graph import GraphRepository
from src.llm.client import LLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXTRACTION_PROMPT = """다음 텍스트에서 엔티티와 관계를 추출하세요.

텍스트:
{text}

다음 JSON 형식으로 응답하세요:
{{
    "entities": [
        {{"name": "엔티티명", "type": "metric|event|entity|document|concept", "properties": {{}}}}
    ],
    "relationships": [
        {{"source": "소스 엔티티명", "target": "타겟 엔티티명", "type": "causes|correlates|belongs_to|precedes", "confidence": 0.0-1.0}}
    ]
}}

JSON만 응답하세요."""


class ExtractorAgent(BaseAgent):
    """텍스트에서 엔티티와 관계를 추출하는 에이전트."""

    def __init__(self, llm_client: LLMClient, graph_repo: GraphRepository) -> None:
        super().__init__(name="extractor")
        self._llm = llm_client
        self._graph = graph_repo

    async def execute(self, context: AgentContext) -> AgentResult:
        """엔티티/관계 추출 및 그래프 저장."""
        documents = context.previous_results.get("documents", [])
        if not documents:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data={"entities": [], "relationships": []},
            )

        all_entities: list[dict[str, Any]] = []
        all_relationships: list[dict[str, Any]] = []

        for doc in documents[:10]:  # Limit to 10 documents
            content = doc.get("content", "")
            if not content or len(content) < 20:
                continue

            try:
                extraction = await self._extract_from_text(content)
                entities = extraction.get("entities", [])
                relationships = extraction.get("relationships", [])

                # Store in graph
                entity_map: dict[str, str] = {}
                for entity_data in entities:
                    node = GraphNode(
                        type=entity_data.get("type", "concept"),
                        name=entity_data["name"],
                        properties=entity_data.get("properties", {}),
                    )
                    await self._graph.create_node(node)
                    entity_map[node.name] = node.id
                    all_entities.append({"id": node.id, "name": node.name, "type": node.type})

                for rel_data in relationships:
                    source_id = entity_map.get(rel_data["source"])
                    target_id = entity_map.get(rel_data["target"])
                    if source_id and target_id:
                        rel = GraphRelationship(
                            source_id=source_id,
                            target_id=target_id,
                            type=rel_data.get("type", "correlates"),
                            confidence=rel_data.get("confidence", 0.5),
                        )
                        await self._graph.create_relationship(rel)
                        all_relationships.append({
                            "source": rel_data["source"],
                            "target": rel_data["target"],
                            "type": rel.type,
                            "confidence": rel.confidence,
                        })

            except Exception as e:
                logger.warning("extraction_failed", error=str(e), content_len=len(content))

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "entities": all_entities,
                "relationships": all_relationships,
                "entity_count": len(all_entities),
                "relationship_count": len(all_relationships),
            },
        )

    async def _extract_from_text(self, text: str) -> dict[str, Any]:
        """LLM을 사용하여 텍스트에서 엔티티/관계 추출."""
        prompt = EXTRACTION_PROMPT.format(text=text[:3000])
        response = await self._llm.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000,
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            return {"entities": [], "relationships": []}
