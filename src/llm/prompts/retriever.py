"""Retriever agent prompt templates."""
from __future__ import annotations

QUERY_ANALYSIS_PROMPT = """다음 사용자 질의를 분석하여 검색 전략을 결정하세요.

질의: {query}

분석 결과를 JSON으로 응답하세요:
{{
    "search_type": "시계열|문서|그래프|복합",
    "keywords": ["핵심 키워드"],
    "time_range": {{"start": "-1h", "stop": "now()"}} 또는 null,
    "measurements": ["관련 메트릭 이름"] 또는 [],
    "entity_names": ["관련 엔티티 이름"] 또는 [],
    "filters": {{}}
}}

JSON만 응답하세요."""
