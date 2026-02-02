"""Metrics API routes for time-series data access."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.data.models.schemas import MetricsQuery, MetricsResponse
from src.api.dependencies import get_timeseries_repo
from src.data.repositories.timeseries import TimeseriesRepository

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


@router.post("/query", response_model=MetricsResponse)
async def query_metrics(
    query: MetricsQuery,
    ts_repo: TimeseriesRepository = Depends(get_timeseries_repo),
) -> MetricsResponse:
    """시계열 메트릭 데이터 조회."""
    import time

    t0 = time.time()
    records = await ts_repo.query_metrics(
        measurement=query.measurement,
        tags=query.tags or None,
        fields=query.fields or None,
        start=query.start,
        stop=query.stop,
        aggregation=query.aggregation,
        group_by=query.group_by,
    )
    elapsed = (time.time() - t0) * 1000

    return MetricsResponse(
        measurement=query.measurement,
        records=records,
        count=len(records),
        query_time_ms=round(elapsed, 2),
    )
