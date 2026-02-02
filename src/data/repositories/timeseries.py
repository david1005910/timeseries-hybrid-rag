"""InfluxDB Repository for time-series data operations."""
from __future__ import annotations

import time
from typing import Any

from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TimeseriesRepository:
    """시계열 데이터 저장/조회를 위한 InfluxDB Repository."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            timeout=10_000,
        )
        self._org = settings.influxdb_org
        self._bucket = settings.influxdb_bucket
        self._query_api: QueryApi = self._client.query_api()

    async def query_metrics(
        self,
        measurement: str,
        tags: dict[str, str] | None = None,
        fields: list[str] | None = None,
        start: str = "-1h",
        stop: str = "now()",
        aggregation: str | None = None,
        group_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """시계열 메트릭 데이터 조회.

        Args:
            measurement: 측정 이름 (예: cpu, memory)
            tags: 필터링할 태그 (예: {"host": "server-01"})
            fields: 조회할 필드 목록
            start: 시작 시간 (Flux 표현, 예: "-1h", "-7d")
            stop: 종료 시간
            aggregation: 집계 함수 (mean, max, min, sum)
            group_by: 그룹핑 간격 (예: "5m", "1h")

        Returns:
            조회된 레코드 리스트
        """
        t0 = time.time()

        flux = f"""
from(bucket: "{self._bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{measurement}")
"""
        if tags:
            for tag_key, tag_value in tags.items():
                flux += f'  |> filter(fn: (r) => r.{tag_key} == "{tag_value}")\n'

        if fields:
            field_filter = " or ".join(f'r._field == "{f}"' for f in fields)
            flux += f"  |> filter(fn: (r) => {field_filter})\n"

        if aggregation and group_by:
            flux += f'  |> aggregateWindow(every: {group_by}, fn: {aggregation}, createEmpty: false)\n'

        flux += '  |> yield(name: "result")'

        tables = self._query_api.query(flux, org=self._org)

        records: list[dict[str, Any]] = []
        for table in tables:
            for record in table.records:
                records.append({
                    "time": record.get_time().isoformat(),
                    "measurement": record.get_measurement(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "tags": {k: v for k, v in record.values.items() if k not in ("_time", "_value", "_field", "_measurement", "result", "table")},
                })

        elapsed = (time.time() - t0) * 1000
        logger.info("timeseries_query", measurement=measurement, record_count=len(records), elapsed_ms=round(elapsed, 2))
        return records

    async def write_metrics(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, float],
    ) -> None:
        """시계열 메트릭 데이터 기록."""
        write_api = self._client.write_api()
        point = {
            "measurement": measurement,
            "tags": tags,
            "fields": fields,
        }
        write_api.write(bucket=self._bucket, org=self._org, record=point)
        logger.info("timeseries_write", measurement=measurement, tags=tags)

    def close(self) -> None:
        self._client.close()
