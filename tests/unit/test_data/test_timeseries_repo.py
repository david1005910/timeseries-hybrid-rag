"""Tests for TimeseriesRepository (InfluxDB)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.repositories.timeseries import TimeseriesRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_influx_record(
    time_val: datetime,
    measurement: str,
    field: str,
    value: float,
    extra_tags: dict | None = None,
) -> MagicMock:
    """Create a mock InfluxDB record with the correct interface."""
    record = MagicMock()
    record.get_time.return_value = time_val
    record.get_measurement.return_value = measurement
    record.get_field.return_value = field
    record.get_value.return_value = value
    base_values = {
        "_time": time_val,
        "_value": value,
        "_field": field,
        "_measurement": measurement,
        "result": "result",
        "table": 0,
    }
    if extra_tags:
        base_values.update(extra_tags)
    record.values = base_values
    return record


def _make_table(records: list[MagicMock]) -> MagicMock:
    table = MagicMock()
    table.records = records
    return table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.influxdb_url = "http://localhost:8086"
    settings.influxdb_token = "test-token"
    settings.influxdb_org = "test-org"
    settings.influxdb_bucket = "test-bucket"
    return settings


@pytest.fixture
def repo(mock_settings):
    """Create a TimeseriesRepository with all InfluxDB dependencies mocked."""
    with (
        patch("src.data.repositories.timeseries.get_settings", return_value=mock_settings),
        patch("src.data.repositories.timeseries.InfluxDBClient") as MockClient,
    ):
        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance
        mock_client_instance.query_api.return_value = MagicMock()

        repository = TimeseriesRepository()
        # Expose internal mocks for test assertions
        repository._mock_client = mock_client_instance
        yield repository


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueryMetrics:
    async def test_basic_query(self, repo: TimeseriesRepository) -> None:
        """Basic measurement-only query returns parsed records."""
        ts = datetime(2025, 1, 31, 14, 0, 0, tzinfo=timezone.utc)
        record = _make_influx_record(ts, "cpu", "usage_percent", 72.5)
        repo._query_api.query.return_value = [_make_table([record])]

        results = await repo.query_metrics(measurement="cpu")

        assert len(results) == 1
        assert results[0]["measurement"] == "cpu"
        assert results[0]["field"] == "usage_percent"
        assert results[0]["value"] == 72.5
        assert results[0]["time"] == ts.isoformat()

        # Verify the flux query was sent to the correct org
        call_args = repo._query_api.query.call_args
        assert call_args.kwargs["org"] == "test-org"
        flux = call_args.args[0]
        assert '"cpu"' in flux

    async def test_query_with_tags(self, repo: TimeseriesRepository) -> None:
        """Tag filters are appended to the Flux query."""
        repo._query_api.query.return_value = []

        await repo.query_metrics(
            measurement="cpu",
            tags={"host": "server-01", "region": "us-east"},
        )

        flux = repo._query_api.query.call_args.args[0]
        assert 'r.host == "server-01"' in flux
        assert 'r.region == "us-east"' in flux

    async def test_query_with_fields(self, repo: TimeseriesRepository) -> None:
        """Field filters are appended as an OR filter clause."""
        repo._query_api.query.return_value = []

        await repo.query_metrics(
            measurement="memory",
            fields=["used", "cached"],
        )

        flux = repo._query_api.query.call_args.args[0]
        assert 'r._field == "used"' in flux
        assert 'r._field == "cached"' in flux
        assert " or " in flux

    async def test_query_with_aggregation_and_group_by(self, repo: TimeseriesRepository) -> None:
        """Aggregation window is added only when both aggregation and group_by are provided."""
        repo._query_api.query.return_value = []

        await repo.query_metrics(
            measurement="cpu",
            aggregation="mean",
            group_by="5m",
        )

        flux = repo._query_api.query.call_args.args[0]
        assert "aggregateWindow" in flux
        assert "every: 5m" in flux
        assert "fn: mean" in flux

    async def test_query_aggregation_without_group_by_skipped(
        self, repo: TimeseriesRepository
    ) -> None:
        """Aggregation alone (without group_by) should not produce an aggregateWindow clause."""
        repo._query_api.query.return_value = []

        await repo.query_metrics(measurement="cpu", aggregation="mean")

        flux = repo._query_api.query.call_args.args[0]
        assert "aggregateWindow" not in flux

    async def test_empty_results(self, repo: TimeseriesRepository) -> None:
        """Empty table list returns an empty list."""
        repo._query_api.query.return_value = []

        results = await repo.query_metrics(measurement="disk")

        assert results == []

    async def test_tags_extracted_excluding_internal_keys(
        self, repo: TimeseriesRepository
    ) -> None:
        """Tags dict in the result excludes InfluxDB internal keys."""
        ts = datetime(2025, 1, 31, 14, 0, 0, tzinfo=timezone.utc)
        record = _make_influx_record(
            ts, "cpu", "usage_percent", 80.0, extra_tags={"host": "web-1"}
        )
        repo._query_api.query.return_value = [_make_table([record])]

        results = await repo.query_metrics(measurement="cpu")

        tags = results[0]["tags"]
        assert tags == {"host": "web-1"}
        # Internal keys must not leak into tags
        for key in ("_time", "_value", "_field", "_measurement", "result", "table"):
            assert key not in tags


class TestWriteMetrics:
    async def test_write_metrics(self, repo: TimeseriesRepository) -> None:
        """write_metrics delegates to the InfluxDB write API."""
        mock_write_api = MagicMock()
        repo._mock_client.write_api.return_value = mock_write_api

        await repo.write_metrics(
            measurement="cpu",
            tags={"host": "server-01"},
            fields={"usage_percent": 65.3},
        )

        mock_write_api.write.assert_called_once()
        call_kwargs = mock_write_api.write.call_args.kwargs
        assert call_kwargs["bucket"] == "test-bucket"
        assert call_kwargs["org"] == "test-org"
        point = call_kwargs["record"]
        assert point["measurement"] == "cpu"
        assert point["tags"] == {"host": "server-01"}
        assert point["fields"] == {"usage_percent": 65.3}
