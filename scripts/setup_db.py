"""Database initialization script.

Creates all required tables and initial schema for:
- PostgreSQL (users, sessions, messages, feedback)
- InfluxDB (metrics bucket)
- Neo4j (constraints and indexes)
- Milvus (document collection)

Usage:
    poetry run python scripts/setup_db.py
"""
from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.utils.logging import setup_logging, get_logger


async def setup_postgresql() -> None:
    """PostgreSQL 테이블 생성."""
    logger = get_logger("setup_db")
    try:
        from src.data.repositories.session import init_db
        await init_db()
        logger.info("postgresql_setup_complete")
    except Exception as e:
        logger.error("postgresql_setup_failed", error=str(e))


async def setup_influxdb() -> None:
    """InfluxDB 버킷 확인."""
    logger = get_logger("setup_db")
    settings = get_settings()
    try:
        from influxdb_client import InfluxDBClient
        client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
        )
        buckets_api = client.buckets_api()
        bucket = buckets_api.find_bucket_by_name(settings.influxdb_bucket)
        if bucket:
            logger.info("influxdb_bucket_exists", bucket=settings.influxdb_bucket)
        else:
            logger.warning("influxdb_bucket_not_found", bucket=settings.influxdb_bucket)
        client.close()
        logger.info("influxdb_setup_complete")
    except Exception as e:
        logger.error("influxdb_setup_failed", error=str(e))


async def setup_neo4j() -> None:
    """Neo4j 제약 조건 및 인덱스 생성."""
    logger = get_logger("setup_db")
    settings = get_settings()
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        async with driver.session() as session:
            # Create uniqueness constraints for each node type
            for node_type in ["metric", "event", "entity", "document", "concept"]:
                try:
                    await session.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type}) REQUIRE n.id IS UNIQUE"
                    )
                except Exception:
                    pass  # Constraint may already exist

            # Create index on name property
            try:
                await session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (n:entity) ON (n.name)"
                )
            except Exception:
                pass

        await driver.close()
        logger.info("neo4j_setup_complete")
    except Exception as e:
        logger.error("neo4j_setup_failed", error=str(e))


async def setup_milvus() -> None:
    """Milvus 컬렉션 확인/생성."""
    logger = get_logger("setup_db")
    try:
        from src.data.repositories.vector import VectorRepository
        repo = VectorRepository()
        repo._ensure_collection()
        repo.close()
        logger.info("milvus_setup_complete")
    except Exception as e:
        logger.error("milvus_setup_failed", error=str(e))


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("setup_db")
    logger.info("database_setup_starting")

    await setup_postgresql()
    await setup_influxdb()
    await setup_neo4j()
    await setup_milvus()

    logger.info("database_setup_complete")


if __name__ == "__main__":
    asyncio.run(main())
