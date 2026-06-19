"""
PostgreSQL checkpointer for LangGraph.

Uses langgraph-checkpoint-postgres so the agent graph resumes
from its last known state after any Railway restart.

The connection pool is created once at startup and reused across
all graph invocations. Never close it between agent turns —
connection acquisition overhead on Railway can add ~40ms per request.
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    """
    Return the singleton connection pool, creating it on first call.

    Pool sizing: min=2 keeps warm connections ready; max=10 is generous
    for a single-agent workload but leaves headroom for concurrent
    interrupt approvals arriving via the SwiftUI endpoints.
    """
    global _pool
    if _pool is None:
        dsn = os.environ["DATABASE_URL"]
        _pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("[CHECKPOINTER] PostgreSQL pool created (min=2, max=10)")
    return _pool


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Return an AsyncPostgresSaver backed by the shared pool.

    LangGraph's postgres checkpointer handles schema creation
    on first use — no manual migration needed.
    """
    pool = await _get_pool()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver


@asynccontextmanager
async def raw_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Yield a raw asyncpg connection from the pool for use by
    foundry.py, eod_report.py, and other modules that need
    direct SQL access outside of LangGraph state.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        yield conn


async def close_pool() -> None:
    """Graceful shutdown — call from the Railway SIGTERM handler."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("[CHECKPOINTER] PostgreSQL pool closed")
