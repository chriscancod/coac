"""
Interrupt queue — pending approval management.

When APEX-AGENT scores an action >= 0.40 it:
  1. Serialises the pending action into PostgreSQL.
  2. Pushes an APNs interrupt notification.
  3. Suspends the LangGraph thread (via checkpointed state).

When the founder approves or rejects via the SwiftUI app
(POST /interrupt/{action_id}/approve|reject), the FastAPI
endpoint calls `resolve_interrupt` here, which:
  - Updates the DB record.
  - Signals the waiting graph thread to resume or abort.

The queue is intentionally single-tenant (one founder).
No locking complexity needed.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Literal, TypedDict

import asyncpg

logger = logging.getLogger(__name__)

InterruptStatus = Literal["pending", "approved", "rejected", "expired"]


class InterruptRecord(TypedDict):
    action_id:         str
    action_description: str
    action_category:   str
    risk_score:        float
    risk_rationale:    str
    payload:           dict          # full action kwargs for replay on approval
    status:            InterruptStatus
    created_at:        str
    resolved_at:       str | None
    founder_note:      str | None


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interrupt_queue (
    action_id           TEXT PRIMARY KEY,
    action_description  TEXT        NOT NULL,
    action_category     TEXT        NOT NULL,
    risk_score          REAL        NOT NULL,
    risk_rationale      TEXT        NOT NULL,
    payload             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT        NOT NULL DEFAULT 'pending',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ,
    founder_note        TEXT
);
"""


async def _ensure_table(conn: asyncpg.Connection) -> None:
    await conn.execute(CREATE_TABLE_SQL)


async def enqueue_interrupt(
    conn: asyncpg.Connection,
    action_description: str,
    action_category: str,
    risk_score: float,
    risk_rationale: str,
    payload: dict,
) -> str:
    """
    Persist a pending action and return a unique action_id.

    The action_id is embedded in the APNs deep link so the SwiftUI
    app knows which record to approve/reject.
    """
    await _ensure_table(conn)
    action_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO interrupt_queue
            (action_id, action_description, action_category,
             risk_score, risk_rationale, payload)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        action_id,
        action_description,
        action_category,
        risk_score,
        risk_rationale,
        json.dumps(payload),
    )
    logger.info(
        "[INTERRUPT] Enqueued %s — risk=%.2f — %s",
        action_id, risk_score, action_description[:80],
    )
    return action_id


async def resolve_interrupt(
    conn: asyncpg.Connection,
    action_id: str,
    decision: Literal["approved", "rejected"],
    founder_note: str | None = None,
) -> InterruptRecord | None:
    """
    Record the founder's decision and return the full interrupt record
    so the API endpoint can relay it back to the waiting graph thread.

    Returns None if the action_id does not exist or is already resolved.
    """
    await _ensure_table(conn)
    row = await conn.fetchrow(
        "SELECT * FROM interrupt_queue WHERE action_id = $1", action_id
    )
    if row is None:
        logger.warning("[INTERRUPT] Unknown action_id: %s", action_id)
        return None
    if row["status"] != "pending":
        logger.warning(
            "[INTERRUPT] action_id %s already resolved as %s", action_id, row["status"]
        )
        return None

    await conn.execute(
        """
        UPDATE interrupt_queue
        SET status       = $1,
            resolved_at  = $2,
            founder_note = $3
        WHERE action_id  = $4
        """,
        decision,
        datetime.utcnow(),
        founder_note,
        action_id,
    )
    logger.info("[INTERRUPT] %s → %s (note: %s)", action_id, decision, founder_note)

    updated = await conn.fetchrow(
        "SELECT * FROM interrupt_queue WHERE action_id = $1", action_id
    )
    return _row_to_record(updated)


async def get_pending(conn: asyncpg.Connection) -> list[InterruptRecord]:
    """Return all unresolved interrupts — used by the SwiftUI pending-list endpoint."""
    await _ensure_table(conn)
    rows = await conn.fetch(
        "SELECT * FROM interrupt_queue WHERE status = 'pending' ORDER BY created_at"
    )
    return [_row_to_record(r) for r in rows]


async def get_interrupt(
    conn: asyncpg.Connection, action_id: str
) -> InterruptRecord | None:
    await _ensure_table(conn)
    row = await conn.fetchrow(
        "SELECT * FROM interrupt_queue WHERE action_id = $1", action_id
    )
    return _row_to_record(row) if row else None


def _row_to_record(row: asyncpg.Record) -> InterruptRecord:
    return InterruptRecord(
        action_id=row["action_id"],
        action_description=row["action_description"],
        action_category=row["action_category"],
        risk_score=row["risk_score"],
        risk_rationale=row["risk_rationale"],
        payload=json.loads(row["payload"]),
        status=row["status"],
        created_at=row["created_at"].isoformat(),
        resolved_at=row["resolved_at"].isoformat() if row["resolved_at"] else None,
        founder_note=row["founder_note"],
    )
