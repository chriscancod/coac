"""
Foundry — Rotation Cycle tracker and suggestion engine.

The Rotation Cycle is the strategic heartbeat of APEX-AGENT.
It ensures Veynor's three divisions (Software, Hardware, Brand)
plus Integration and Learning all receive dedicated focus
in a predictable cadence.

Cycle positions are persisted in PostgreSQL so they survive
Railway restarts and remain coherent across COFOUNDER days.

The agent SUGGESTS the next rotation step; the founder APPROVES
or OVERRIDES. All overrides are logged with a reason.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from typing import TypedDict

import asyncpg

logger = logging.getLogger(__name__)


class RotationPhase(str, Enum):
    SOFTWARE    = "Software"
    HARDWARE    = "Hardware"
    INTEGRATION = "Integration"
    BRAND       = "Brand"
    LEARNING    = "Learning"


# Ordered cycle — index determines next step arithmetic.
CYCLE_ORDER: list[RotationPhase] = [
    RotationPhase.SOFTWARE,
    RotationPhase.HARDWARE,
    RotationPhase.INTEGRATION,
    RotationPhase.BRAND,
    RotationPhase.LEARNING,
]

# Cycle 2 context injected into every proposal so the suggestion
# always references current strategic priorities.
CYCLE_2_CONTEXT: dict[RotationPhase, str] = {
    RotationPhase.SOFTWARE: (
        "Strip to bare essentials. Prioritize deterministic, Veynor-Native "
        "performance. Every shipped feature must be production-grade."
    ),
    RotationPhase.HARDWARE: (
        "Migrate from salvaged components to custom PCB. All designs must be "
        "DFM-ready. Advance routing, footprints, and BOM consolidation."
    ),
    RotationPhase.INTEGRATION: (
        "Reduce signal latency between custom hardware and VeyQ OS. "
        "Target sub-millisecond round-trip. Measure before and after every change."
    ),
    RotationPhase.BRAND: (
        "Maintain 2AM / NOCTIS momentum. Ship listings, resolve logistics, "
        "and ensure brand assets stay consistent with Dark Luxury standards."
    ),
    RotationPhase.LEARNING: (
        "Capture Vault entries. Document decisions, failures, and measurements. "
        "Identify what slowed Founder's Velocity this cycle and fix the bottleneck."
    ),
}

# Physical-digital duality requirement flags per phase.
PHYSICAL_DELIVERABLE_REQUIRED: dict[RotationPhase, bool] = {
    RotationPhase.SOFTWARE:    False,
    RotationPhase.HARDWARE:    True,
    RotationPhase.INTEGRATION: True,   # latency benchmarks need a physical board
    RotationPhase.BRAND:       False,
    RotationPhase.LEARNING:    False,
}


class RotationState(TypedDict):
    current_phase: str
    current_index: int
    last_advanced: str       # ISO datetime
    pending_override: bool
    override_log: list[dict]


class FoundrySuggestion(TypedDict):
    current_phase: str
    suggested_next_phase: str
    cycle_2_context: str
    physical_deliverable_required: bool
    physical_deliverable_note: str
    rationale: str


# ── PostgreSQL helpers ─────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS foundry_rotation (
    id                SERIAL PRIMARY KEY,
    current_phase     TEXT        NOT NULL DEFAULT 'Software',
    current_index     INTEGER     NOT NULL DEFAULT 0,
    last_advanced     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pending_override  BOOLEAN     NOT NULL DEFAULT FALSE,
    override_log      JSONB       NOT NULL DEFAULT '[]'::jsonb
);
"""

async def _ensure_table(conn: asyncpg.Connection) -> None:
    await conn.execute(CREATE_TABLE_SQL)
    # Seed a row if the table is empty (first run).
    count = await conn.fetchval("SELECT COUNT(*) FROM foundry_rotation")
    if count == 0:
        await conn.execute(
            "INSERT INTO foundry_rotation (current_phase, current_index) VALUES ($1, $2)",
            RotationPhase.SOFTWARE.value, 0,
        )


async def load_state(conn: asyncpg.Connection) -> RotationState:
    await _ensure_table(conn)
    row = await conn.fetchrow(
        "SELECT * FROM foundry_rotation ORDER BY id DESC LIMIT 1"
    )
    return RotationState(
        current_phase=row["current_phase"],
        current_index=row["current_index"],
        last_advanced=row["last_advanced"].isoformat(),
        pending_override=row["pending_override"],
        override_log=json.loads(row["override_log"]),
    )


async def save_state(conn: asyncpg.Connection, state: RotationState) -> None:
    await conn.execute(
        """
        UPDATE foundry_rotation
        SET current_phase    = $1,
            current_index    = $2,
            last_advanced    = $3,
            pending_override = $4,
            override_log     = $5::jsonb
        WHERE id = (SELECT id FROM foundry_rotation ORDER BY id DESC LIMIT 1)
        """,
        state["current_phase"],
        state["current_index"],
        datetime.fromisoformat(state["last_advanced"]),
        state["pending_override"],
        json.dumps(state["override_log"]),
    )


# ── Rotation logic ─────────────────────────────────────────────────────────────

def next_phase(current_index: int) -> tuple[RotationPhase, int]:
    """Return the next phase and its index, wrapping at end of cycle."""
    next_index = (current_index + 1) % len(CYCLE_ORDER)
    return CYCLE_ORDER[next_index], next_index


def build_suggestion(state: RotationState) -> FoundrySuggestion:
    """
    Build the human-readable suggestion card embedded in Tomorrow's Plan.

    The suggestion always describes the NEXT phase so the founder can
    approve or override it before the agent advances the pointer.
    """
    current      = RotationPhase(state["current_phase"])
    nxt, _       = next_phase(state["current_index"])
    needs_phys   = PHYSICAL_DELIVERABLE_REQUIRED[nxt]

    if needs_phys:
        phys_note = (
            f"A physical deliverable is REQUIRED for {nxt.value} phase. "
            "If no physical artifact currently exists (print / board / mockup), "
            "add producing one as a sub-task in the plan."
        )
    else:
        phys_note = f"No physical deliverable required for {nxt.value} phase."

    return FoundrySuggestion(
        current_phase=current.value,
        suggested_next_phase=nxt.value,
        cycle_2_context=CYCLE_2_CONTEXT[nxt],
        physical_deliverable_required=needs_phys,
        physical_deliverable_note=phys_note,
        rationale=(
            f"Cycle completed {current.value}. "
            f"Natural next step is {nxt.value}. "
            "Founder may approve or override."
        ),
    )


async def advance_rotation(conn: asyncpg.Connection) -> FoundrySuggestion:
    """
    Advance the rotation pointer after founder approval and return the
    suggestion card for the newly active phase.

    Call this from plan_approver.py after POST /plan/approve is received.
    """
    state = await load_state(conn)
    _, next_index = next_phase(state["current_index"])
    new_phase = CYCLE_ORDER[next_index]

    state["current_phase"]   = new_phase.value
    state["current_index"]   = next_index
    state["last_advanced"]   = datetime.utcnow().isoformat()
    state["pending_override"] = False

    await save_state(conn, state)
    logger.info("[FOUNDRY] Rotation advanced to %s", new_phase.value)
    return build_suggestion(state)


async def log_override(
    conn: asyncpg.Connection,
    reason: str,
    founder_directive: str,
) -> None:
    """
    Record a founder override. The rotation pointer is NOT advanced.
    The agent holds the current position until the next eligible COFOUNDER day.
    """
    state = await load_state(conn)
    entry = {
        "timestamp":         datetime.utcnow().isoformat(),
        "held_phase":        state["current_phase"],
        "reason":            reason,
        "founder_directive": founder_directive,
    }
    state["override_log"].append(entry)
    state["pending_override"] = True
    await save_state(conn, state)
    logger.info(
        "[FOUNDRY] Override logged — held at %s. Reason: %s",
        state["current_phase"], reason,
    )
