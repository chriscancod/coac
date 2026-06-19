"""
EOD Report generator — runs at 18:00 on COFOUNDER days only.

The report is written like a co-founder briefing a board:
narrative prose, not a technical log. It covers what happened,
what's blocked, autonomous decisions made with their risk scores,
and the current Rotation Cycle position.

The full report is stored in PostgreSQL and a compressed summary
is pushed via APNs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TypedDict

import asyncpg
from openai import AsyncOpenAI

from foundry import build_suggestion, load_state
from notifications.apns import send_eod_report

logger = logging.getLogger(__name__)
oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

CREATE_EOD_TABLE = """
CREATE TABLE IF NOT EXISTS eod_reports (
    id            SERIAL PRIMARY KEY,
    report_date   DATE        NOT NULL DEFAULT CURRENT_DATE,
    division      TEXT        NOT NULL DEFAULT 'all',
    full_report   TEXT        NOT NULL,
    summary       TEXT        NOT NULL,
    rotation_pos  TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class EODContext(TypedDict):
    completed_software: list[str]
    completed_hardware: list[str]
    completed_brand:    list[str]
    blockers:           list[dict]           # [{description, reason}]
    autonomous_decisions: list[dict]         # [{action, risk_score, outcome}]
    rotation_position:  str


async def generate_and_send(
    conn: asyncpg.Connection,
    context: EODContext,
) -> str:
    """
    Generate the EOD narrative, store it, and push via APNs.

    Returns the full report text.
    """
    await conn.execute(CREATE_EOD_TABLE)

    rotation_state   = await load_state(conn)
    rotation_position = rotation_state["current_phase"]
    suggestion       = build_suggestion(rotation_state)

    full_report = await _generate_narrative(context, rotation_position, suggestion)
    summary     = await _generate_summary(full_report)

    await conn.execute(
        """
        INSERT INTO eod_reports (full_report, summary, rotation_pos)
        VALUES ($1, $2, $3)
        """,
        full_report, summary, rotation_position,
    )

    tomorrow_preview = _plan_preview_from_suggestion(suggestion)
    await send_eod_report(summary, tomorrow_preview)

    logger.info("[EOD] Report generated and pushed — rotation=%s", rotation_position)
    return full_report


async def _generate_narrative(
    ctx: EODContext,
    rotation_position: str,
    suggestion: dict,
) -> str:
    """
    Use o4-mini to write the board-level narrative.
    The prompt is exhaustive so the model has everything it needs
    without having to infer from sparse context.
    """
    autonomous_block = "\n".join(
        f"  - {d['action']} (risk={d['risk_score']:.0%}) → {d['outcome']}"
        for d in ctx["autonomous_decisions"]
    ) or "  None today."

    blockers_block = "\n".join(
        f"  - {b['description']}: {b['reason']}"
        for b in ctx["blockers"]
    ) or "  None."

    prompt = f"""
You are writing an executive EOD report for Veynor — a technology company in Cycle 2: Hardening.

Write exactly like a co-founder briefing a board: executive narrative prose, not a bullet list or
technical log. Three paragraphs maximum. Every sentence must carry operational weight.
No filler. No passive voice. No status-report clichés.

Structure:
1. What we accomplished today (grouped by division where relevant)
2. Current blockers and the concrete reason each one exists
3. Autonomous decisions APEX-AGENT made today with their risk scores and outcomes,
   plus the current Rotation Cycle position

Context data:

SOFTWARE COMPLETED:
{chr(10).join(f'  - {t}' for t in ctx['completed_software']) or '  None.'}

HARDWARE COMPLETED:
{chr(10).join(f'  - {t}' for t in ctx['completed_hardware']) or '  None.'}

BRAND COMPLETED:
{chr(10).join(f'  - {t}' for t in ctx['completed_brand']) or '  None.'}

BLOCKERS:
{blockers_block}

AUTONOMOUS DECISIONS TODAY:
{autonomous_block}

ROTATION CYCLE POSITION: {rotation_position}
SUGGESTED NEXT PHASE: {suggestion['suggested_next_phase']}
"""

    resp = await oai.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "You are APEX-AGENT's reporting engine. Write only the report. No preamble."},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


async def _generate_summary(full_report: str) -> str:
    """Generate the one-paragraph APNs push body from the full report."""
    resp = await oai.chat.completions.create(
        model="o4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Compress the following EOD report into a single sentence "
                    "under 200 characters. No filler. Keep the most operationally "
                    "significant fact. Output only the sentence."
                ),
            },
            {"role": "user", "content": full_report},
        ],
    )
    return resp.choices[0].message.content.strip()[:200]


def _plan_preview_from_suggestion(suggestion: dict) -> str:
    return (
        f"Tomorrow: {suggestion['suggested_next_phase']} phase. "
        f"{suggestion['cycle_2_context'][:150]}"
    )


async def get_latest_report(conn: asyncpg.Connection) -> dict | None:
    """Fetch the most recent EOD report — used by the SwiftUI EOD endpoint."""
    await conn.execute(CREATE_EOD_TABLE)
    row = await conn.fetchrow(
        "SELECT * FROM eod_reports ORDER BY created_at DESC LIMIT 1"
    )
    if not row:
        return None
    return {
        "report_date":    row["report_date"].isoformat(),
        "full_report":    row["full_report"],
        "summary":        row["summary"],
        "rotation_pos":   row["rotation_pos"],
        "created_at":     row["created_at"].isoformat(),
    }
