"""
Code tools — Git operations, Railway deploys, and local file edits.

Every public function:
  1. Calls assert_cofounder_or_raise() — hard JARVIS gate.
  2. Scores the action through risk.py — auto-executes or queues interrupt.
  3. On failure, retries up to 2 times with o4-mini self-correction.
  4. Third failure → critical APNs alert, full context logged.

Railway deploys are triggered via the Railway CLI (railway up) because
the REST API does not expose a stable trigger endpoint without a paid
plan. The CLI is expected to be installed in the Railway service image.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from governor import assert_cofounder_or_raise
from notifications.apns import send_critical_alert, send_interrupt
from notifications.interrupt_queue import enqueue_interrupt
from risk import score, ActionCategory
from state.checkpointer import raw_connection

logger = logging.getLogger(__name__)
oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

MAX_RETRIES = 2


# ── Internal retry + correction loop ─────────────────────────────────────────

async def _with_retry(
    label: str,
    action_category: ActionCategory,
    fn,                     # async callable returning str (result/error)
    *,
    blast_radius_tier: str = "single_file",
    self_correct_prompt: str = "",
) -> str:
    """
    Execute fn() with up to MAX_RETRIES o4-mini self-correction passes.

    On each failure, o4-mini receives the error and the self_correct_prompt
    and returns a corrected command or file content which is fed back into fn().
    """
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await fn(last_error if attempt > 0 else "")
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning("[CODE] Attempt %d/%d failed for '%s': %s",
                           attempt + 1, MAX_RETRIES + 1, label, last_error)
            if attempt < MAX_RETRIES:
                correction = await _self_correct(label, last_error, self_correct_prompt)
                logger.info("[CODE] o4-mini correction for '%s': %s", label, correction[:200])
                last_error = correction
            else:
                ctx = f"label={label} | category={action_category} | error={last_error}"
                await send_critical_alert(label, MAX_RETRIES, ctx)
                raise RuntimeError(
                    f"[CODE] '{label}' failed after {MAX_RETRIES} retries. "
                    "Critical alert sent. Manual intervention required."
                ) from exc


async def _self_correct(label: str, error: str, context: str) -> str:
    resp = await oai.chat.completions.create(
        model="o4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are APEX-AGENT's self-correction engine. "
                    "Analyse the error and return ONLY the corrected command or content. "
                    "No explanation. No markdown. Raw output only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Action: {label}\n"
                    f"Error: {error}\n"
                    f"Context: {context}"
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


async def _gate_and_execute(
    label: str,
    category: ActionCategory,
    blast_radius_tier: str,
    payload: dict,
    fn,
    self_correct_prompt: str = "",
) -> dict[str, Any]:
    """
    Standard gate: check governor → score → execute or interrupt.
    Returns dict with keys: executed (bool), result (str), risk (float).
    """
    mode = assert_cofounder_or_raise(label)
    risk = score(category, blast_radius_tier)

    if risk.requires_approval:
        async with raw_connection() as conn:
            action_id = await enqueue_interrupt(
                conn,
                action_description=label,
                action_category=category,
                risk_score=risk.total,
                risk_rationale=risk.rationale,
                payload=payload,
            )
        await send_interrupt(label, risk.total, risk.rationale, action_id)
        return {
            "executed":  False,
            "result":    f"Interrupt queued — action_id={action_id}",
            "risk":      risk.total,
            "action_id": action_id,
        }

    result = await _with_retry(
        label, category, fn,
        blast_radius_tier=blast_radius_tier,
        self_correct_prompt=self_correct_prompt,
    )
    return {"executed": True, "result": result, "risk": risk.total}


# ── Public tool functions ─────────────────────────────────────────────────────

async def edit_file(
    file_path: str,
    new_content: str,
    commit_message: str,
) -> dict[str, Any]:
    """
    Overwrite a local file and stage the change (no push).

    Risk classification: file_edit / single_file → typically auto-executes.
    Push is a separate, higher-risk operation requiring its own approval.
    """
    label = f"edit_file: {file_path}"

    async def _fn(_prev_error: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content, encoding="utf-8")
        _git(["add", file_path])
        return f"Wrote {len(new_content)} bytes to {file_path} and staged."

    return await _gate_and_execute(
        label, "file_edit", "single_file",
        {"file_path": file_path, "commit_message": commit_message},
        _fn,
    )


async def git_commit(repo_path: str, message: str) -> dict[str, Any]:
    """Commit all staged changes. Does not push."""
    label = f"git_commit: {repo_path}"

    async def _fn(_: str) -> str:
        _git(["commit", "-m", message], cwd=repo_path)
        return f"Committed in {repo_path}: {message}"

    return await _gate_and_execute(
        label, "git_commit", "repo",
        {"repo_path": repo_path, "message": message},
        _fn,
        self_correct_prompt="Return only a corrected git commit message.",
    )


async def git_push(repo_path: str, branch: str = "main") -> dict[str, Any]:
    """
    Push to remote. Scored as git_push / repo — likely triggers an interrupt
    because blast radius crosses into the shared remote state.
    """
    label = f"git_push: {repo_path} → {branch}"

    async def _fn(_: str) -> str:
        _git(["push", "origin", branch], cwd=repo_path)
        return f"Pushed {branch} in {repo_path}."

    return await _gate_and_execute(
        label, "git_push", "repo",
        {"repo_path": repo_path, "branch": branch},
        _fn,
    )


async def railway_deploy(service_name: str, env: str = "production") -> dict[str, Any]:
    """
    Trigger a Railway deploy via the CLI.

    Scored as railway_deploy / production_system — always requires approval
    because it touches the live environment.
    """
    label = f"railway_deploy: {service_name} ({env})"

    async def _fn(_: str) -> str:
        result = subprocess.run(
            ["railway", "up", "--service", service_name],
            capture_output=True, text=True, check=True,
            env={**os.environ, "RAILWAY_ENV": env},
        )
        return result.stdout.strip() or "Deploy triggered successfully."

    return await _gate_and_execute(
        label, "railway_deploy", "production_system",
        {"service_name": service_name, "env": env},
        _fn,
    )


# ── Shell helper ───────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True, text=True, check=True,
        cwd=cwd or os.getcwd(),
    )
    return result.stdout.strip()
