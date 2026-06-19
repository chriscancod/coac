"""
CAD tools — Fusion 360 and Tinkercad local operations.

Fusion 360 is controlled via its Python API (scripts executed
through the Fusion 360 CLI or the API sandbox). Tinkercad is
browser-based — operations target locally exported STL/STEP/JSON
files rather than the live UI.

Physical-digital duality: every CAD action logs whether a physical
artifact (print, board, mockup) exists. If none exists, the tool
returns a warning that the plan must include producing one.

Risk classification:
  - cad_overwrite (modify/replace existing design) = 0.25 irreversibility
  - cad_export    (generate output files)           = 0.10 irreversibility
Both are purely local, so no external_effect premium applies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from governor import assert_cofounder_or_raise
from notifications.apns import send_critical_alert, send_interrupt
from notifications.interrupt_queue import enqueue_interrupt
from risk import score
from state.checkpointer import raw_connection
from tools.code_tools import _with_retry

logger = logging.getLogger(__name__)

CAD_WORKSPACE = Path(os.getenv("CAD_WORKSPACE_PATH", "/workspace/cad"))


# ── Fusion 360 ─────────────────────────────────────────────────────────────────

async def fusion_run_script(
    script_path: str,
    design_name: str,
    physical_artifact_exists: bool = False,
) -> dict[str, Any]:
    """
    Execute a Fusion 360 Python script via the Fusion CLI.

    Requires Fusion 360 installed locally with API scripting enabled.
    script_path must point to a .py file in the CAD_WORKSPACE.

    The physical_artifact_exists flag drives the duality check —
    set False when no physical board/print/mockup yet exists.
    """
    assert_cofounder_or_raise(f"fusion_run_script: {design_name}")
    risk_score = score("cad_overwrite", "single_file")

    phys_warning = _duality_warning(design_name, physical_artifact_exists)

    if risk_score.requires_approval:
        async with raw_connection() as conn:
            action_id = await enqueue_interrupt(
                conn,
                action_description=f"Fusion 360 script on {design_name}",
                action_category="cad_overwrite",
                risk_score=risk_score.total,
                risk_rationale=risk_score.rationale,
                payload={"script_path": script_path, "design_name": design_name},
            )
        await send_interrupt(
            f"Fusion 360 script: {design_name}",
            risk_score.total,
            risk_score.rationale,
            action_id,
        )
        return {
            "executed":  False,
            "result":    f"Interrupt queued — action_id={action_id}",
            "risk":      risk_score.total,
            "action_id": action_id,
            "physical_artifact_warning": phys_warning,
        }

    async def _fn(_prev_error: str) -> str:
        result = subprocess.run(
            ["FusionCLI", "run-script", script_path],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or f"Script executed on {design_name}."

    result = await _with_retry(
        f"fusion_script:{design_name}", "cad_overwrite", _fn,
        blast_radius_tier="single_file",
        self_correct_prompt="Return a corrected Fusion 360 Python script.",
    )
    return {
        "executed": True,
        "result":   result,
        "risk":     risk_score.total,
        "physical_artifact_warning": phys_warning,
    }


async def fusion_export(
    design_name: str,
    output_format: str = "STEP",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Export a Fusion 360 design to STEP, STL, or F3D.

    Export is read/generate-only — it does not mutate the source design.
    Scored as cad_export (lower risk than overwrite).
    """
    assert_cofounder_or_raise(f"fusion_export: {design_name}")
    risk_score = score("cad_export", "single_file")

    out_dir = Path(output_dir or CAD_WORKSPACE / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{design_name}.{output_format.lower()}"

    async def _fn(_: str) -> str:
        result = subprocess.run(
            ["FusionCLI", "export", design_name, "--format", output_format, "--out", str(out_file)],
            capture_output=True, text=True, check=True,
        )
        return f"Exported {design_name} → {out_file}"

    result = await _with_retry(
        f"fusion_export:{design_name}", "cad_export", _fn,
        blast_radius_tier="single_file",
    )
    return {"executed": True, "result": result, "output_path": str(out_file), "risk": risk_score.total}


# ── Tinkercad ─────────────────────────────────────────────────────────────────

async def tinkercad_update_design(
    design_json_path: str,
    patch: dict,
    physical_artifact_exists: bool = False,
) -> dict[str, Any]:
    """
    Merge a patch dict into a locally-exported Tinkercad design JSON.

    Tinkercad does not have a public REST API for programmatic design
    mutation, so we operate on the exported JSON (downloaded via the
    Tinkercad export flow) and re-import via their CLI bridge if available.

    This is the correct tradeoff: we never scrape or automate the browser
    UI because Tinkercad can detect it and lock the account.
    """
    assert_cofounder_or_raise(f"tinkercad_update: {design_json_path}")
    risk_score = score("cad_overwrite", "single_file")
    phys_warning = _duality_warning(design_json_path, physical_artifact_exists)

    if risk_score.requires_approval:
        async with raw_connection() as conn:
            action_id = await enqueue_interrupt(
                conn,
                action_description=f"Tinkercad design patch: {design_json_path}",
                action_category="cad_overwrite",
                risk_score=risk_score.total,
                risk_rationale=risk_score.rationale,
                payload={"design_json_path": design_json_path, "patch": patch},
            )
        await send_interrupt(
            f"Tinkercad patch: {design_json_path}",
            risk_score.total, risk_score.rationale, action_id,
        )
        return {
            "executed": False,
            "result":   f"Interrupt queued — action_id={action_id}",
            "risk":     risk_score.total,
            "physical_artifact_warning": phys_warning,
        }

    async def _fn(_: str) -> str:
        path = Path(design_json_path)
        # Backup before mutating — single-file blast radius depends on this.
        backup = path.with_suffix(".bak.json")
        shutil.copy2(path, backup)

        design = json.loads(path.read_text())
        design.update(patch)
        path.write_text(json.dumps(design, indent=2))
        return f"Design patched and saved. Backup at {backup}."

    result = await _with_retry(
        f"tinkercad_patch:{design_json_path}", "cad_overwrite", _fn,
        blast_radius_tier="single_file",
    )
    return {
        "executed": True,
        "result":   result,
        "risk":     risk_score.total,
        "physical_artifact_warning": phys_warning,
    }


# ── Duality check ──────────────────────────────────────────────────────────────

def _duality_warning(design_ref: str, artifact_exists: bool) -> str | None:
    """
    Physical-digital duality rule: hardware/integration tasks must
    reference a physical deliverable. If none exists, flag it.
    """
    if artifact_exists:
        return None
    return (
        f"⚠️  PHYSICAL-DIGITAL DUALITY: No physical artifact confirmed for '{design_ref}'. "
        "The plan must include a sub-task to produce a physical deliverable "
        "(print, PCB order, or mockup) before this phase is considered complete."
    )
