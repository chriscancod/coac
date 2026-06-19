"""
APEX-AGENT — LangGraph entrypoint.

Architecture overview:
  - FastAPI serves the SwiftUI integration endpoints.
  - APScheduler fires the 18:00 EOD flow on COFOUNDER days.
  - LangGraph manages agent state with PostgreSQL checkpointing.
  - The Governor is checked at the top of every node — JARVIS mode
    blocks all graph execution and routes to conversational-only handling.

Graph nodes:
  governor_check  → decide COFOUNDER vs JARVIS, short-circuit if JARVIS
  plan_gate       → halt if no approved plan exists for today
  agent_loop      → main ReAct loop with tools
  eod_trigger     → 18:00 EOD report + plan generation (scheduler-driven)

Thread IDs: we use a single thread_id ("apex-main") so state is
continuous across restarts. A new thread is only opened when explicitly
testing or debugging.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import date, datetime

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI
from pydantic import BaseModel

from foundry import load_state as load_rotation_state
from governor import get_mode
from notifications.apns import send_interrupt
from notifications.interrupt_queue import (
    get_interrupt,
    get_pending,
    resolve_interrupt,
)
from prompts.system_prompt import SYSTEM_PROMPT
from reporting.eod_report import generate_and_send as generate_eod, get_latest_report
from reporting.plan_approver import (
    approve_plan,
    get_plan,
    generate_plan,
    revise_plan,
)
from state.checkpointer import close_pool, get_checkpointer, raw_connection
from tools.brand_tools import create_listing, fulfill_order, get_open_orders, update_listing
from tools.cad_tools import fusion_export, fusion_run_script, tinkercad_update_design
from tools.code_tools import edit_file, git_commit, git_push, railway_deploy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

oai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

THREAD_ID = "apex-main"

# ── Agent state schema ─────────────────────────────────────────────────────────

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages:         Annotated[list, add_messages]
    governor_mode:    str
    approved_plan:    dict | None
    rotation_phase:   str
    interrupt_halted: bool
    retry_counts:     dict[str, int]


# ── Graph nodes ────────────────────────────────────────────────────────────────

async def governor_check_node(state: AgentState) -> AgentState:
    """
    First node in every graph invocation.
    Sets governor_mode and short-circuits if JARVIS.
    """
    mode = get_mode()
    return {**state, "governor_mode": mode["mode"]}


async def plan_gate_node(state: AgentState) -> AgentState:
    """
    Block execution if no approved plan exists for today.
    The agent sits idle until POST /plan/approve is received.
    """
    if state["governor_mode"] != "COFOUNDER":
        return state

    today = date.today().isoformat()
    async with raw_connection() as conn:
        plan = await get_plan(conn, today)

    if plan and plan.get("status") == "approved":
        return {**state, "approved_plan": plan, "interrupt_halted": False}

    logger.info("[PLAN GATE] No approved plan for %s — agent halted", today)
    return {**state, "interrupt_halted": True}


async def agent_loop_node(state: AgentState) -> AgentState:
    """
    Main ReAct agent loop using o4-mini with tool use.

    Tool definitions are bound from the tools/ modules.
    The loop runs until the model returns no further tool calls
    or until an interrupt is raised by the risk engine.
    """
    if state["governor_mode"] != "COFOUNDER" or state.get("interrupt_halted"):
        return state

    tools = _build_tool_definitions()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *state["messages"],
    ]

    response = await oai.chat.completions.create(
        model="o4-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    new_messages = [*state["messages"], msg]

    if not msg.tool_calls:
        return {**state, "messages": new_messages}

    # Execute tool calls sequentially — tool results are appended
    # and the loop will be re-entered by LangGraph's conditional edge.
    tool_results = []
    for tc in msg.tool_calls:
        result = await _dispatch_tool(tc.function.name, tc.function.arguments)
        tool_results.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      str(result),
        })

    return {**state, "messages": [*new_messages, *tool_results]}


async def jarvis_node(state: AgentState) -> AgentState:
    """
    JARVIS mode: conversational only, no tool calls.
    Responds to the last user message with expert advisory output.
    """
    if not state["messages"]:
        return state

    last_user = next(
        (m for m in reversed(state["messages"]) if getattr(m, "role", None) == "user"),
        None,
    )
    if not last_user:
        return state

    response = await oai.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": last_user.content},
        ],
    )
    reply = response.choices[0].message
    return {**state, "messages": [*state["messages"], reply]}


def _route_after_governor(state: AgentState) -> str:
    if state["governor_mode"] == "JARVIS":
        return "jarvis"
    return "plan_gate"


def _route_after_plan_gate(state: AgentState) -> str:
    if state.get("interrupt_halted"):
        return END
    return "agent_loop"


def _route_after_agent_loop(state: AgentState) -> str:
    """Continue looping if the last message has tool calls, else end."""
    if not state["messages"]:
        return END
    last = state["messages"][-1]
    # tool results were appended — check whether the message before them had tool_calls
    if any(m.get("role") == "tool" for m in state["messages"][-3:] if isinstance(m, dict)):
        return "agent_loop"
    return END


# ── Graph construction ─────────────────────────────────────────────────────────

async def build_graph() -> CompiledStateGraph:
    checkpointer = await get_checkpointer()

    g = StateGraph(AgentState)
    g.add_node("governor_check", governor_check_node)
    g.add_node("jarvis",         jarvis_node)
    g.add_node("plan_gate",      plan_gate_node)
    g.add_node("agent_loop",     agent_loop_node)

    g.set_entry_point("governor_check")
    g.add_conditional_edges("governor_check", _route_after_governor)
    g.add_conditional_edges("plan_gate",      _route_after_plan_gate)
    g.add_conditional_edges("agent_loop",     _route_after_agent_loop)
    g.add_edge("jarvis", END)

    return g.compile(checkpointer=checkpointer)


# ── Tool dispatch ──────────────────────────────────────────────────────────────

import json as _json


async def _dispatch_tool(name: str, arguments_str: str) -> dict:
    args = _json.loads(arguments_str)
    dispatch = {
        "edit_file":               edit_file,
        "git_commit":              git_commit,
        "git_push":                git_push,
        "railway_deploy":          railway_deploy,
        "fusion_run_script":       fusion_run_script,
        "fusion_export":           fusion_export,
        "tinkercad_update_design": tinkercad_update_design,
        "create_listing":          create_listing,
        "update_listing":          update_listing,
        "fulfill_order":           fulfill_order,
        "get_open_orders":         get_open_orders,
    }
    fn = dispatch.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return await fn(**args)


def _build_tool_definitions() -> list[dict]:
    """OpenAI tool schema definitions for every callable tool."""
    return [
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Overwrite a local file with new content and stage it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path":      {"type": "string"},
                        "new_content":    {"type": "string"},
                        "commit_message": {"type": "string"},
                    },
                    "required": ["file_path", "new_content", "commit_message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Commit staged changes in a git repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"type": "string"},
                        "message":   {"type": "string"},
                    },
                    "required": ["repo_path", "message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_push",
                "description": "Push committed changes to remote origin.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"type": "string"},
                        "branch":    {"type": "string"},
                    },
                    "required": ["repo_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "railway_deploy",
                "description": "Trigger a Railway deployment for a named service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service_name": {"type": "string"},
                        "env":          {"type": "string"},
                    },
                    "required": ["service_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fusion_run_script",
                "description": "Execute a Fusion 360 Python script on a named design.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_path":              {"type": "string"},
                        "design_name":              {"type": "string"},
                        "physical_artifact_exists": {"type": "boolean"},
                    },
                    "required": ["script_path", "design_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fusion_export",
                "description": "Export a Fusion 360 design to STEP, STL, or F3D.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "design_name":   {"type": "string"},
                        "output_format": {"type": "string", "enum": ["STEP", "STL", "F3D"]},
                        "output_dir":    {"type": "string"},
                    },
                    "required": ["design_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tinkercad_update_design",
                "description": "Patch a locally-exported Tinkercad design JSON file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "design_json_path":         {"type": "string"},
                        "patch":                    {"type": "object"},
                        "physical_artifact_exists": {"type": "boolean"},
                    },
                    "required": ["design_json_path", "patch"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_listing",
                "description": "Create a Shopify product listing for 2AM / NOCTIS.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title":     {"type": "string"},
                        "body_html": {"type": "string"},
                        "price":     {"type": "string"},
                        "sku":       {"type": "string"},
                        "tags":      {"type": "array", "items": {"type": "string"}},
                        "published": {"type": "boolean"},
                    },
                    "required": ["title", "body_html", "price", "sku"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_listing",
                "description": "Patch an existing Shopify product.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "updates":    {"type": "object"},
                    },
                    "required": ["product_id", "updates"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fulfill_order",
                "description": "Mark a Shopify order as fulfilled (always requires approval).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id":         {"type": "string"},
                        "tracking_number":  {"type": "string"},
                        "tracking_company": {"type": "string"},
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_open_orders",
                "description": "Fetch open Shopify orders. Read-only.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


# ── APScheduler EOD job ────────────────────────────────────────────────────────

_graph: CompiledStateGraph | None = None


async def _eod_job() -> None:
    """
    Fires at 18:00 on COFOUNDER days.
    Builds EOD context from the day's LangGraph message history,
    generates the report and tomorrow's plan, and pushes via APNs.
    """
    mode = get_mode()
    if mode["mode"] != "COFOUNDER":
        logger.info("[SCHEDULER] EOD job skipped — JARVIS day (%s)", mode["day_name"])
        return

    logger.info("[SCHEDULER] Running 18:00 EOD job")

    async with raw_connection() as conn:
        # Pull today's message history from checkpointed state to extract context.
        # In production this would be richer — pulling from a dedicated audit table.
        eod_context = {
            "completed_software":    [],
            "completed_hardware":    [],
            "completed_brand":       [],
            "blockers":              [],
            "autonomous_decisions":  [],
            "rotation_position":     (await load_rotation_state(conn))["current_phase"],
        }

        await generate_eod(conn, eod_context)

        plan = await generate_plan(conn, eod_context)
        logger.info("[SCHEDULER] Tomorrow's plan generated — %d tasks", len(plan["tasks"]))


# ── FastAPI app ────────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _graph
    _graph = await build_graph()

    scheduler.add_job(_eod_job, "cron", hour=18, minute=0)
    scheduler.start()
    logger.info("[MAIN] APEX-AGENT online — graph compiled, scheduler started")

    yield

    scheduler.shutdown()
    await close_pool()
    logger.info("[MAIN] APEX-AGENT shutdown complete")


app = FastAPI(title="APEX-AGENT", lifespan=_lifespan)


# ── Plan endpoints ─────────────────────────────────────────────────────────────

class ApproveRequest(BaseModel):
    plan_date: str | None = None


class ReviseRequest(BaseModel):
    plan_date:          str | None = None
    founder_directive:  str


@app.post("/plan/approve")
async def approve_endpoint(req: ApproveRequest):
    plan_date = req.plan_date or date.today().isoformat()
    async with raw_connection() as conn:
        plan = await approve_plan(conn, plan_date)
    return JSONResponse({"status": "approved", "plan": plan})


@app.post("/plan/revise")
async def revise_endpoint(req: ReviseRequest):
    plan_date = req.plan_date or date.today().isoformat()
    async with raw_connection() as conn:
        eod = await get_latest_report(conn) or {}
        plan = await revise_plan(conn, plan_date, req.founder_directive, eod)
    return JSONResponse({"status": "revised", "plan": plan})


@app.get("/plan/{plan_date}")
async def get_plan_endpoint(plan_date: str):
    async with raw_connection() as conn:
        plan = await get_plan(conn, plan_date)
    if not plan:
        raise HTTPException(404, detail=f"No plan for {plan_date}")
    return JSONResponse(plan)


# ── Interrupt endpoints ────────────────────────────────────────────────────────

@app.post("/interrupt/{action_id}/approve")
async def approve_interrupt(action_id: str, founder_note: str | None = None):
    async with raw_connection() as conn:
        record = await resolve_interrupt(conn, action_id, "approved", founder_note)
    if not record:
        raise HTTPException(404, detail=f"Interrupt {action_id} not found or already resolved")
    return JSONResponse({"status": "approved", "action_id": action_id})


@app.post("/interrupt/{action_id}/reject")
async def reject_interrupt(action_id: str, founder_note: str | None = None):
    async with raw_connection() as conn:
        record = await resolve_interrupt(conn, action_id, "rejected", founder_note)
    if not record:
        raise HTTPException(404, detail=f"Interrupt {action_id} not found or already resolved")
    return JSONResponse({"status": "rejected", "action_id": action_id})


@app.get("/interrupt/pending")
async def list_pending_interrupts():
    async with raw_connection() as conn:
        pending = await get_pending(conn)
    return JSONResponse({"count": len(pending), "interrupts": pending})


@app.get("/interrupt/{action_id}")
async def get_interrupt_endpoint(action_id: str):
    async with raw_connection() as conn:
        record = await get_interrupt(conn, action_id)
    if not record:
        raise HTTPException(404, detail=f"Interrupt {action_id} not found")
    return JSONResponse(record)


# ── EOD / reporting endpoints ──────────────────────────────────────────────────

@app.get("/eod/today")
async def get_eod_today():
    async with raw_connection() as conn:
        report = await get_latest_report(conn)
    if not report:
        raise HTTPException(404, detail="No EOD report yet today")
    return JSONResponse(report)


# ── Rotation state endpoint ────────────────────────────────────────────────────

@app.get("/rotation/state")
async def get_rotation_state():
    async with raw_connection() as conn:
        state = await load_rotation_state(conn)
    return JSONResponse(state)


# ── Conversation endpoint (JARVIS / COFOUNDER chat) ───────────────────────────

class MessageRequest(BaseModel):
    message: str
    thread_id: str = THREAD_ID


@app.post("/message")
async def send_message(req: MessageRequest):
    if _graph is None:
        raise HTTPException(503, detail="Graph not yet initialised")

    config = {"configurable": {"thread_id": req.thread_id}}
    result = await _graph.ainvoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )

    last = result["messages"][-1]
    return JSONResponse({
        "governor_mode": result.get("governor_mode"),
        "response":      getattr(last, "content", str(last)),
    })


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    mode = get_mode()
    return JSONResponse({"status": "ok", "governor_mode": mode["mode"], "day": mode["day_name"]})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
