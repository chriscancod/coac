from datetime import datetime
from typing import TypedDict


COFOUNDER_DAYS = {0, 1, 3, 4}   # Mon, Tue, Thu, Fri
JARVIS_DAYS    = {2, 5, 6}      # Wed, Sat, Sun


class GovernorState(TypedDict):
    mode: str
    autonomous: bool
    risk_threshold: float
    tool_use: bool
    description: str
    day_name: str


def get_mode(dt: datetime | None = None) -> GovernorState:
    """
    Returns the active Governor mode based on the current weekday.

    COFOUNDER: full autonomy below risk threshold, EOD reporting active.
    JARVIS: conversational-only, zero tool use — founder is executing,
            agent is the advisor in the room.
    """
    now = dt or datetime.now()
    day = now.weekday()
    day_name = now.strftime("%A")

    if day in COFOUNDER_DAYS:
        return GovernorState(
            mode="COFOUNDER",
            autonomous=True,
            risk_threshold=0.40,
            tool_use=True,
            description=(
                "Autonomous execution below 40% risk. "
                "Interrupt and halt for anything at or above 40%."
            ),
            day_name=day_name,
        )

    return GovernorState(
        mode="JARVIS",
        autonomous=False,
        risk_threshold=0.0,
        tool_use=False,
        description=(
            "Conversational only. Zero tool calls. Zero state changes. "
            "Answer and advise when asked. EOD flow suspended."
        ),
        day_name=day_name,
    )


def assert_cofounder_or_raise(action_label: str, dt: datetime | None = None) -> GovernorState:
    """
    Hard gate: raises RuntimeError if called during JARVIS mode.
    Use at the top of every tool that mutates state or calls external APIs.
    """
    mode = get_mode(dt)
    if mode["mode"] != "COFOUNDER":
        raise RuntimeError(
            f"[GOVERNOR] Action '{action_label}' blocked — JARVIS MODE active on {mode['day_name']}. "
            "No tool calls permitted. Advise only."
        )
    return mode
