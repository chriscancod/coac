"""
Raw APNs push notifications via httpx HTTP/2 + JWT auth.

No Firebase, no third-party push service. Direct to Apple.

Token signing uses ES256 (ECDSA P-256) with the p8 private key
issued from the Apple Developer portal. The JWT is cached and
reused for up to 55 minutes (Apple invalidates after 60 min).

Required environment variables:
    APNS_KEY_ID       — 10-character key ID from Apple Developer
    APNS_TEAM_ID      — 10-character team ID
    APNS_BUNDLE_ID    — e.g. com.veynor.apex
    APNS_KEY_PATH     — local path to AuthKey_XXXXXXXXXX.p8
    APNS_DEVICE_TOKEN — hex device token for the founder's iPhone
    APNS_PRODUCTION   — "true" for production gateway, else sandbox
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import jwt as pyjwt   # PyJWT

logger = logging.getLogger(__name__)

NotificationType = Literal["interrupt", "eod_report", "critical_alert"]

# APNs gateway URLs
APNS_PROD_HOST    = "https://api.push.apple.com"
APNS_SANDBOX_HOST = "https://api.sandbox.push.apple.com"

_jwt_cache: dict[str, float | str] = {"token": "", "issued_at": 0.0}


def _load_private_key() -> ec.EllipticCurvePrivateKey:
    key_path = Path(os.environ["APNS_KEY_PATH"])
    raw = key_path.read_bytes()
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError("APNs key must be an EC private key (P-256)")
    return key


def _get_jwt() -> str:
    """
    Return a valid APNs provider JWT.

    Tokens are cached for 55 minutes to avoid Apple's 60-minute
    hard expiry. Generating a new token per request would cause
    APNs to reject bursts of notifications as replays.
    """
    now = time.time()
    if now - float(_jwt_cache["issued_at"]) < 55 * 60 and _jwt_cache["token"]:
        return str(_jwt_cache["token"])

    key    = _load_private_key()
    issued = int(now)

    token = pyjwt.encode(
        {"iss": os.environ["APNS_TEAM_ID"], "iat": issued},
        key,
        algorithm="ES256",
        headers={"alg": "ES256", "kid": os.environ["APNS_KEY_ID"]},
    )
    _jwt_cache["token"]     = token
    _jwt_cache["issued_at"] = float(issued)
    logger.debug("[APNs] JWT refreshed (iat=%d)", issued)
    return token


def _apns_host() -> str:
    use_prod = os.getenv("APNS_PRODUCTION", "false").lower() == "true"
    return APNS_PROD_HOST if use_prod else APNS_SANDBOX_HOST


@dataclass
class PushResult:
    success: bool
    apns_id: str
    status_code: int
    error_reason: str | None = None


async def _send(
    device_token: str,
    payload: dict,
    push_type: str,
    priority: int = 10,
    collapse_id: str | None = None,
) -> PushResult:
    """
    Low-level HTTP/2 POST to APNs.

    httpx is used instead of requests because APNs requires HTTP/2
    and requests does not support it natively. The h2 package must
    be installed for httpx to negotiate h2.
    """
    bundle_id   = os.environ["APNS_BUNDLE_ID"]
    token       = _get_jwt()
    url         = f"{_apns_host()}/3/device/{device_token}"

    headers = {
        "authorization":  f"bearer {token}",
        "apns-topic":     bundle_id,
        "apns-push-type": push_type,
        "apns-priority":  str(priority),
        "content-type":   "application/json",
    }
    if collapse_id:
        # collapse-id ensures that if multiple interrupts pile up,
        # only the latest one is shown — avoids notification flood.
        headers["apns-collapse-id"] = collapse_id

    async with httpx.AsyncClient(http2=True) as client:
        resp = await client.post(url, headers=headers, content=json.dumps(payload))

    apns_id = resp.headers.get("apns-id", "")
    if resp.status_code == 200:
        logger.info("[APNs] Push delivered (apns-id=%s, type=%s)", apns_id, push_type)
        return PushResult(success=True, apns_id=apns_id, status_code=200)

    body   = resp.json() if resp.content else {}
    reason = body.get("reason", "unknown")
    logger.error(
        "[APNs] Push failed — %d %s (apns-id=%s)", resp.status_code, reason, apns_id
    )
    return PushResult(
        success=False,
        apns_id=apns_id,
        status_code=resp.status_code,
        error_reason=reason,
    )


# ── Public notification helpers ───────────────────────────────────────────────

async def send_interrupt(
    action_description: str,
    risk_score: float,
    risk_rationale: str,
    action_id: str,
) -> PushResult:
    """
    Push an interrupt notification when a risky action needs approval.

    The deep link encodes the action_id so the SwiftUI app can route
    the approval/rejection to POST /interrupt/{action_id}/approve|reject.
    """
    device_token = os.environ["APNS_DEVICE_TOKEN"]
    payload = {
        "aps": {
            "alert": {
                "title": f"⚠️ Action Requires Approval — Risk {int(risk_score * 100)}%",
                "body":  action_description[:200],
            },
            "sound":             "default",
            "badge":             1,
            "category":          "APEX_INTERRUPT",
            "interruption-level": "time-sensitive",
        },
        "action_id":      action_id,
        "risk_score":     risk_score,
        "risk_rationale": risk_rationale,
        "deep_link":      f"apex://interrupt/{action_id}",
    }
    return await _send(
        device_token,
        payload,
        push_type="alert",
        priority=10,
        # Collapse so repeated risk checks for the same action don't
        # stack up — founder sees the latest state only.
        collapse_id=f"interrupt-{action_id}",
    )


async def send_eod_report(summary: str, plan_preview: str) -> PushResult:
    """
    18:00 executive push — taps open the EOD report in the SwiftUI app.
    """
    device_token = os.environ["APNS_DEVICE_TOKEN"]
    payload = {
        "aps": {
            "alert": {
                "title": "Veynor EOD — Daily Brief Ready",
                "body":  summary[:200],
            },
            "sound":    "default",
            "badge":    1,
            "category": "APEX_EOD",
        },
        "plan_preview": plan_preview[:500],
        "deep_link":    "apex://eod/today",
    }
    return await _send(device_token, payload, push_type="alert", priority=5)


async def send_critical_alert(
    failed_action: str,
    retry_count: int,
    full_context: str,
) -> PushResult:
    """
    Sent when an action has exhausted all 2 self-correction retries.
    Uses APNs critical alert category — plays sound even in Silent mode.

    Note: critical alerts require the com.apple.developer.usernotifications
    .critical-alerts entitlement in the app. Without it, the push still
    delivers but without the override-silent behaviour.
    """
    device_token = os.environ["APNS_DEVICE_TOKEN"]
    payload = {
        "aps": {
            "alert": {
                "title": "🚨 APEX CRITICAL — Manual Intervention Required",
                "body":  f"Action failed after {retry_count} retries: {failed_action[:150]}",
            },
            "sound": {
                "name":     "default",
                "critical": 1,
                "volume":   1.0,
            },
            "badge":             1,
            "category":          "APEX_CRITICAL",
            "interruption-level": "critical",
        },
        "failed_action": failed_action,
        "retry_count":   retry_count,
        "context":       full_context[:2000],
        "deep_link":     "apex://critical/latest",
    }
    return await _send(device_token, payload, push_type="alert", priority=10)
