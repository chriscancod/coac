"""
Brand tools — 2AM / NOCTIS sales and listing operations.

These tools interact with external commerce platforms (Shopify by default).
Brand sales and listings carry significant external side-effects:
a published listing is immediately visible to customers.

Risk classification:
  brand_listing = 0.20 irreversibility + 0.15 external_effect = 0.35 → borderline
  brand_sale    = 0.35 irreversibility + 0.25 external_effect = 0.60 → always approval

Environment variables required:
  SHOPIFY_SHOP_URL       — e.g. 2am-veynor.myshopify.com
  SHOPIFY_ADMIN_API_KEY  — Admin API access token
  SHOPIFY_API_VERSION    — e.g. 2024-01
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from governor import assert_cofounder_or_raise
from notifications.apns import send_interrupt
from notifications.interrupt_queue import enqueue_interrupt
from risk import score
from state.checkpointer import raw_connection
from tools.code_tools import _with_retry

logger = logging.getLogger(__name__)

SHOPIFY_BASE = "https://{shop}/admin/api/{version}"


def _shopify_headers() -> dict[str, str]:
    return {
        "X-Shopify-Access-Token": os.environ["SHOPIFY_ADMIN_API_KEY"],
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    return SHOPIFY_BASE.format(
        shop=os.environ["SHOPIFY_SHOP_URL"],
        version=os.environ.get("SHOPIFY_API_VERSION", "2024-01"),
    )


# ── Listings ──────────────────────────────────────────────────────────────────

async def create_listing(
    title: str,
    body_html: str,
    price: str,
    sku: str,
    tags: list[str] | None = None,
    published: bool = False,
) -> dict[str, Any]:
    """
    Create a Shopify product listing for 2AM / NOCTIS.

    published=False creates a draft — lower risk. published=True
    is immediately live and triggers the higher external_effect score.
    The risk engine will typically still auto-execute a draft listing
    (score ~0.35) but require approval for a live publish.
    """
    label = f"create_listing: {title} (published={published})"
    assert_cofounder_or_raise(label)

    category = "brand_listing"
    risk_score = score(category, "single_file")

    if risk_score.requires_approval:
        async with raw_connection() as conn:
            action_id = await enqueue_interrupt(
                conn,
                action_description=label,
                action_category=category,
                risk_score=risk_score.total,
                risk_rationale=risk_score.rationale,
                payload={
                    "title": title, "body_html": body_html, "price": price,
                    "sku": sku, "tags": tags, "published": published,
                },
            )
        await send_interrupt(label, risk_score.total, risk_score.rationale, action_id)
        return {
            "executed": False,
            "result":   f"Interrupt queued — action_id={action_id}",
            "risk":     risk_score.total,
        }

    async def _fn(_: str) -> str:
        payload = {
            "product": {
                "title":       title,
                "body_html":   body_html,
                "tags":        ", ".join(tags or []),
                "published":   published,
                "variants":    [{"price": price, "sku": sku}],
            }
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_base_url()}/products.json",
                json=payload,
                headers=_shopify_headers(),
            )
            resp.raise_for_status()
            product_id = resp.json()["product"]["id"]
        return f"Listing created — product_id={product_id} (published={published})"

    result = await _with_retry(label, category, _fn, blast_radius_tier="single_file")
    return {"executed": True, "result": result, "risk": risk_score.total}


async def update_listing(
    product_id: str,
    updates: dict,
) -> dict[str, Any]:
    """
    Patch an existing Shopify product. Scored identically to create_listing.
    """
    label = f"update_listing: product_id={product_id}"
    assert_cofounder_or_raise(label)

    category   = "brand_listing"
    risk_score = score(category, "single_file")

    if risk_score.requires_approval:
        async with raw_connection() as conn:
            action_id = await enqueue_interrupt(
                conn,
                action_description=label,
                action_category=category,
                risk_score=risk_score.total,
                risk_rationale=risk_score.rationale,
                payload={"product_id": product_id, "updates": updates},
            )
        await send_interrupt(label, risk_score.total, risk_score.rationale, action_id)
        return {
            "executed": False,
            "result":   f"Interrupt queued — action_id={action_id}",
            "risk":     risk_score.total,
        }

    async def _fn(_: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{_base_url()}/products/{product_id}.json",
                json={"product": updates},
                headers=_shopify_headers(),
            )
            resp.raise_for_status()
        return f"Listing updated — product_id={product_id}"

    result = await _with_retry(label, category, _fn, blast_radius_tier="single_file")
    return {"executed": True, "result": result, "risk": risk_score.total}


# ── Order / sales actions ─────────────────────────────────────────────────────

async def fulfill_order(
    order_id: str,
    tracking_number: str | None = None,
    tracking_company: str | None = None,
) -> dict[str, Any]:
    """
    Mark a Shopify order as fulfilled.

    brand_sale scored at 0.60 → always requires approval.
    Fulfillment is irreversible once the customer receives a
    shipped notification from Shopify.
    """
    label = f"fulfill_order: order_id={order_id}"
    assert_cofounder_or_raise(label)

    category   = "brand_sale"
    risk_score = score(
        category, "single_file",
        touches_customer_data=True,   # order contains PII
    )

    # brand_sale always exceeds threshold — interrupt is always queued.
    async with raw_connection() as conn:
        action_id = await enqueue_interrupt(
            conn,
            action_description=label,
            action_category=category,
            risk_score=risk_score.total,
            risk_rationale=risk_score.rationale,
            payload={
                "order_id": order_id,
                "tracking_number": tracking_number,
                "tracking_company": tracking_company,
            },
        )
    await send_interrupt(label, risk_score.total, risk_score.rationale, action_id)
    return {
        "executed":  False,
        "result":    f"Interrupt queued — action_id={action_id}",
        "risk":      risk_score.total,
        "action_id": action_id,
    }


async def get_open_orders() -> dict[str, Any]:
    """
    Fetch open Shopify orders. Read-only — no interrupt required.
    Still gates on COFOUNDER mode because it touches customer data
    and we never call external APIs in JARVIS mode.
    """
    assert_cofounder_or_raise("get_open_orders")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_base_url()}/orders.json",
            params={"status": "open", "limit": 50},
            headers=_shopify_headers(),
        )
        resp.raise_for_status()
        orders = resp.json().get("orders", [])

    return {
        "executed": True,
        "order_count": len(orders),
        "orders": [
            {
                "id":    o["id"],
                "name":  o["name"],
                "total": o["total_price"],
                "created_at": o["created_at"],
            }
            for o in orders
        ],
    }
