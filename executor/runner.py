"""
Action executor — takes a specialist's AgentAction and calls the real API.

Design:
  - Each downstream service has a base URL mapping
  - Executor makes the HTTP call via httpx
  - Returns raw JSON response + structured ExecutorResult
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from agents.base import AgentAction

# ── Service base URL registry ─────────────────────────────────
# Add new services here as you add agents.
_SERVICE_URLS: dict[str, str] = {
    "rag-memory": os.getenv("RAG_MEMORY_URL", "http://localhost:8001"),
    # "billing":  os.getenv("BILLING_URL",    "http://localhost:8002"),
    # "crm":      os.getenv("CRM_URL",        "http://localhost:8003"),
}

_TIMEOUT = httpx.Timeout(30.0)


@dataclass
class ExecutorResult:
    status_code: int
    body:        dict | str
    success:     bool
    error:       str | None = None


def execute(action: AgentAction, service_name: str) -> ExecutorResult:
    """
    Execute an AgentAction against the appropriate downstream service.
    """
    # If specialist answered directly (no API call needed)
    if action.direct_answer is not None:
        return ExecutorResult(
            status_code=200,
            body={"answer": action.direct_answer},
            success=True,
        )

    if not action.method or not action.path:
        return ExecutorResult(
            status_code=400,
            body={},
            success=False,
            error="Agent returned no method/path and no direct_answer",
        )

    base_url = _SERVICE_URLS.get(service_name)
    if not base_url:
        return ExecutorResult(
            status_code=500,
            body={},
            success=False,
            error=f"No service URL registered for '{service_name}'",
        )

    url    = f"{base_url}{action.path}"
    method = action.method.upper()

    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            if method == "GET":
                r = client.get(url, params=action.params or {})
            elif method == "POST":
                r = client.post(url, json=action.body or {})
            elif method == "DELETE":
                r = client.request("DELETE", url, json=action.body or {})
            elif method == "PUT":
                r = client.put(url, json=action.body or {})
            else:
                raise ValueError(f"Unsupported method: {method}")

        try:
            body = r.json()
        except Exception:
            body = r.text

        return ExecutorResult(
            status_code=r.status_code,
            body=body,
            success=r.status_code < 400,
        )

    except httpx.ConnectError as e:
        return ExecutorResult(
            status_code=503,
            body={},
            success=False,
            error=f"Service unreachable at {base_url}: {e}",
        )
    except Exception as e:
        return ExecutorResult(
            status_code=500,
            body={},
            success=False,
            error=str(e),
        )
