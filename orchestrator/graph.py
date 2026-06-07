"""
LangGraph-based orchestrator.

Graph topology:
  START
    │
    ▼
  classify          ← GPT-4o picks specialist via function calling
    │
    ├─► rag_memory  ← RagMemoryAgent.run()
    ├─► billing     ← BillingAgent.run()
    ├─► hr          ← HRAgent.run()
    └─► general     ← GeneralAgent.run()
          │
          ▼
         END

The graph produces an AgentResponse.  The HTTP execution step remains
outside the graph (in api/server.py and executor/runner.py) so the
two concerns stay separate.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TypedDict

from openai import OpenAI

from agents.base import AgentResponse
from agents.registry import agent_tool_definitions, get_agent
from langgraph.graph import StateGraph, END


# ── Graph state ───────────────────────────────────────────────

class GraphState(TypedDict):
    user_message:   str
    context:        dict
    routed_to:      str | None
    agent_response: AgentResponse | None
    error:          str | None


# ── OrchestratorResult (same public interface as old router.py) ──

@dataclass
class OrchestratorResult:
    routed_to:      str
    agent_response: AgentResponse
    tool_call_id:   str
    context:        dict = field(default_factory=dict)


# ── OpenAI client ─────────────────────────────────────────────

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# ── Node: classify ────────────────────────────────────────────

_ORCHESTRATOR_SYSTEM = """
You are an intelligent routing orchestrator for a multi-agent system.

Your ONLY job is to analyse the user's intent and call exactly one specialist tool.
Never answer directly — always call a tool.

Available specialists:
- rag-memory : ingest documents / text, query knowledge, manage typed memory,
               GDPR erase, retention sweeps
- billing    : invoices, subscription plans, payment methods, refunds, charges
- hr         : leave policies, benefits, onboarding, performance reviews,
               compensation, compliance, offboarding
- general    : greetings, general questions, system capabilities, anything else

When routing, extract context from the message:
  - tenant_id  (default "default")
  - user_id    (default "anonymous")
"""


def classify_node(state: GraphState) -> dict:
    """Use GPT-4o function calling to pick the right specialist."""
    tools    = agent_tool_definitions()
    messages = [
        {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
        {"role": "user",   "content": state["user_message"]},
    ]
    resp = _get_client().chat.completions.create(
        model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
        messages=messages,
        tools=tools,
        tool_choice="required",
        temperature=0,
    ).choices[0].message

    if not resp.tool_calls:
        return {"routed_to": "general", "context": {}, "error": "No tool call returned"}

    tool_call  = resp.tool_calls[0]
    args       = json.loads(tool_call.function.arguments)
    agent_name = tool_call.function.name.replace("route_to_", "").replace("_", "-")
    context    = args.get("context") or {}
    return {"routed_to": agent_name, "context": context, "error": None}


# ── Conditional edge ──────────────────────────────────────────

_KNOWN_NODES = {"rag_memory", "billing", "hr", "general"}


def _pick_node(state: GraphState) -> str:
    """Map agent name → graph node name, fall back to 'general'."""
    name = (state.get("routed_to") or "general").replace("-", "_")
    return name if name in _KNOWN_NODES else "general"


# ── Specialist nodes ──────────────────────────────────────────

def _run_specialist(state: GraphState, agent_name: str) -> dict:
    try:
        agent = get_agent(agent_name)
        resp  = agent.run(state["user_message"], state.get("context"))
        return {"agent_response": resp}
    except Exception as exc:
        return {
            "agent_response": AgentResponse(
                agent_name=agent_name, raw_output="", error=str(exc)
            )
        }


def rag_memory_node(state: GraphState) -> dict:
    return _run_specialist(state, "rag-memory")

def billing_node(state: GraphState) -> dict:
    return _run_specialist(state, "billing")

def hr_node(state: GraphState) -> dict:
    return _run_specialist(state, "hr")

def general_node(state: GraphState) -> dict:
    return _run_specialist(state, "general")


# ── Build & compile graph ─────────────────────────────────────

def _build_graph():
    g = StateGraph(GraphState)

    g.add_node("classify",   classify_node)
    g.add_node("rag_memory", rag_memory_node)
    g.add_node("billing",    billing_node)
    g.add_node("hr",         hr_node)
    g.add_node("general",    general_node)

    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        _pick_node,
        {
            "rag_memory": "rag_memory",
            "billing":    "billing",
            "hr":         "hr",
            "general":    "general",
        },
    )
    for node in ("rag_memory", "billing", "hr", "general"):
        g.add_edge(node, END)

    return g.compile()


_graph = _build_graph()


# ── Public API (same interface as old router.py) ──────────────

def route(user_message: str) -> OrchestratorResult:
    """Route a user message through the LangGraph pipeline."""
    state = _graph.invoke({
        "user_message":   user_message,
        "context":        {},
        "routed_to":      None,
        "agent_response": None,
        "error":          None,
    })

    agent_resp = state["agent_response"] or AgentResponse(
        agent_name="unknown",
        raw_output="",
        error=state.get("error") or "No agent response",
    )

    return OrchestratorResult(
        routed_to=state.get("routed_to") or "unknown",
        agent_response=agent_resp,
        tool_call_id="",
        context=state.get("context", {}),
    )
