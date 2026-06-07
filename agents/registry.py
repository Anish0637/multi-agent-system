"""
Agent registry — single source of truth for all specialist agents.
Add new agents here; the orchestrator discovers them automatically.
"""
from __future__ import annotations

from agents.base import BaseAgent
from agents.rag_agent import RagMemoryAgent
from agents.billing_agent import BillingAgent
from agents.hr_agent import HRAgent
from agents.general_agent import GeneralAgent

# ── Register specialists here ─────────────────────────────────

_REGISTRY: dict[str, BaseAgent] = {}


def _register(agent: BaseAgent) -> None:
    _REGISTRY[agent.name] = agent


_register(RagMemoryAgent())
_register(BillingAgent())
_register(HRAgent())
_register(GeneralAgent())


def get_agent(name: str) -> BaseAgent:
    if name not in _REGISTRY:
        raise KeyError(f"No agent registered with name '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def all_agents() -> list[BaseAgent]:
    return list(_REGISTRY.values())


def agent_tool_definitions() -> list[dict]:
    """
    Build OpenAI function-calling tool definitions from the registry.
    The orchestrator uses these to select the right specialist.
    """
    tools = []
    for agent in all_agents():
        tools.append({
            "type": "function",
            "function": {
                "name":        f"route_to_{agent.name.replace('-', '_')}",
                "description": agent.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_message": {
                            "type":        "string",
                            "description": "The exact user message to forward to this specialist.",
                        },
                        "context": {
                            "type":        "object",
                            "description": "Optional extracted context (tenant_id, user_id, etc.)",
                        },
                    },
                    "required": ["user_message"],
                },
            },
        })
    return tools
