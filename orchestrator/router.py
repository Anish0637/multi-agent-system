"""
Orchestrator — the central router.

Flow:
  1. Receive user message
  2. Call GPT-4o with all specialist tool definitions (function calling)
  3. GPT selects the right specialist via tool_choice
  4. Forward message to that specialist
  5. Specialist returns AgentAction (what HTTP call to make)
  6. Action executor makes the call
  7. Return final result
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from agents.base import AgentResponse
from agents.registry import agent_tool_definitions, get_agent

_ORCHESTRATOR_SYSTEM = """
You are an intelligent routing orchestrator for a multi-agent system.

Your ONLY job is to analyse the user's intent and call the correct specialist
tool. You must ALWAYS call exactly one tool — never answer directly.

Available specialists:
- rag-memory : ingest docs, query knowledge, manage memory, GDPR erase

When routing, extract any relevant context from the message:
  - tenant_id (company/team scope, default "default")
  - user_id   (individual user, default "anonymous")
  - file paths, document content, query text

Pass these as the `context` parameter alongside `user_message`.
"""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@dataclass
class OrchestratorResult:
    routed_to:    str
    agent_response: AgentResponse
    tool_call_id: str
    context:      dict = field(default_factory=dict)


def route(user_message: str) -> OrchestratorResult:
    """
    Route a user message to the appropriate specialist.
    Returns the specialist's AgentResponse ready for the executor.
    """
    tools   = agent_tool_definitions()
    messages = [
        {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
        {"role": "user",   "content": user_message},
    ]

    # Step 1: Orchestrator selects specialist
    resp: ChatCompletionMessage = _get_client().chat.completions.create(
        model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
        messages=messages,
        tools=tools,
        tool_choice="required",   # must always call a tool
        temperature=0,
    ).choices[0].message

    if not resp.tool_calls:
        raise RuntimeError("Orchestrator returned no tool call — check tool definitions.")

    tool_call  = resp.tool_calls[0]
    tool_name  = tool_call.function.name                     # e.g. route_to_rag_memory
    args       = json.loads(tool_call.function.arguments)
    agent_name = tool_name.replace("route_to_", "").replace("_", "-")  # → rag-memory
    forwarded  = args.get("user_message", user_message)
    context    = args.get("context") or {}

    # Step 2: Specialist processes the request
    specialist    = get_agent(agent_name)
    agent_response = specialist.run(forwarded, context)

    return OrchestratorResult(
        routed_to=agent_name,
        agent_response=agent_response,
        tool_call_id=tool_call.id,
        context=context,
    )
