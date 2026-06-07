"""
Base agent ABC.
Every specialist inherits from this — it handles LLM calls,
skill loading, and structured action output.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from skills.loader import load_skill_with_subskills

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


class AgentAction(BaseModel):
    """Structured output from a specialist — describes what HTTP call to make."""
    method:   str            # GET | POST | DELETE | PUT
    path:     str            # e.g. /api/rag/query
    body:     dict | None    = None
    params:   dict | None    = None
    direct_answer: str | None = None  # if no API call needed, answer directly


class AgentResponse(BaseModel):
    agent_name:  str
    action:      AgentAction | None = None
    raw_output:  str
    error:       str | None = None


class BaseAgent(ABC):
    """
    Abstract specialist agent.
    Subclasses declare their name, skill paths, and action schema.
    """

    name: str
    description: str
    skill_path: str
    subskill_paths: list[str] = []

    _SYSTEM_SUFFIX = """
## Output Format

You MUST respond with a JSON object matching this schema:
{
  "method":        "GET" | "POST" | "DELETE" | "PUT",
  "path":          "/api/...",
  "body":          {...} or null,
  "params":        {"key": "value"} or null,
  "direct_answer": "string if no API call needed, else null"
}

If the user's request can be answered directly without an API call, set
direct_answer to your answer and set method/path to null.
Only output the JSON object — no markdown, no explanation.
"""

    @property
    def system_prompt(self) -> str:
        skill_text = load_skill_with_subskills(self.skill_path, *self.subskill_paths)
        return f"{skill_text}\n\n{self._SYSTEM_SUFFIX}"

    def run(self, user_message: str, context: dict | None = None) -> AgentResponse:
        """
        Run the specialist with the given user message.
        Returns a structured AgentResponse with either an action or direct answer.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context: {json.dumps(context)}",
            })
        messages.append({"role": "user", "content": user_message})

        try:
            resp = _get_client().chat.completions.create(
                model=os.getenv("SPECIALIST_MODEL", "gpt-4o"),
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
            action = AgentAction(**data)
            return AgentResponse(agent_name=self.name, action=action, raw_output=raw)
        except Exception as e:
            return AgentResponse(agent_name=self.name, raw_output="", error=str(e))
