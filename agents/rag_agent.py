"""RAG Memory specialist agent."""
from __future__ import annotations

from agents.base import BaseAgent


class RagMemoryAgent(BaseAgent):
    name        = "rag-memory"
    description = (
        "Handles all RAG memory operations: ingest documents/text/files, "
        "query facts with hybrid retrieval, manage policies and preferences, "
        "store episodes, replay traces, inspect context, GDPR erasure, "
        "retention sweeps. Backed by http://localhost:8001."
    )
    skill_path = "/Users/anishkumar/.agents/skills/rag-memory/SKILL.md"
    subskill_paths = [
        "/Users/anishkumar/.agents/skills/rag-memory/sub-skills/ingest.md",
        "/Users/anishkumar/.agents/skills/rag-memory/sub-skills/retrieval.md",
        "/Users/anishkumar/.agents/skills/rag-memory/sub-skills/gdpr.md",
    ]
