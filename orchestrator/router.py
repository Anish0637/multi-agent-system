"""
Orchestrator router — thin shim that delegates to the LangGraph graph.
Kept for backward compatibility so api/server.py and main.py need no changes.
"""
from orchestrator.graph import route, OrchestratorResult  # noqa: F401

__all__ = ["route", "OrchestratorResult"]
