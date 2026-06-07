"""Tests for the multi-agent system."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from agents.base import AgentAction, AgentResponse
from agents.registry import get_agent, all_agents, agent_tool_definitions
from executor.runner import execute, ExecutorResult


# ── Registry ──────────────────────────────────────────────────
def test_registry_has_rag_agent():
    agent = get_agent("rag-memory")
    assert agent.name == "rag-memory"


def test_registry_raises_on_unknown():
    with pytest.raises(KeyError):
        get_agent("does-not-exist")


def test_tool_definitions_generated():
    tools = agent_tool_definitions()
    assert len(tools) >= 1
    names = [t["function"]["name"] for t in tools]
    assert "route_to_rag_memory" in names


# ── Executor ──────────────────────────────────────────────────
def test_executor_direct_answer():
    action = AgentAction(method="GET", path="/unused", direct_answer="42 is the answer.")
    result = execute(action, service_name="rag-memory")
    assert result.success is True
    assert result.body == {"answer": "42 is the answer."}


def test_executor_unknown_service():
    action = AgentAction(method="GET", path="/health")
    result = execute(action, service_name="nonexistent-service")
    assert result.success is False
    assert "nonexistent-service" in result.error


@patch("executor.runner.httpx.Client")
def test_executor_get_call(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok"}
    mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp

    action = AgentAction(method="GET", path="/health")
    result = execute(action, service_name="rag-memory")
    assert result.success is True
    assert result.body == {"status": "ok"}


@patch("executor.runner.httpx.Client")
def test_executor_post_call(mock_client_cls):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"stored": 3, "deduped": 0}
    mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp

    action = AgentAction(
        method="POST",
        path="/api/rag/ingest/text",
        body={"text": "hello", "source_name": "test", "tenant_id": "default"},
    )
    result = execute(action, service_name="rag-memory")
    assert result.success is True
    assert result.body["stored"] == 3


@patch("executor.runner.httpx.Client")
def test_executor_service_down(mock_client_cls):
    import httpx as _httpx
    mock_client_cls.return_value.__enter__.return_value.get.side_effect = \
        _httpx.ConnectError("refused")

    action = AgentAction(method="GET", path="/health")
    result = execute(action, service_name="rag-memory")
    assert result.success is False
    assert result.status_code == 503


# ── Agent base ────────────────────────────────────────────────
@patch("agents.base._get_client")
def test_agent_run_returns_action(mock_get_client):
    import json
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = json.dumps({
        "method": "POST",
        "path": "/api/rag/query",
        "body": {"query": "test", "tenant_id": "default", "user_id": "u1"},
        "params": None,
        "direct_answer": None,
    })
    mock_get_client.return_value.chat.completions.create.return_value = mock_completion

    agent = get_agent("rag-memory")
    resp  = agent.run("What are vehicle requirements?")
    assert resp.error is None
    assert resp.action.method == "POST"
    assert resp.action.path == "/api/rag/query"
