"""
Multi-Agent System — FastAPI server.

Single endpoint: POST /agent
  → orchestrator picks specialist
  → specialist generates AgentAction
  → executor calls downstream service
  → returns unified response
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from orchestrator.router import route
from executor.runner import execute
from agents.registry import all_agents


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load all skill files on startup so first request isn't slow
    for agent in all_agents():
        try:
            _ = agent.system_prompt
        except FileNotFoundError as e:
            print(f"[warn] Could not load skill for {agent.name}: {e}")
    yield


app = FastAPI(
    title="Multi-Agent System",
    description="Orchestrator + specialist agents — each skill is a worker",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentRequest(BaseModel):
    message:   str
    tenant_id: str  = "default"
    user_id:   str  = "anonymous"


class AgentReply(BaseModel):
    routed_to:   str
    action_taken: dict | None
    result:      dict | str
    success:     bool
    error:       str | None = None


@app.get("/health")
def health():
    agents = [{"name": a.name, "description": a.description} for a in all_agents()]
    return {"status": "ok", "agents": agents}


@app.post("/agent", response_model=AgentReply)
def agent_endpoint(req: AgentRequest):
    """
    Send a natural language message. The orchestrator routes it to the
    correct specialist, which generates an action, which is executed.
    """
    # Enrich message with scope so orchestrator can extract it
    enriched = f"[tenant_id={req.tenant_id}, user_id={req.user_id}] {req.message}"

    try:
        # 1. Orchestrate — pick specialist + generate action
        orch_result = route(enriched)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {e}")

    agent_resp = orch_result.agent_response
    if agent_resp.error:
        return AgentReply(
            routed_to=orch_result.routed_to,
            action_taken=None,
            result={},
            success=False,
            error=agent_resp.error,
        )

    action = agent_resp.action

    # 2. Execute — call downstream service
    exec_result = execute(action, service_name=orch_result.routed_to)

    return AgentReply(
        routed_to=orch_result.routed_to,
        action_taken={
            "method": action.method,
            "path":   action.path,
            "body":   action.body,
        } if action and not action.direct_answer else None,
        result=exec_result.body,
        success=exec_result.success,
        error=exec_result.error,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8002, reload=True)
