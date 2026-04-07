"""FastAPI service exposing OpenEnv-style reset/step/state endpoints."""

from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import EmailTriageEnv, Action


class ResetRequest(BaseModel):
    """Request body for reset endpoint."""

    task: str = Field(default="easy", pattern="^(easy|medium|hard)$")


app = FastAPI(title="Email Triage OpenEnv API", version="1.0.0")

# Single in-memory environment instance for a simple Space deployment.
_current_env = EmailTriageEnv(task="easy")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "email-triage-v1",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> Dict[str, Any]:
    global _current_env
    _current_env = EmailTriageEnv(task=req.task)
    result = _current_env.reset()
    return {
        "observation": result.observation.model_dump(),
        "info": result.info,
    }


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    try:
        result = _current_env.step(action)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return _current_env.state()
