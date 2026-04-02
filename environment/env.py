"""FastAPI endpoints for ZeroTrace environment — API v1.

Versioned routes under /api/v1/:
  POST /api/v1/reset            — reset environment
  POST /api/v1/step             — take an action
  GET  /api/v1/state            — current observation
  GET  /api/v1/tasks            — list all tasks
  GET  /api/v1/leaderboard      — aggregated leaderboard
  POST /api/v1/leaderboard/record — record a completed run
  GET  /api/v1/replay/{task_id} — fetch replay log
  GET  /health                  — health check (unversioned, kept for compat)

Legacy unversioned routes (/reset, /step, /state, /tasks) are preserved as
deprecated aliases so existing inference.py continues to work unchanged.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional

from .models import (
    Observation, Action, Reward, StepResult,
    ResetRequest, StepRequest,
)
from .state_machine import (
    reset_episode, step_episode, get_episode_state,
    get_episode_replay, get_original_code,
)
from leaderboard.store import save_result, get_leaderboard
from leaderboard.replay import save_replay
from tasks import TASKS

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

api_app = FastAPI(
    title="ZeroTrace API",
    description="OpenEnv-compliant autonomous code repair benchmark",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check (unversioned — always reachable)
# ---------------------------------------------------------------------------

@api_app.get("/health", tags=["meta"])
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "project": "zerotrace", "api_version": "v1"}


# ---------------------------------------------------------------------------
# API v1 — Core environment
# ---------------------------------------------------------------------------

@api_app.post("/api/v1/reset", response_model=Observation, tags=["environment"])
async def v1_reset(request: Optional[ResetRequest] = None) -> Observation:
    """Reset the environment to a specific task."""
    task_id = request.task_id if request and request.task_id else "level1_keyerror"
    _validate_task(task_id)
    try:
        return reset_episode(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.post("/api/v1/step", response_model=StepResult, tags=["environment"])
async def v1_step(request: StepRequest) -> StepResult:
    """Take an action in the environment."""
    _validate_task(request.task_id)
    try:
        result = step_episode(request.task_id, request.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Auto-record to leaderboard when episode ends
    if result.done and request.model_name:
        try:
            save_result(
                model=request.model_name,
                task_id=request.task_id,
                reward=result.reward.value,
                steps=result.observation.step_count,
            )
            # Persist replay
            replay = get_episode_replay(request.task_id)
            if replay:
                save_replay(request.task_id, replay)
        except Exception:
            pass  # leaderboard/replay save is best-effort

    return result


@api_app.get("/api/v1/state", response_model=Observation, tags=["environment"])
async def v1_state(task_id: str = Query(...)) -> Observation:
    """Get the current observation state."""
    _validate_task(task_id)
    try:
        return get_episode_state(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.get("/api/v1/tasks", tags=["environment"])
async def v1_list_tasks() -> Dict[str, Any]:
    """List all available tasks."""
    return {
        "tasks": [
            {
                "id": tid,
                "name": t["name"],
                "level": t["level"],
                "difficulty": t["difficulty"],
                "description": (
                    t["description"][:120] + "..."
                    if len(t["description"]) > 120
                    else t["description"]
                ),
            }
            for tid, t in TASKS.items()
        ]
    }


# ---------------------------------------------------------------------------
# API v1 — Leaderboard
# ---------------------------------------------------------------------------

@api_app.get("/api/v1/leaderboard", tags=["leaderboard"])
async def v1_leaderboard() -> Dict[str, Any]:
    """Return aggregated leaderboard data."""
    return get_leaderboard()


class RecordRequest(StepRequest.__class__):
    pass


from pydantic import BaseModel

class LeaderboardRecord(BaseModel):
    model: str
    task_id: str
    reward: float
    steps: int


@api_app.post("/api/v1/leaderboard/record", tags=["leaderboard"])
async def v1_record(body: LeaderboardRecord) -> Dict[str, str]:
    """Manually record a result on the leaderboard."""
    _validate_task(body.task_id)
    if not (-1.0 <= body.reward <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"reward must be in [-1.0, 1.0], got {body.reward}"
        )
    try:
        save_result(
            model=body.model,
            task_id=body.task_id,
            reward=body.reward,
            steps=body.steps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# API v1 — Replay
# ---------------------------------------------------------------------------

@api_app.get("/api/v1/replay/{task_id}", tags=["replay"])
async def v1_replay(task_id: str) -> Dict[str, Any]:
    """Return the replay log for the current episode of a task."""
    _validate_task(task_id)
    replay = get_episode_replay(task_id)
    original = get_original_code(task_id)
    return {
        "task_id": task_id,
        "original_code": original,
        "steps": replay,
        "total_steps": len(replay),
    }


# ---------------------------------------------------------------------------
# Legacy unversioned aliases (backward-compatible with inference.py)
# ---------------------------------------------------------------------------

_DEPRECATION_HEADERS = {"X-Deprecation-Notice": "Use /api/v1/* endpoints"}


@api_app.post("/reset", response_model=Observation, tags=["legacy"],
              include_in_schema=False)
async def legacy_reset(request: Optional[ResetRequest] = None) -> Observation:
    return await v1_reset(request)


@api_app.post("/step", response_model=StepResult, tags=["legacy"],
              include_in_schema=False)
async def legacy_step(request: StepRequest) -> StepResult:
    return await v1_step(request)


@api_app.get("/state", response_model=Observation, tags=["legacy"],
             include_in_schema=False)
async def legacy_state(task_id: str = Query(...)) -> Observation:
    return await v1_state(task_id)


@api_app.get("/tasks", tags=["legacy"], include_in_schema=False)
async def legacy_tasks() -> Dict[str, Any]:
    return await v1_list_tasks()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_task(task_id: str) -> None:
    """Raise 400 if task_id is not in the allowlist."""
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: '{task_id}'. "
                   f"Valid tasks: {list(TASKS.keys())}",
        )
