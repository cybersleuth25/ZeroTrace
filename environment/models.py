"""Pydantic models for ZeroTrace environment.

These models define the observation space, action space, and reward structure
for the OpenEnv-compliant autonomous code repair benchmark.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional


class Observation(BaseModel):
    """What the agent sees at each step."""

    task_id: str = Field(..., description="Unique identifier for the current task")
    level: int = Field(..., ge=1, le=6, description="Difficulty level (1-6)")
    buggy_code: str = Field(..., description="The current code being worked on")
    terminal_output: str = Field(default="", description="Output from last action")
    test_results: Dict[str, Any] = Field(
        default_factory=lambda: {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "details": [],
        },
        description="Results from running unit tests",
    )
    step_count: int = Field(default=0, ge=0, description="Current step number")
    reward: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Cumulative reward"
    )
    done: bool = Field(default=False, description="Whether the episode is complete")
    # Multi-turn memory: last N conversation turns for the agent
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent conversation history for multi-turn memory",
    )


class Action(BaseModel):
    """What the agent can do."""

    action_type: Literal[
        "INSPECT_ERROR",
        "EDIT_CODE",
        "RUN_COMPILER",
        "EXECUTE_UNIT_TEST",
        "QUERY_CONTEXT",
        "SUBMIT_FIX",
        "SEARCH_DOCS",
        "RUN_SNIPPET",
    ] = Field(..., description="Type of action to perform")
    patched_code: Optional[str] = Field(
        default=None,
        description="The corrected code (required for EDIT_CODE, SUBMIT_FIX, RUN_SNIPPET)",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Explanation / query text (used as query for SEARCH_DOCS)",
    )


class Reward(BaseModel):
    """Structured reward signal."""

    value: float = Field(..., ge=-1.0, le=1.0, description="Total reward value")
    partial_credit: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Credit for partial test passes"
    )
    penalty: float = Field(
        default=0.0, ge=-1.0, le=0.0, description="Penalty for bad behaviour"
    )
    reason: str = Field(..., description="Explanation of the reward")


class StepResult(BaseModel):
    """Full result from one step."""

    observation: Observation = Field(..., description="Current observation state")
    reward: Reward = Field(..., description="Reward from this step")
    done: bool = Field(..., description="Whether the episode is complete")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional step information"
    )


class ResetRequest(BaseModel):
    """Request to reset the environment to a specific task."""

    task_id: str = Field(..., description="Task ID to reset to")


class StepRequest(BaseModel):
    """Request to take an action in the environment."""

    model_config = {"protected_namespaces": ()}

    task_id: str = Field(..., description="Task ID for this step")
    action: Action = Field(..., description="Action to take")
    model_name: Optional[str] = Field(
        default=None,
        description="Optional: model name for leaderboard recording",
    )
