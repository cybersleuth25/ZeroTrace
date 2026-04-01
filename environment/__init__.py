"""Environment module for ZeroTrace.

Provides the core environment functionality including:
- Pydantic models for observation, action, and reward spaces
- Sandbox for safe code execution
- Test runner with graders for each task level
- Episode state machine for managing agent interactions
- FastAPI endpoints for the OpenEnv API
"""

from .models import (
    Observation,
    Action,
    Reward,
    StepResult,
    ResetRequest,
    StepRequest,
)
from .sandbox import run_code_safely, check_syntax
from .test_runner import run_tests, grade_level1, grade_level2, grade_level3
from .state_machine import (
    EpisodeState,
    compute_reward,
    reset_episode,
    step_episode,
    get_episode_state,
)
from .env import api_app

__all__ = [
    "Observation",
    "Action",
    "Reward",
    "StepResult",
    "ResetRequest",
    "StepRequest",
    "run_code_safely",
    "check_syntax",
    "run_tests",
    "grade_level1",
    "grade_level2",
    "grade_level3",
    "EpisodeState",
    "compute_reward",
    "reset_episode",
    "step_episode",
    "get_episode_state",
    "api_app",
]
