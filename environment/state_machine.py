"""Episode state machine for ZeroTrace environment.

Manages the state of a code repair episode, tracking progress,
computing rewards, replay logs, and multi-turn conversation history.
"""

from typing import Any, Dict, List, Optional

from security.scanner import scan_code
from agent.tools import search_docs, run_code_snippet
from .models import Observation, Action, Reward, StepResult
from .sandbox import run_code_safely, check_syntax
from .test_runner import run_tests
from tasks import TASKS

# How many past turns to include in conversation_history sent to the agent
_HISTORY_WINDOW = 5


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def _clamp_reward_score(v: float) -> float:
    """Clamp a score/partial_credit to (0, 1) exclusive."""
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return round(v, 4)


def compute_reward(
    action_type: str,
    test_results: Dict[str, Any],
    step_count: int,
    prev_passed: int,
    code: str,
) -> Reward:
    """Compute reward based on action and test results.

    Returns:
        Reward object with value, partial_credit, penalty, reason.
    """
    passed = test_results.get("passed", 0)
    total = test_results.get("total", 0)
    partial = passed / total if total > 0 else 0.0
    penalty = -((step_count - 1) * 0.02) # Linear speed decay
    reason = ""

    # Penalties
    if action_type == "SUBMIT_FIX" and passed < total:
        penalty -= 0.3
        reason = "Submitted fix with failing tests"

    elif action_type == "RUN_COMPILER":
        syntax = check_syntax(code)
        if syntax["valid"]:
            penalty -= 0.1
            reason = "Ran compiler when no syntax error present"
        else:
            reason = "Compiler check found syntax error"

    # Full reward when all tests pass
    if passed == total and total > 0:
        return Reward(value=0.99, partial_credit=0.99, penalty=0.0,
                      reason="All tests pass")

    # Progress reward
    if passed > prev_passed and not reason:
        partial = round(passed / total, 2) if total > 0 else 0.0
        reason = f"Progress: {passed}/{total} tests passing"

    if not reason:
        reason = "No tests passing yet" if passed == 0 else f"{passed}/{total} tests passing"

    value = round(max(-1.0, min(1.0, partial + penalty)), 3)
    # Clamp value and partial_credit to open interval (0, 1)
    clamped_value = _clamp_reward_score(value) if -1.0 < value else value
    clamped_partial = _clamp_reward_score(partial)
    return Reward(value=round(value, 3), partial_credit=clamped_partial,
                  penalty=round(penalty, 2), reason=reason)


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

class EpisodeState:
    """Manages the state of a single code repair episode."""

    MAX_STEPS = 15

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.level: int = 1
        self.original_buggy_code: str = ""   # immutable — stored for diff view
        self.current_code: str = ""
        self.terminal_output: str = ""
        self.test_results: Dict[str, Any] = {
            "passed": 0, "failed": 0, "total": 0, "details": [], "score": 0.01
        }
        self.step_count: int = 0
        self.cumulative_reward: float = 0.01
        self.done: bool = False
        self.prev_passed: int = 0
        # Multi-turn memory
        self._history: List[Dict[str, Any]] = []
        # Replay log
        self.replay_log: List[Dict[str, Any]] = []

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = TASKS[task_id]
        self.task_id = task_id
        self.level = task["level"]
        self.original_buggy_code = task["buggy_code"]
        self.current_code = task["buggy_code"]
        self.terminal_output = ""
        self.step_count = 0
        self.cumulative_reward = 0.01
        self.done = False
        self.prev_passed = 0
        self._history = []
        self.replay_log = []

        self.test_results = run_tests(self.level, self.current_code)
        result = run_code_safely(self.current_code)
        self.terminal_output = result["stderr"] or result["stdout"]

        return self.get_observation()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self.get_observation(),
                reward=Reward(value=0.01, partial_credit=0.01, penalty=0.0,
                              reason="Episode already complete"),
                done=True,
                info={"error": "Episode already complete"},
            )

        self.step_count += 1
        info: Dict[str, Any] = {"action": action.action_type}

        # ── Dispatch action ──────────────────────────────────────────────────

        if action.action_type == "INSPECT_ERROR":
            result = run_code_safely(self.current_code)
            self.terminal_output = result["stderr"] or result["stdout"] or "No errors found."
            info["inspection"] = self.terminal_output

        elif action.action_type == "EDIT_CODE":
            if action.patched_code:
                # Security scan before touching the sandbox
                scan = scan_code(action.patched_code)
                if not scan.safe:
                    self.terminal_output = (
                        f"[BLOCKED] Security scan blocked this code:\n{scan.reason}"
                    )
                    info["security_blocked"] = True
                else:
                    self.current_code = action.patched_code
                    result = run_code_safely(self.current_code)
                    self.terminal_output = result["stderr"] or result["stdout"]
                    info["code_updated"] = True
            else:
                self.terminal_output = "Error: No patched_code provided for EDIT_CODE"
                info["error"] = "No patched_code provided"

        elif action.action_type == "RUN_COMPILER":
            syntax = check_syntax(self.current_code)
            if syntax["valid"]:
                self.terminal_output = "Syntax OK: No errors found."
            else:
                self.terminal_output = (
                    f"SyntaxError at line {syntax['line']}: {syntax['error']}"
                )
            info["syntax_valid"] = syntax["valid"]

        elif action.action_type == "EXECUTE_UNIT_TEST":
            self.prev_passed = self.test_results.get("passed", 0)
            self.test_results = run_tests(self.level, self.current_code)
            details = "\n".join(self.test_results.get("details", []))
            self.terminal_output = (
                f"Test Results:\n{details}\n\nScore: {self.test_results['score']}"
            )
            info["test_results"] = self.test_results

        elif action.action_type == "QUERY_CONTEXT":
            task = TASKS.get(self.task_id, {})
            self.terminal_output = (
                f"Task: {task.get('name', 'Unknown')}\n"
                f"Level: {self.level}\n"
                f"Difficulty: {task.get('difficulty', 'Unknown')}\n"
                f"Description: {task.get('description', 'No description')}\n\n"
                "Hint: Focus on the bug described and handle all edge cases."
            )
            info["context_provided"] = True

        elif action.action_type == "SEARCH_DOCS":
            query = action.rationale or action.patched_code or ""
            self.terminal_output = search_docs(query)
            info["docs_searched"] = query

        elif action.action_type == "RUN_SNIPPET":
            if action.patched_code:
                scan = scan_code(action.patched_code)
                if not scan.safe:
                    self.terminal_output = (
                        f"[BLOCKED] Security scan blocked snippet:\n{scan.reason}"
                    )
                    info["security_blocked"] = True
                else:
                    res = run_code_snippet(action.patched_code)
                    self.terminal_output = (
                        res["stdout"] or res["stderr"] or "No output."
                    )
                    info["snippet_result"] = self.terminal_output
            else:
                self.terminal_output = "Error: No code provided for RUN_SNIPPET"

        elif action.action_type == "SUBMIT_FIX":
            if action.patched_code:
                scan = scan_code(action.patched_code)
                if not scan.safe:
                    self.terminal_output = (
                        f"[BLOCKED] Security scan blocked submission:\n{scan.reason}"
                    )
                    info["security_blocked"] = True
                else:
                    self.current_code = action.patched_code

            self.prev_passed = self.test_results.get("passed", 0)
            self.test_results = run_tests(self.level, self.current_code)

            if self.test_results["passed"] == self.test_results["total"]:
                self.done = True
                self.terminal_output = (
                    f"[SUCCESS] All {self.test_results['total']} tests passed!"
                )
            else:
                failed = self.test_results["total"] - self.test_results["passed"]
                details = "\n".join(self.test_results.get("details", []))
                self.terminal_output = (
                    f"[FAILED] {failed} test(s) still failing.\n\n{details}"
                )

            info["submitted"] = True
            info["final_score"] = self.test_results["score"]

        # Max steps
        if self.step_count >= self.MAX_STEPS:
            self.done = True
            info["max_steps_reached"] = True

        reward = compute_reward(
            action_type=action.action_type,
            test_results=self.test_results,
            step_count=self.step_count,
            prev_passed=self.prev_passed,
            code=self.current_code,
        )
        self.cumulative_reward = max(-1.0, min(1.0, reward.value))

        # ── Update history & replay ─────────────────────────────────────────
        self._history.append({
            "step": self.step_count,
            "action_type": action.action_type,
            "terminal_output": self.terminal_output[:500],  # trim for prompt size
            "test_passed": self.test_results.get("passed", 0),
            "test_total": self.test_results.get("total", 0),
            "reward": self.cumulative_reward,
        })

        self.replay_log.append({
            "step": self.step_count,
            "action": {
                "action_type": action.action_type,
                "rationale": action.rationale,
            },
            "code": self.current_code,
            "terminal_output": self.terminal_output,
            "test_results": self.test_results,
            "reward": {"value": reward.value, "reason": reward.reason},
            "done": self.done,
        })

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=self.done,
            info=info,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_observation(self) -> Observation:
        recent_history = self._history[-_HISTORY_WINDOW:]
        return Observation(
            task_id=self.task_id or "",
            level=self.level,
            buggy_code=self.current_code,
            terminal_output=self.terminal_output,
            test_results=self.test_results,
            step_count=self.step_count,
            reward=self.cumulative_reward,
            done=self.done,
            conversation_history=recent_history,
        )

    def get_original_code(self) -> str:
        """Return the original buggy code before any edits (for diff view)."""
        return self.original_buggy_code


# ---------------------------------------------------------------------------
# Global episode registry (in-process, keyed by task_id)
# ---------------------------------------------------------------------------
_episodes: Dict[str, EpisodeState] = {}


def reset_episode(task_id: str) -> Observation:
    episode = EpisodeState()
    _episodes[task_id] = episode
    return episode.reset(task_id)


def step_episode(task_id: str, action: Action) -> StepResult:
    if task_id not in _episodes:
        raise ValueError(f"No active episode for task '{task_id}'. Call reset first.")
    return _episodes[task_id].step(action)


def get_episode_state(task_id: str) -> Observation:
    if task_id not in _episodes:
        raise ValueError(f"No active episode for task '{task_id}'. Call reset first.")
    return _episodes[task_id].get_observation()


def get_episode_replay(task_id: str) -> List[Dict[str, Any]]:
    """Return replay log for a task's current episode."""
    if task_id not in _episodes:
        return []
    return _episodes[task_id].replay_log


def get_original_code(task_id: str) -> str:
    """Return the original buggy code for diff computation."""
    if task_id not in _episodes:
        return ""
    return _episodes[task_id].get_original_code()
