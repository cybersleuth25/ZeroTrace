"""Prompt builder for ZeroTrace agent.

Supports:
  - Single-turn: build_prompt(obs)
  - Multi-turn:  build_messages(obs, history) → full messages list
"""

import json
from typing import Any, Dict, List, Tuple

from environment.models import Observation

# ---------------------------------------------------------------------------
# System prompt (shared)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are ZERO, an autonomous code repair agent. Fix Python and PyTorch bugs efficiently.

## Your Actions
- INSPECT_ERROR      : Re-run the current code to see error messages.
- EDIT_CODE          : Modify the code. You MUST include patched_code.
- RUN_COMPILER       : Check syntax (use only if you suspect a syntax error).
- EXECUTE_UNIT_TEST  : Run the test suite to score your current fix.
- QUERY_CONTEXT      : Get the task description and a hint.
- SEARCH_DOCS        : Search offline Python/PyTorch docs. Set rationale = your query.
- RUN_SNIPPET        : Run a short verification snippet. Set patched_code = snippet.
- SUBMIT_FIX         : Submit your final fix. Include patched_code.

## Response Format
You MUST use this exact structure:

---DIAGNOSIS---
[What is the bug and why does it cause problems]

---CHAIN-OF-THOUGHT---
[Step-by-step reasoning]

---ACTION---
{"action_type": "ACTION_NAME", "patched_code": "FULL_CODE_OR_SNIPPET", "rationale": "Why"}

---PATCHED_CODE---
```python
[Full fixed code if action is EDIT_CODE or SUBMIT_FIX]
```

## Strategy
1. INSPECT_ERROR to see the failure.
2. EDIT_CODE with your complete fix.
3. EXECUTE_UNIT_TEST to verify.
4. SUBMIT_FIX when all tests pass.

## Rules
- patched_code must be the COMPLETE file, not a diff or snippet.
- Handle all edge cases (empty inputs, zeros, missing keys, etc.).
- For SEARCH_DOCS, put your query in "rationale".
- SUBMIT_FIX only when tests are green.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompt(obs: Observation) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for single-turn inference.

    Args:
        obs: Current observation from the environment.

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    return _SYSTEM_PROMPT, _format_user_turn(obs)


def build_messages(
    obs: Observation,
    history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build a full messages list for multi-turn inference.

    Includes up to the last 5 conversation turns so the agent remembers
    what it has already tried.

    Args:
        obs:     Current observation.
        history: List of {"role": "user"|"assistant", "content": str} dicts
                 maintained by the caller (app.py).

    Returns:
        List of message dicts ready to pass to client.chat_completion().
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT}
    ]
    # Include last 5 turns (10 messages: 5 user + 5 assistant)
    for turn in history[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    # Always append the current state as the latest user message
    messages.append({"role": "user", "content": _format_user_turn(obs)})
    return messages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_user_turn(obs: Observation) -> str:
    """Format the current observation as a user-turn message."""
    passed = obs.test_results.get("passed", 0)
    total = obs.test_results.get("total", 0)
    details = obs.test_results.get("details", [])

    lines = [
        f"## Current State",
        f"",
        f"| Field        | Value |",
        f"|--------------|-------|",
        f"| Task         | `{obs.task_id}` |",
        f"| Level        | {obs.level} |",
        f"| Step         | {obs.step_count} |",
        f"| Tests Passed | {passed}/{total} |",
        f"| Score        | {obs.reward:.2f} |",
        f"| Done         | {obs.done} |",
        f"",
        f"## Current Code",
        f"```python",
        obs.buggy_code,
        f"```",
        f"",
        f"## Terminal Output",
        f"```",
        obs.terminal_output if obs.terminal_output else "No output yet. Use INSPECT_ERROR.",
        f"```",
        f"",
        f"## Test Details",
    ]
    for d in details:
        lines.append(f"- {d}")
    if not details:
        lines.append("_(No test results yet — use EXECUTE_UNIT_TEST)_")

    lines += ["", "What action will you take to fix this bug?"]
    return "\n".join(lines)
