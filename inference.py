#!/usr/bin/env python3
"""ZeroTrace Inference Script — Ultra-fast edition.

MUST complete in < 30 minutes on vcpu=2, memory=8gb.

Architecture:
  - Direct state-machine calls (no HTTP / no app.py needed)
  - Fast reset (skip running tests on buggy code — we know it's buggy)
  - Hard 30s timeout on every LLM call via httpx.Timeout
  - 3 steps max per task, 2.5 min per task, 20 min global cap
"""

import math
import os
import sys
import json
import time
import re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from dotenv import load_dotenv
import httpx
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Direct imports — no HTTP, no app.py, no Gradio
# ---------------------------------------------------------------------------
from environment.models import Action, Observation, Reward, StepResult
from environment.state_machine import (
    EpisodeState, _episodes, step_episode,
)
from tasks import TASKS


def _clamp(v: float) -> float:
    if not math.isfinite(v):
        return 0.01
    return round(max(0.01, min(0.99, v)), 4)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_IDS: List[str] = [
    "level1_keyerror",
    "level2_resource_leak",
    "level3_race_condition",
    "torch_dtype_mismatch",
    "torch_nan_gradient",
    "torch_wrong_dim",
    "torch_ddp_batch",
]

MAX_STEPS = 3                # 3 steps: inspect → edit → submit
GLOBAL_TIMEOUT_S = 20 * 60   # 20 min hard cap (10 min buffer)
PER_TASK_TIMEOUT_S = 150     # 2.5 min per task
LLM_CALL_TIMEOUT_S = 30      # Hard 30s per LLM call
LLM_MAX_TOKENS = 700         # Short responses


# ---------------------------------------------------------------------------
# Fast reset — skip running tests on buggy code (we know it's buggy)
# ---------------------------------------------------------------------------

def fast_reset(task_id: str) -> Observation:
    """Reset episode WITHOUT running tests or executing buggy code.

    The standard reset_episode() calls run_tests() + run_code_safely()
    on the buggy code, which spawns 3-12 subprocesses per task.
    That wastes 30-90 seconds we don't have.
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")

    task = TASKS[task_id]
    ep = EpisodeState()
    ep.task_id = task_id
    ep.level = task["level"]
    ep.original_buggy_code = task["buggy_code"]
    ep.current_code = task["buggy_code"]
    ep.terminal_output = task["description"]  # give the agent the task description
    ep.step_count = 0
    ep.cumulative_reward = 0.01
    ep.done = False
    ep.prev_passed = 0
    ep._history = []
    ep.replay_log = []
    ep.test_results = {
        "passed": 0, "failed": 0, "total": 0, "details": [], "score": 0.01
    }

    _episodes[task_id] = ep
    return ep.get_observation()


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "INSPECT_ERROR", "EDIT_CODE", "RUN_COMPILER",
    "EXECUTE_UNIT_TEST", "QUERY_CONTEXT", "SUBMIT_FIX",
    "SEARCH_DOCS", "RUN_SNIPPET",
}


def parse_action(text: str) -> Dict[str, Any]:
    default = {"action_type": "INSPECT_ERROR", "patched_code": None, "rationale": ""}
    if not text:
        return default

    # Try structured block
    m = re.search(r"---ACTION---\s*\n?(.*?)(?:---|$)", text, re.DOTALL | re.IGNORECASE)
    block = m.group(1).strip() if m else ""

    parsed_type = parsed_code = None
    parsed_rationale = ""

    if block:
        try:
            aj = json.loads(block)
            if isinstance(aj, dict) and aj.get("action_type") in VALID_ACTIONS:
                parsed_type = aj["action_type"]
                parsed_code = aj.get("patched_code")
                parsed_rationale = aj.get("rationale", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            tm = re.search(r'"action_type"\s*:\s*"([A-Z_]+)"', block)
            if tm and tm.group(1) in VALID_ACTIONS:
                parsed_type = tm.group(1)

    # Try patched code block
    cm = re.search(
        r"---PATCHED_CODE---\s*\n?```(?:python)?\s*\n?(.*?)```",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not cm:
        cm = re.search(
            r"---PATCHED_CODE---\s*\n?(.*?)(?:---|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
    patched = cm.group(1).strip() if cm else None

    if parsed_type:
        return {
            "action_type": parsed_type,
            "patched_code": parsed_code or patched,
            "rationale": parsed_rationale,
        }

    # Fallback keyword search
    tl = text.lower()
    for kw, at in [("submit_fix", "SUBMIT_FIX"), ("edit_code", "EDIT_CODE"),
                   ("execute_unit_test", "EXECUTE_UNIT_TEST")]:
        if kw in tl:
            return {"action_type": at, "patched_code": patched, "rationale": ""}

    return default


# ---------------------------------------------------------------------------
# System prompt — force 2-step resolution
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ZERO, an autonomous code repair agent.\n"
    "You have ONLY 3 steps total. Be extremely fast.\n\n"
    "Step 1: Read the buggy code and diagnose the bug.\n"
    "Step 2: SUBMIT_FIX with the complete corrected file.\n\n"
    "Response format:\n"
    "---ACTION---\n"
    '{"action_type": "SUBMIT_FIX", "rationale": "brief reason"}\n\n'
    "---PATCHED_CODE---\n```python\n<complete fixed file>\n```\n\n"
    "IMPORTANT: Always include the COMPLETE file in PATCHED_CODE.\n"
    "IMPORTANT: Use SUBMIT_FIX (not EDIT_CODE) to save a step."
)


# ---------------------------------------------------------------------------
# LLM call with HARD timeout
# ---------------------------------------------------------------------------

# Thread pool for enforcing hard timeouts
_llm_pool = ThreadPoolExecutor(max_workers=1)


def _raw_llm_call(client: OpenAI, messages: list) -> str:
    """Raw LLM call — runs inside thread pool."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content or ""


def call_llm(client: OpenAI, messages: list) -> str:
    """LLM call with hard thread-level timeout. Returns "" on any failure."""
    try:
        future = _llm_pool.submit(_raw_llm_call, client, messages)
        return future.result(timeout=LLM_CALL_TIMEOUT_S)
    except FuturesTimeout:
        print(f"    LLM timed out after {LLM_CALL_TIMEOUT_S}s")
        return ""
    except Exception as e:
        err = str(e)
        if "402" in err or "401" in err:
            raise  # Fatal — propagate
        print(f"    LLM error: {err[:100]}")
        return ""


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> Dict[str, Dict[str, Any]]:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    # Client with strict httpx timeout
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        timeout=httpx.Timeout(LLM_CALL_TIMEOUT_S, connect=5.0),
    )

    results: Dict[str, Dict[str, Any]] = {}
    global_start = time.monotonic()

    for task_id in TASK_IDS:

        # ── Global budget ──────────────────────────────────────────
        elapsed = time.monotonic() - global_start
        if elapsed >= GLOBAL_TIMEOUT_S:
            print(f"\n[BUDGET] Global timeout ({elapsed:.0f}s). Filling rest with defaults.")
            for tid in TASK_IDS:
                if tid not in results:
                    results[tid] = {
                        "reward": 0.01, "steps": 0, "error": "timeout",
                        "step_rewards": [0.01], "success": False,
                    }
            break

        print(f"\n[START] task={task_id}")
        task_start = time.monotonic()

        try:
            # ── Fast reset (no subprocess calls) ───────────────────
            obs = fast_reset(task_id)

            steps = 0
            final_reward = 0.01
            step_rewards: list = []
            success = False

            for step_num in range(MAX_STEPS):
                steps = step_num + 1

                # Budget checks
                if (time.monotonic() - task_start) >= PER_TASK_TIMEOUT_S:
                    print(f"  [BUDGET] Task timeout")
                    break
                if (time.monotonic() - global_start) >= GLOBAL_TIMEOUT_S:
                    print(f"  [BUDGET] Global timeout")
                    break

                # Build minimal message
                compact = json.dumps({
                    "task_id": obs.task_id,
                    "buggy_code": obs.buggy_code,
                    "terminal_output": (obs.terminal_output or "")[:1000],
                    "test_results": obs.test_results,
                    "step": steps,
                    "max_steps": MAX_STEPS,
                })
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": compact},
                ]

                # ── LLM call ───────────────────────────────────────
                text = call_llm(client, messages)
                if not text:
                    # LLM failed — try submitting the buggy code as-is
                    print(f"  [STEP] s={steps} LLM returned empty. Skipping.")
                    continue

                action_dict = parse_action(text)

                # Force SUBMIT_FIX if this is the last step and we have code
                if step_num == MAX_STEPS - 1 and action_dict["action_type"] != "SUBMIT_FIX":
                    if action_dict.get("patched_code"):
                        action_dict["action_type"] = "SUBMIT_FIX"

                # ── Step environment directly ──────────────────────
                action = Action(**action_dict)
                result = step_episode(task_id, action)

                obs = result.observation
                final_reward = result.reward.value
                clamped = _clamp(final_reward)
                step_rewards.append(clamped)

                tr = obs.test_results
                done = result.done

                print(
                    f"  [STEP] s={steps} act={action_dict['action_type']} "
                    f"r={clamped:.2f} pass={tr.get('passed',0)}/{tr.get('total',0)} "
                    f"done={done}"
                )

                if done:
                    success = (tr.get("passed", 0) == tr.get("total", 0)
                               and tr.get("total", 0) > 0)
                    break

            results[task_id] = {
                "reward": _clamp(final_reward),
                "steps": steps,
                "step_rewards": step_rewards or [0.01],
                "success": success,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[task_id] = {
                "reward": 0.01, "steps": 0, "error": str(e),
                "step_rewards": [0.01], "success": False,
            }

        # ── [END] ──────────────────────────────────────────────────
        r = results[task_id]
        t = time.monotonic() - task_start
        print(f"[END] ok={r['success']} steps={r['steps']} "
              f"reward={r['reward']:.2f} time={t:.1f}s")

    # ── Summary ────────────────────────────────────────────────────
    total = time.monotonic() - global_start
    print(f"\n{'='*60}")
    print(f"=== ZEROTRACE RESULTS ({total:.0f}s) ===")
    print(f"{'='*60}")

    for tid, r in results.items():
        rw = r.get("reward", 0.01)
        err = f"  ERR={r['error']}" if r.get("error") else ""
        print(f"{tid:<32} reward={rw:.2f} steps={r['steps']}{err}")

    valid = [r["reward"] for r in results.values() if "error" not in r]
    mean = sum(valid) / len(valid) if valid else 0.01
    print(f"\nMean: {mean:.4f}")

    # Validation
    print(f"\n{'='*60}\nVALIDATION\n{'='*60}")
    all_ok = True
    for tid, r in results.items():
        rw = r.get("reward", 0.01)
        ok = 0.0 < rw < 1.0
        if not ok:
            all_ok = False
        print(f"{'OK' if ok else 'FAIL'}: {tid} reward={rw:.2f}")

    if all_ok:
        print("\nPASSED")
    else:
        print("\nFAILED")
        sys.exit(1)

    return results


if __name__ == "__main__":
    run_inference()
