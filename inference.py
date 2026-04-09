#!/usr/bin/env python3
"""ZeroTrace Inference Script.

Runs the ZeroTrace agent on all tasks and prints structured results.
Must complete in under 30 minutes on vcpu=2, memory=8gb.

ARCHITECTURE: Calls the state machine DIRECTLY (no HTTP round-trips to app.py).
This eliminates Gradio startup, HTTP overhead, and server-side latency.

Required environment variables:
  API_BASE_URL  : The API endpoint for the LLM
  MODEL_NAME    : The model identifier
  HF_TOKEN      : Your HuggingFace API key
"""

import math
import os
import sys
import json
import time
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Direct imports — bypass HTTP, call state machine in-process
# ---------------------------------------------------------------------------
from environment.state_machine import reset_episode, step_episode
from environment.models import Action


def _clamp(v: float) -> float:
    """Clamp a score to the open interval (0, 1)."""
    if not math.isfinite(v):
        return 0.01
    clamped = max(0.01, min(0.99, v))
    return round(clamped, 4)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TASKS: List[str] = [
    "level1_keyerror",
    "level2_resource_leak",
    "level3_race_condition",
    "torch_dtype_mismatch",
    "torch_nan_gradient",
    "torch_wrong_dim",
    "torch_ddp_batch",
]

MAX_STEPS = 6                # Aggressive — 6 steps max per task
GLOBAL_TIMEOUT_S = 25 * 60   # 25 min hard cap (5 min buffer)
PER_TASK_TIMEOUT_S = 180     # 3 min per task
LLM_TIMEOUT_S = 45           # 45s per LLM call
LLM_MAX_TOKENS = 800         # Shorter responses = faster
VALID_ACTIONS = {
    "INSPECT_ERROR", "EDIT_CODE", "RUN_COMPILER",
    "EXECUTE_UNIT_TEST", "QUERY_CONTEXT", "SUBMIT_FIX",
    "SEARCH_DOCS", "RUN_SNIPPET",
}


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Dict[str, Any]:
    """Parse an action dict from LLM response text."""
    default = {
        "action_type": "INSPECT_ERROR",
        "patched_code": None,
        "rationale": "Parsing failed",
    }
    if not text:
        return default

    action_match = re.search(
        r"---ACTION---\s*\n?(.*?)(?:---|$)", text, re.DOTALL | re.IGNORECASE
    )
    action_block = action_match.group(1).strip() if action_match else ""

    parsed_type = None
    parsed_code = None
    parsed_rationale = ""

    if action_block:
        try:
            aj = json.loads(action_block)
            if isinstance(aj, dict) and aj.get("action_type") in VALID_ACTIONS:
                parsed_type = aj["action_type"]
                parsed_code = aj.get("patched_code")
                parsed_rationale = aj.get("rationale", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            m = re.search(r'"action_type"\s*:\s*"([A-Z_]+)"', action_block)
            if m and m.group(1) in VALID_ACTIONS:
                parsed_type = m.group(1)

    code_match = re.search(
        r"---PATCHED_CODE---\s*\n?```(?:python)?\s*\n?(.*?)```",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not code_match:
        code_match = re.search(
            r"---PATCHED_CODE---\s*\n?(.*?)(?:---|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
    patched_block = code_match.group(1).strip() if code_match else None

    if parsed_type:
        return {
            "action_type": parsed_type,
            "patched_code": parsed_code or patched_block,
            "rationale": parsed_rationale,
        }

    tl = text.lower()
    for kw, at in [("submit_fix", "SUBMIT_FIX"), ("edit_code", "EDIT_CODE"),
                   ("execute_unit_test", "EXECUTE_UNIT_TEST"),
                   ("search_docs", "SEARCH_DOCS"), ("run_snippet", "RUN_SNIPPET")]:
        if kw in tl:
            return {"action_type": at, "patched_code": patched_block, "rationale": ""}

    return default


# ---------------------------------------------------------------------------
# System prompt — concise, pushing for fast resolution
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return (
        "You are ZERO, an autonomous code repair agent. Fix Python/PyTorch bugs.\n"
        "Respond ONLY with the structured format below. Be extremely concise.\n\n"
        "---ACTION---\n"
        '{"action_type": "ACTION_NAME", "rationale": "brief why"}\n\n'
        "---PATCHED_CODE---\n```python\n[full fixed file]\n```\n\n"
        "Actions: INSPECT_ERROR, EDIT_CODE, SUBMIT_FIX, EXECUTE_UNIT_TEST\n"
        "Strategy: 1.INSPECT_ERROR 2.EDIT_CODE+SUBMIT_FIX (include full file in PATCHED_CODE)\n"
        "You have very few steps. Fix the bug and SUBMIT_FIX as fast as possible."
    )


# ---------------------------------------------------------------------------
# LLM call with timeout and single retry
# ---------------------------------------------------------------------------

def _call_llm(client: OpenAI, messages: list) -> str:
    """Call the LLM with timeout and one retry on transient failure."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT_S,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            if "402" in err or "payment required" in err.lower():
                raise RuntimeError("402 Payment Required") from e
            if "401" in err or "unauthorized" in err.lower():
                raise RuntimeError("401 Unauthorized. Check HF_TOKEN.") from e
            if attempt == 0:
                print(f"  LLM retry in 2s: {err[:100]}")
                time.sleep(2)
            else:
                print(f"  LLM failed: {err[:100]}")
                return ""
    return ""


# ---------------------------------------------------------------------------
# Main inference loop — DIRECT state machine calls, no HTTP
# ---------------------------------------------------------------------------

def run_inference() -> Dict[str, Dict[str, Any]]:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    if HF_TOKEN.strip().endswith("xxx"):
        print("ERROR: HF_TOKEN is a placeholder")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results: Dict[str, Dict[str, Any]] = {}
    global_start = time.monotonic()

    for task_id in TASKS:

        # ── Global time budget ─────────────────────────────────────
        elapsed = time.monotonic() - global_start
        if elapsed >= GLOBAL_TIMEOUT_S:
            print(f"[BUDGET] Global timeout ({elapsed:.0f}s). Skipping rest.")
            for tid in TASKS[TASKS.index(task_id):]:
                if tid not in results:
                    results[tid] = {
                        "reward": 0.01, "steps": 0, "error": "Global timeout",
                        "step_rewards": [0.01], "success": False,
                    }
            break

        print(f"\n[START] task={task_id} model={MODEL_NAME}")
        task_start = time.monotonic()

        try:
            # ── Direct reset (no HTTP) ─────────────────────────────
            obs = reset_episode(task_id)
            obs_dict = obs.model_dump()

            steps = 0
            final_reward = 0.01
            step_rewards: list = []
            success = False
            force_test = False
            consec_inspect = 0
            conv_messages: list = []

            for step_num in range(MAX_STEPS):
                steps = step_num + 1

                # Per-task budget
                if (time.monotonic() - task_start) >= PER_TASK_TIMEOUT_S:
                    print(f"  [BUDGET] Task timeout. Moving on.")
                    break
                # Global budget (inner)
                if (time.monotonic() - global_start) >= GLOBAL_TIMEOUT_S:
                    print(f"  [BUDGET] Global timeout. Breaking.")
                    break

                # Build messages — keep context window small
                messages = [{"role": "system", "content": build_system_prompt()}]
                messages += conv_messages[-6:]  # last 3 turns only
                # Send compact observation (only what the LLM needs)
                compact_obs = {
                    "task_id": obs_dict.get("task_id", ""),
                    "buggy_code": obs_dict.get("buggy_code", ""),
                    "terminal_output": (obs_dict.get("terminal_output", "") or "")[:1500],
                    "test_results": obs_dict.get("test_results", {}),
                    "step": steps,
                    "done": obs_dict.get("done", False),
                }
                messages.append({"role": "user", "content": json.dumps(compact_obs)})

                # ── LLM call ───────────────────────────────────────
                text = _call_llm(client, messages)

                conv_messages += [
                    {"role": "user", "content": json.dumps(compact_obs)},
                    {"role": "assistant", "content": text},
                ]

                action_dict = parse_action(text)

                # Guardrails
                if force_test and action_dict["action_type"] in {"INSPECT_ERROR", "QUERY_CONTEXT"}:
                    action_dict = {"action_type": "EXECUTE_UNIT_TEST",
                                   "patched_code": None, "rationale": "Auto-test"}
                if action_dict["action_type"] == "INSPECT_ERROR":
                    consec_inspect += 1
                    if consec_inspect >= 2:
                        action_dict = {"action_type": "EXECUTE_UNIT_TEST",
                                       "patched_code": None, "rationale": "Auto-test"}
                        consec_inspect = 0
                else:
                    consec_inspect = 0

                if action_dict["action_type"] == "EDIT_CODE":
                    force_test = True
                elif action_dict["action_type"] in {"EXECUTE_UNIT_TEST", "SUBMIT_FIX"}:
                    force_test = False

                # ── Direct step (no HTTP) ──────────────────────────
                action = Action(**action_dict)
                result = step_episode(task_id, action)

                obs = result.observation
                obs_dict = obs.model_dump()
                final_reward = result.reward.value
                clamped_r = _clamp(final_reward)
                step_rewards.append(clamped_r)

                tr = obs.test_results
                passed = tr.get("passed", 0)
                total = tr.get("total", 0)
                done = result.done

                print(
                    f"  [STEP] s={steps} act={action_dict['action_type']} "
                    f"r={clamped_r:.2f} done={done}"
                )

                if done:
                    success = (passed == total and total > 0)
                    break

            results[task_id] = {
                "reward": _clamp(final_reward),
                "steps": steps,
                "step_rewards": step_rewards,
                "success": success,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            results[task_id] = {
                "reward": 0.01, "steps": 0, "error": str(e),
                "step_rewards": [0.01], "success": False,
            }

        # ── [END] log ──────────────────────────────────────────────
        r = results[task_id]
        t_elapsed = time.monotonic() - task_start
        rcsv = ",".join(f"{rw:.2f}" for rw in r.get("step_rewards", [r["reward"]]))
        print(f"[END] ok={r.get('success',False)} steps={r['steps']} "
              f"rewards={rcsv} time={t_elapsed:.1f}s")

    # ── Final summary ──────────────────────────────────────────────
    total_time = time.monotonic() - global_start
    print("\n" + "=" * 60)
    print(f"=== ZEROTRACE RESULTS  (total {total_time:.1f}s) ===")
    print("=" * 60)

    for tid, r in results.items():
        rw = r.get("reward", 0.01)
        s = r.get("steps", 0)
        e = r.get("error", "")
        sfx = f"  ERR: {e}" if e else ""
        print(f"{tid:<32} reward={rw:.2f}  steps={s}{sfx}")

    valid = [r["reward"] for r in results.values() if "error" not in r]
    mean = sum(valid) / len(valid) if valid else 0.01
    print(f"\nMean Reward: {mean:.4f}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    all_ok = True
    for tid, r in results.items():
        rw = r.get("reward", 0.01)
        ok = 0.0 < rw < 1.0
        if not ok:
            all_ok = False
        print(f"{'OK' if ok else 'FAIL'}: {tid} reward={rw:.2f}")

    if all_ok:
        print("\nPASSED: All rewards in (0, 1)")
    else:
        print("\nFAILED: Rewards out of range")
        sys.exit(1)

    return results


if __name__ == "__main__":
    run_inference()
