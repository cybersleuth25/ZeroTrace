#!/usr/bin/env python3
"""ZeroTrace Inference Script.

Runs the ZeroTrace agent on all tasks and prints results.
Must complete in under 20 minutes.

Environment variables:
  MODEL_NAME          : HuggingFace model to use (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN            : HuggingFace API token
  ZEROTRACE_BASE_URL  : Base URL of the running app.py (default: http://localhost:7860)
  RUN_TORCH_TASKS     : Set to '1' to include PyTorch tasks (default: '0')
"""

import os
import sys
import json
import time
import re
import requests
from typing import Dict, Any, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE = os.environ.get("ZEROTRACE_BASE_URL", "http://localhost:7860")
RUN_TORCH = os.environ.get("RUN_TORCH_TASKS", "0") == "1"

# Classic tasks always run; PyTorch tasks are opt-in
CLASSIC_TASKS = [
    "level1_keyerror",
    "level2_resource_leak",
    "level3_race_condition",
]
TORCH_TASKS = [
    "torch_dtype_mismatch",
    "torch_nan_gradient",
    "torch_wrong_dim",
]
TASKS: List[str] = CLASSIC_TASKS + (TORCH_TASKS if RUN_TORCH else [])

MAX_STEPS = 15
VALID_ACTIONS = {
    "INSPECT_ERROR", "EDIT_CODE", "RUN_COMPILER",
    "EXECUTE_UNIT_TEST", "QUERY_CONTEXT", "SUBMIT_FIX",
    "SEARCH_DOCS", "RUN_SNIPPET",
}


# ---------------------------------------------------------------------------
# Action parser (standalone copy — no dependency on app.py)
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
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return (
        "You are ZERO, an autonomous code repair agent. Fix Python and PyTorch bugs.\n\n"
        "## Response Format\n"
        "---DIAGNOSIS---\n[What is wrong]\n\n"
        "---CHAIN-OF-THOUGHT---\n[Your reasoning]\n\n"
        "---ACTION---\n"
        '{"action_type": "ACTION", "patched_code": "CODE", "rationale": "WHY"}\n\n'
        "---PATCHED_CODE---\n[Full fixed code if EDIT_CODE or SUBMIT_FIX]\n\n"
        "## Actions\n"
        "- INSPECT_ERROR: See error messages\n"
        "- EDIT_CODE: Modify code (include patched_code)\n"
        "- EXECUTE_UNIT_TEST: Run tests\n"
        "- SEARCH_DOCS: Search Python/PyTorch docs (set rationale=query)\n"
        "- SUBMIT_FIX: Submit final fix (include patched_code)\n\n"
        "## Strategy\n"
        "1. INSPECT_ERROR → 2. EDIT_CODE → 3. EXECUTE_UNIT_TEST → 4. SUBMIT_FIX\n"
        "Include the COMPLETE file in patched_code, not just a diff."
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> Dict[str, Dict[str, Any]]:
    print("=" * 60)
    print("ZEROTRACE INFERENCE (HuggingFace)")
    print("=" * 60)
    print(f"Model      : {MODEL_NAME}")
    print(f"Base URL   : {BASE}")
    print(f"Tasks      : {len(TASKS)} ({', '.join(TASKS)})")
    print("=" * 60)

    # Token validation
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Get one at huggingface.co/settings/tokens")
        sys.exit(1)
    if HF_TOKEN.strip().endswith("xxx"):
        print("ERROR: HF_TOKEN appears to be a placeholder. Set a real token.")
        sys.exit(1)

    # Server health check
    try:
        health = requests.get(f"{BASE}/health", timeout=10)
        health.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach ZeroTrace server at {BASE}")
        print("Start app.py first, then rerun inference.py")
        print(f"Details: {e}")
        sys.exit(1)

    client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)
    results: Dict[str, Dict[str, Any]] = {}

    for task_id in TASKS:
        print(f"\n{'=' * 40}")
        print(f"Task: {task_id}")
        print("=" * 40)

        try:
            # Reset — use versioned route, fall back to legacy
            reset_resp = requests.post(
                f"{BASE}/api/v1/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            reset_resp.raise_for_status()
            obs = reset_resp.json()

            steps = 0
            final_reward = 0.0
            force_test = False
            consec_inspect = 0
            conv_messages: list = []

            for step_num in range(MAX_STEPS):
                steps = step_num + 1
                print(f"\n--- Step {steps} ---")

                # Build messages (multi-turn)
                messages = [{"role": "system", "content": build_system_prompt()}]
                messages += conv_messages[-10:]   # last 5 turns
                messages.append({"role": "user", "content": json.dumps(obs, indent=2)})

                try:
                    response = client.chat_completion(messages=messages, max_tokens=1500)
                    text = response.choices[0].message.content or ""
                except HfHubHTTPError as e:
                    err = str(e)
                    if "402" in err or "payment required" in err.lower():
                        raise RuntimeError("402 Payment Required from HuggingFace.") from e
                    if "401" in err or "unauthorized" in err.lower():
                        raise RuntimeError("401 Unauthorized. Check HF_TOKEN.") from e
                    raise RuntimeError(f"HuggingFace API Error: {err[:180]}") from e
                except Exception as e:
                    print(f"LLM Error: {e}")
                    text = ""

                # Update conversation history
                conv_messages += [
                    {"role": "user", "content": json.dumps(obs, indent=2)},
                    {"role": "assistant", "content": text},
                ]

                action = parse_action(text)

                # Guardrails
                if force_test and action["action_type"] in {"INSPECT_ERROR", "QUERY_CONTEXT"}:
                    action = {"action_type": "EXECUTE_UNIT_TEST",
                               "patched_code": None, "rationale": "Auto-test after edit"}
                if action["action_type"] == "INSPECT_ERROR":
                    consec_inspect += 1
                    if consec_inspect >= 3:
                        action = {"action_type": "EXECUTE_UNIT_TEST",
                                   "patched_code": None, "rationale": "Auto-test: repeated inspects"}
                        consec_inspect = 0
                else:
                    consec_inspect = 0

                if action["action_type"] == "EDIT_CODE":
                    force_test = True
                elif action["action_type"] in {"EXECUTE_UNIT_TEST", "SUBMIT_FIX"}:
                    force_test = False

                print(f"Action: {action['action_type']}")

                step_resp = requests.post(
                    f"{BASE}/api/v1/step",
                    json={
                        "task_id": task_id,
                        "action": action,
                        "model_name": MODEL_NAME,   # for leaderboard auto-record
                    },
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()

                obs = step_data["observation"]
                final_reward = step_data["reward"]["value"]

                print(f"Reward: {final_reward:.2f}")
                tr = obs.get("test_results", {})
                print(f"Tests: {tr.get('passed', 0)}/{tr.get('total', 0)}")

                if step_data["done"]:
                    print("Episode complete!")
                    break

                time.sleep(0.5)

            results[task_id] = {"reward": final_reward, "steps": steps}

        except Exception as e:
            print(f"Error on {task_id}: {e}")
            results[task_id] = {"reward": 0.0, "steps": 0, "error": str(e)}

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("=== ZEROTRACE INFERENCE RESULTS ===")
    print("=" * 60)

    for tid, r in results.items():
        reward = r.get("reward", 0.0)
        steps = r.get("steps", 0)
        error = r.get("error", "")
        suffix = f"  ERROR: {error}" if error else ""
        print(f"{tid:<32} Reward: {reward:.2f}  Steps: {steps}{suffix}")

    valid_rewards = [r["reward"] for r in results.values() if "error" not in r]
    mean = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
    print(f"\nMean Reward: {mean:.4f}")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    all_valid = True
    for tid, r in results.items():
        rw = r.get("reward", 0.0)
        ok = -1.0 <= rw <= 1.0
        if not ok:
            all_valid = False
        print(f"{'OK' if ok else 'FAIL'}: {tid} reward {rw:.2f}")

    if all_valid:
        print("\nValidation PASSED: All rewards in [-1.0, 1.0]")
    else:
        print("\nValidation FAILED: Rewards out of range")
        sys.exit(1)

    return results


if __name__ == "__main__":
    run_inference()
