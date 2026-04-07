#!/usr/bin/env python3
"""ZeroTrace Inference Script.

Runs the ZeroTrace agent on all tasks and prints structured results.
Must complete in under 20 minutes on vcpu=2, memory=8gb.

Required environment variables:
  API_BASE_URL  : The API endpoint for the LLM (e.g. https://api-inference.huggingface.co/v1)
  MODEL_NAME    : The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      : Your HuggingFace API key

Optional:
  ZEROTRACE_BASE_URL : Base URL of the running app.py (default: http://localhost:7860)
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _clamp(v: float) -> float:
    """Clamp a score to the open interval (0, 1)."""
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    return round(v, 4)

# ---------------------------------------------------------------------------
# Configuration (uses the MANDATORY hackathon env vars)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BASE = os.getenv("ZEROTRACE_BASE_URL", "http://localhost:7860")

# Tasks to run
TASKS: List[str] = [
    "level1_keyerror",
    "level2_resource_leak",
    "level3_race_condition",
]

MAX_STEPS = 15
VALID_ACTIONS = {
    "INSPECT_ERROR", "EDIT_CODE", "RUN_COMPILER",
    "EXECUTE_UNIT_TEST", "QUERY_CONTEXT", "SUBMIT_FIX",
    "SEARCH_DOCS", "RUN_SNIPPET",
}


# ---------------------------------------------------------------------------
# Action parser (standalone — no dependency on app.py)
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
        "1. INSPECT_ERROR -> 2. EDIT_CODE -> 3. EXECUTE_UNIT_TEST -> 4. SUBMIT_FIX\n"
        "Include the COMPLETE file in patched_code, not just a diff."
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> Dict[str, Dict[str, Any]]:
    # ── Validate env vars ──────────────────────────────────────────
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Get one at huggingface.co/settings/tokens")
        sys.exit(1)
    if HF_TOKEN.strip().endswith("xxx"):
        print("ERROR: HF_TOKEN appears to be a placeholder. Set a real token.")
        sys.exit(1)

    # ── Server health check ────────────────────────────────────────
    try:
        health = requests.get(f"{BASE}/health", timeout=10)
        health.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach ZeroTrace server at {BASE}")
        print("Start app.py first, then rerun inference.py")
        print(f"Details: {e}")
        sys.exit(1)

    # ── OpenAI Client (MANDATORY per hackathon rules) ──────────────
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    results: Dict[str, Dict[str, Any]] = {}

    for task_id in TASKS:

        # ── [START] log ────────────────────────────────────────────
        print(f"[START] task={task_id} env=zerotrace model={MODEL_NAME}")

        try:
            # Reset episode
            reset_resp = requests.post(
                f"{BASE}/api/v1/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            reset_resp.raise_for_status()
            obs = reset_resp.json()

            steps = 0
            final_reward = 0.01
            step_rewards: list = []
            success = False
            force_test = False
            consec_inspect = 0
            conv_messages: list = []

            for step_num in range(MAX_STEPS):
                steps = step_num + 1

                # Build messages (multi-turn)
                messages = [{"role": "system", "content": build_system_prompt()}]
                messages += conv_messages[-10:]   # last 5 turns
                messages.append({"role": "user", "content": json.dumps(obs, indent=2)})

                # ── LLM call via OpenAI Client ─────────────────────
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=1500,
                    )
                    text = response.choices[0].message.content or ""
                except Exception as e:
                    err = str(e)
                    if "402" in err or "payment required" in err.lower():
                        raise RuntimeError("402 Payment Required. Add billing or use a smaller model.") from e
                    if "401" in err or "unauthorized" in err.lower():
                        raise RuntimeError("401 Unauthorized. Check HF_TOKEN.") from e
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

                # Step the environment
                step_resp = requests.post(
                    f"{BASE}/api/v1/step",
                    json={
                        "task_id": task_id,
                        "action": action,
                        "model_name": MODEL_NAME,
                    },
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()

                obs = step_data["observation"]
                final_reward = step_data["reward"]["value"]
                clamped_step_r = _clamp(final_reward)
                step_rewards.append(clamped_step_r)
                tr = obs.get("test_results", {})
                passed = tr.get("passed", 0)
                total = tr.get("total", 0)
                done = step_data["done"]
                step_error = step_data.get("info", {}).get("error")
                error_str = str(step_error).replace("\n", " ") if step_error else "null"

                # ── [STEP] log ─────────────────────────────────────
                print(
                    f"[STEP] step={steps} action={action['action_type']} "
                    f"reward={clamped_step_r:.2f} done={str(done).lower()} "
                    f"error={error_str}"
                )

                if done:
                    success = (passed == total and total > 0)
                    break

                time.sleep(0.5)

            results[task_id] = {
                "reward": _clamp(final_reward),
                "steps": steps,
                "step_rewards": step_rewards,
                "success": success,
            }

        except Exception as e:
            print(f"ERROR on {task_id}: {e}")
            results[task_id] = {
                "reward": 0.01, "steps": 0, "error": str(e),
                "step_rewards": [0.01], "success": False,
            }

        # ── [END] log ──────────────────────────────────────────────
        r = results[task_id]
        rewards_csv = ",".join(
            f"{rw:.2f}" for rw in r.get("step_rewards", [r["reward"]])
        )
        print(
            f"[END] success={str(r.get('success', False)).lower()} "
            f"steps={r.get('steps', 0)} rewards={rewards_csv}"
        )

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== ZEROTRACE INFERENCE RESULTS ===")
    print("=" * 60)

    for tid, r in results.items():
        reward = r.get("reward", 0.01)
        steps = r.get("steps", 0)
        error = r.get("error", "")
        suffix = f"  ERROR: {error}" if error else ""
        print(f"{tid:<32} Reward: {reward:.2f}  Steps: {steps}{suffix}")

    valid_rewards = [r["reward"] for r in results.values() if "error" not in r]
    mean = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.01
    print(f"\nMean Reward: {mean:.4f}")

    # Validation — scores must be strictly inside (0, 1)
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    all_valid = True
    for tid, r in results.items():
        rw = r.get("reward", 0.01)
        ok = 0.0 < rw < 1.0
        if not ok:
            all_valid = False
        print(f"{'OK' if ok else 'FAIL'}: {tid} reward {rw:.2f}")

    if all_valid:
        print("\nValidation PASSED: All rewards strictly in (0, 1)")
    else:
        print("\nValidation FAILED: Rewards out of range (0, 1)")
        sys.exit(1)

    return results


if __name__ == "__main__":
    run_inference()
