"""ZeroTrace agent implementation.

parse_action() is used by both app.py (UI) and inference.py (CLI).
run_agent_turn() is a convenience wrapper for OpenAI-compatible clients.
"""

import json
import re
from typing import Any, Dict, Optional

from environment.models import Observation, Action
from .prompt_builder import build_prompt

_VALID_ACTIONS = {
    "INSPECT_ERROR", "EDIT_CODE", "RUN_COMPILER",
    "EXECUTE_UNIT_TEST", "QUERY_CONTEXT", "SUBMIT_FIX",
    "SEARCH_DOCS", "RUN_SNIPPET",
}


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Dict[str, Any]:
    """Parse an action dict from LLM response text.

    Tries multiple extraction strategies in priority order:
      1. JSON in ---ACTION--- block
      2. Regex key extraction from ---ACTION--- block
      3. ---PATCHED_CODE--- block + keyword mention
      4. Keyword scan of full text

    Returns a safe default (INSPECT_ERROR) if all strategies fail.
    """
    default_action: Dict[str, Any] = {
        "action_type": "INSPECT_ERROR",
        "patched_code": None,
        "rationale": "Could not parse action from response",
    }

    if not text:
        return default_action

    # ── 1. Extract ---ACTION--- block ────────────────────────────────────────
    action_block = ""
    action_match = re.search(
        r"---ACTION---\s*\n?(.*?)(?:---|$)", text, re.DOTALL | re.IGNORECASE
    )
    if action_match:
        action_block = action_match.group(1).strip()

    parsed_type: Optional[str] = None
    parsed_code: Optional[str] = None
    parsed_rationale: str = ""

    if action_block:
        # Strategy 1: full JSON parse
        try:
            aj = json.loads(action_block)
            if isinstance(aj, dict) and aj.get("action_type") in _VALID_ACTIONS:
                parsed_type = aj["action_type"]
                parsed_code = aj.get("patched_code")
                parsed_rationale = aj.get("rationale", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Strategy 2: regex key extraction
        if not parsed_type:
            m = re.search(r'"action_type"\s*:\s*"([A-Z_]+)"', action_block)
            if m and m.group(1) in _VALID_ACTIONS:
                parsed_type = m.group(1)
            r = re.search(r'"rationale"\s*:\s*"(.*?)"', action_block, re.DOTALL)
            if r:
                parsed_rationale = r.group(1).strip()

    # ── 2. Extract ---PATCHED_CODE--- block ──────────────────────────────────
    # Try fenced code block first, then raw block
    code_match = re.search(
        r"---PATCHED_CODE---\s*\n?```(?:python)?\s*\n?(.*?)```",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not code_match:
        code_match = re.search(
            r"---PATCHED_CODE---\s*\n?(.*?)(?:---|$)",
            text, re.DOTALL | re.IGNORECASE,
        )
    patched_code_block = code_match.group(1).strip() if code_match else None

    if parsed_type:
        return {
            "action_type": parsed_type,
            "patched_code": parsed_code or patched_code_block,
            "rationale": parsed_rationale,
        }

    # ── 3. Keyword scan of full text ─────────────────────────────────────────
    tl = text.lower()
    kw_map = [
        ("submit_fix",        "SUBMIT_FIX"),
        ("submit fix",        "SUBMIT_FIX"),
        ("edit_code",         "EDIT_CODE"),
        ("edit code",         "EDIT_CODE"),
        ("execute_unit_test", "EXECUTE_UNIT_TEST"),
        ("run test",          "EXECUTE_UNIT_TEST"),
        ("run_compiler",      "RUN_COMPILER"),
        ("check syntax",      "RUN_COMPILER"),
        ("search_docs",       "SEARCH_DOCS"),
        ("search docs",       "SEARCH_DOCS"),
        ("run_snippet",       "RUN_SNIPPET"),
        ("query_context",     "QUERY_CONTEXT"),
    ]
    for keyword, action_type in kw_map:
        if keyword in tl:
            return {
                "action_type": action_type,
                "patched_code": patched_code_block,
                "rationale": f"Inferred from keyword '{keyword}'",
            }

    return default_action


# ---------------------------------------------------------------------------
# Convenience runner (used in tests / CLI scenarios)
# ---------------------------------------------------------------------------

def run_agent_turn(
    state: Observation,
    client: Any,
    model: str,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Run one agent turn using an OpenAI-compatible client.

    Args:
        state:      Current observation.
        client:     OpenAI / HuggingFace InferenceClient.
        model:      Model identifier.
        max_tokens: Token limit for the response.

    Returns:
        Dict with action, raw_response, diagnosis, chain_of_thought.
    """
    system_prompt, user_prompt = build_prompt(state)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_response = response.choices[0].message.content or ""
    except Exception as e:
        raw_response = f"LLM error: {e}"

    action_dict = parse_action(raw_response)
    action = Action(
        action_type=action_dict["action_type"],
        patched_code=action_dict.get("patched_code"),
        rationale=action_dict.get("rationale"),
    )

    diagnosis = ""
    dm = re.search(r"---DIAGNOSIS---\s*\n?(.*?)(?:---|$)", raw_response,
                   re.DOTALL | re.IGNORECASE)
    if dm:
        diagnosis = dm.group(1).strip()

    cot = ""
    cm = re.search(r"---CHAIN-OF-THOUGHT---\s*\n?(.*?)(?:---|$)", raw_response,
                   re.DOTALL | re.IGNORECASE)
    if cm:
        cot = cm.group(1).strip()

    return {
        "action": action,
        "raw_response": raw_response,
        "diagnosis": diagnosis,
        "chain_of_thought": cot,
    }
