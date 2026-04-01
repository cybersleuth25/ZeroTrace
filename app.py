#!/usr/bin/env python3
"""ZeroTrace v2 — Main Application

Aesthetic overhaul: hand-crafted dark UI with SVG icons,
Space Grotesk + JetBrains Mono, glass panels, and a
design language that doesn't scream template.
"""

import difflib
import json
import os
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

from environment.env import api_app
from environment.models import Observation, Action
from environment.state_machine import (
    reset_episode, step_episode, get_episode_replay, get_original_code,
)
from agent.zerotrace_agent import parse_action
from agent.prompt_builder import build_prompt, build_messages
from leaderboard.store import get_leaderboard, save_result
from leaderboard.replay import list_replays, load_replay
from reports.exporter import generate_report, save_report
from tasks import TASKS

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
_DEFAULT_TOKEN = os.environ.get("HF_TOKEN", "")
_MAX_STEPS = 15


# ---------------------------------------------------------------------------
# SVG icon library (inline, no external deps)
# ---------------------------------------------------------------------------

class _Icons:
    """Minimal inline SVG icons. 16x16 viewBox, stroke-based."""

    _S = 'xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'

    HEXAGON   = f'<svg {_S}><polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2"/></svg>'
    PLAY      = f'<svg {_S}><polygon points="5 3 19 12 5 21 5 3"/></svg>'
    ZAP       = f'<svg {_S}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
    ROTATE    = f'<svg {_S}><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>'
    TERMINAL  = f'<svg {_S}><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>'
    CODE      = f'<svg {_S}><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>'
    AWARD     = f'<svg {_S}><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>'
    FILM      = f'<svg {_S}><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/><line x1="17" y1="17" x2="22" y2="17"/></svg>'
    COLUMNS   = f'<svg {_S}><path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18"/></svg>'
    FILE_TEXT = f'<svg {_S}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>'
    FOLDER    = f'<svg {_S}><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>'
    DOWNLOAD  = f'<svg {_S}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
    SEARCH    = f'<svg {_S}><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
    CHECK     = f'<svg {_S} stroke="#22c55e"><polyline points="20 6 9 17 4 12"/></svg>'
    X_MARK    = f'<svg {_S} stroke="#ef4444"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>'
    MINUS     = f'<svg {_S} stroke="#f59e0b"><line x1="5" y1="12" x2="19" y2="12"/></svg>'

ICO = _Icons()


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_ACCENT         = "#00d4ff"
_ACCENT_DIM     = "#006680"
_BG_BASE        = "#0a0e14"
_BG_RAISED      = "#111720"
_BG_CARD        = "rgba(17, 23, 32, 0.7)"
_BORDER         = "#1c2333"
_BORDER_ACCENT  = "#00394d"
_TEXT            = "#c5cdd8"
_TEXT_DIM        = "#5c6a7a"
_SUCCESS         = "#22c55e"
_WARN            = "#f59e0b"
_DANGER          = "#ef4444"

_FONT_SANS      = "'Space Grotesk', system-ui, -apple-system, sans-serif"
_FONT_MONO      = "'JetBrains Mono', 'Fira Code', monospace"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_client(model_name: str, api_key: str) -> InferenceClient:
    return InferenceClient(model=model_name, token=api_key)


def _hf_error_msg(e: HfHubHTTPError, model_name: str) -> str:
    msg = str(e)
    if "401" in msg or "unauthorized" in msg.lower():
        return "Invalid HuggingFace token. Check huggingface.co/settings/tokens"
    if "402" in msg or "payment required" in msg.lower():
        return "402: Payment required. Add billing credits or switch to a smaller model."
    if "404" in msg or "not found" in msg.lower():
        return f"Model '{model_name}' not found. Try 'Qwen/Qwen2.5-72B-Instruct'"
    if "rate" in msg.lower():
        return "Rate limited. Wait a moment and retry."
    return f"API error: {msg[:200]}"


def _reward_html(reward: float) -> str:
    """Render a styled reward badge with inline SVG indicator."""
    if reward >= 0.8:
        color, icon, label = _SUCCESS, ICO.CHECK, "PASS"
    elif reward >= 0.5:
        color, icon, label = _WARN, ICO.MINUS, "PARTIAL"
    else:
        color, icon, label = _DANGER, ICO.X_MARK, "FAIL"
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:6px 14px;border-radius:6px;'
        f'background:{color}15;border:1px solid {color}40;'
        f'font-family:{_FONT_MONO};font-size:0.95rem;color:{color}">'
        f'{icon}'
        f'<span style="font-weight:600;letter-spacing:0.04em">{reward:.2f}</span>'
        f'<span style="opacity:0.7;font-size:0.8rem;text-transform:uppercase;'
        f'letter-spacing:0.08em">{label}</span>'
        f'</div>'
    )


def _make_diff(original: str, patched: str) -> str:
    orig_lines = original.splitlines(keepends=True)
    patch_lines = patched.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, patch_lines,
        fromfile="original (buggy)",
        tofile="current (patched)",
        lineterm="",
    ))
    return "".join(diff) if diff else "No changes yet."


def _status_msg(text: str, variant: str = "info") -> str:
    """Plain status text — no emojis."""
    prefix_map = {"ok": "[OK]", "err": "[ERR]", "info": "[---]"}
    return f"{prefix_map.get(variant, '')} {text}"


# ---------------------------------------------------------------------------
# Tab 1: Benchmark
# ---------------------------------------------------------------------------

def reset_task(task_id: str) -> tuple:
    if not task_id or task_id not in TASKS:
        return "", "", "", "", "Select a valid task", "", _status_msg("Select a valid task")
    try:
        obs = reset_episode(task_id)
        obs_dict = obs.model_dump()
        diff = _make_diff(TASKS[task_id]["buggy_code"], obs.buggy_code)
        return (
            obs.buggy_code,
            obs.terminal_output,
            json.dumps(obs_dict, indent=2),
            "",
            diff,
            _reward_html(obs.reward),
            _status_msg(f"Task reset: {task_id}", "ok"),
        )
    except Exception as e:
        return "", "", "", "", "", "", _status_msg(f"Error: {e}", "err")


def run_agent_step(
    task_id: str,
    model_name: str,
    api_key: str,
    current_obs_json: str,
    history_text: str,
    conv_history: List[Dict],
) -> tuple:
    empty = ("", "", current_obs_json, "", "", history_text, "", conv_history, "")

    if not task_id:
        return *empty[:-1], _status_msg("Select a task first")
    if not model_name:
        return *empty[:-1], _status_msg("Enter a model name")
    if not api_key:
        return *empty[:-1], _status_msg("Enter a HuggingFace token")

    try:
        obs_dict = json.loads(current_obs_json) if current_obs_json else None
        if not obs_dict:
            obs = reset_episode(task_id)
            obs_dict = obs.model_dump()
        else:
            obs = Observation(**obs_dict)

        if obs.done:
            return (
                obs.buggy_code, obs.terminal_output,
                json.dumps(obs_dict, indent=2),
                "", _reward_html(obs.reward), history_text,
                _make_diff(TASKS[task_id]["buggy_code"], obs.buggy_code),
                conv_history,
                _status_msg(f"Episode complete. Final: {obs.reward:.2f}", "ok"),
            )

        messages = build_messages(obs, conv_history)

        try:
            client = _make_client(model_name, api_key)
            response = client.chat_completion(messages=messages, max_tokens=1500)
            llm_response = response.choices[0].message.content or ""
        except HfHubHTTPError as e:
            return *empty[:-1], _status_msg(_hf_error_msg(e, model_name), "err")
        except Exception as e:
            return *empty[:-1], _status_msg(f"{type(e).__name__}: {e}", "err")

        new_conv = conv_history + [
            {"role": "user", "content": messages[-1]["content"]},
            {"role": "assistant", "content": llm_response},
        ]

        action_dict = parse_action(llm_response)
        action = Action(**action_dict)
        result = step_episode(task_id, action)
        new_obs = result.observation

        step_info = (
            f"\n[step {new_obs.step_count}]  "
            f"action={action.action_type}  "
            f"reward={result.reward.value:.2f}  "
            f"({result.reward.reason})\n"
        )
        new_history = history_text + step_info

        is_done = (
            result.done
            and new_obs.test_results.get("passed") == new_obs.test_results.get("total")
        )
        status = (
            _status_msg(f"All tests passed in {new_obs.step_count} steps", "ok")
            if is_done
            else _status_msg(f"Step {new_obs.step_count}: {action.action_type}")
        )

        diff = _make_diff(TASKS[task_id]["buggy_code"], new_obs.buggy_code)

        return (
            new_obs.buggy_code,
            new_obs.terminal_output,
            json.dumps(new_obs.model_dump(), indent=2),
            f"--- agent response ---\n{llm_response[:1200]}...",
            _reward_html(new_obs.reward),
            new_history,
            diff,
            new_conv,
            status,
        )

    except Exception as e:
        return *empty[:-1], _status_msg(f"Error: {e}", "err")


def run_full_episode(
    task_id: str,
    model_name: str,
    api_key: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    empty = ("", "", "", "", "", "", "", [], "")

    if not task_id:
        return *empty[:-1], _status_msg("Select a task first")
    if not model_name:
        return *empty[:-1], _status_msg("Enter a model name")
    if not api_key:
        return *empty[:-1], _status_msg("Enter a HuggingFace token")

    try:
        obs = reset_episode(task_id)
        history_text = f"=== {task_id} ===\n"
        conv_history: List[Dict] = []

        try:
            client = _make_client(model_name, api_key)
        except Exception as e:
            return *empty[:-1], _status_msg(f"Client error: {e}", "err")

        force_test_after_edit = False
        consecutive_inspect = 0
        progress(0, desc="Initialising...")

        for step in range(_MAX_STEPS):
            if obs.done:
                break

            progress(
                (step + 1) / _MAX_STEPS,
                desc=f"Step {step + 1}/{_MAX_STEPS}",
            )

            messages = build_messages(obs, conv_history)

            try:
                response = client.chat_completion(messages=messages, max_tokens=1500)
                llm_response = response.choices[0].message.content or ""
            except HfHubHTTPError as e:
                return *empty[:-1], _status_msg(_hf_error_msg(e, model_name), "err")
            except Exception as e:
                return *empty[:-1], _status_msg(f"LLM error: {e}", "err")

            conv_history = conv_history + [
                {"role": "user", "content": messages[-1]["content"]},
                {"role": "assistant", "content": llm_response},
            ]

            action_dict = parse_action(llm_response)
            action = Action(**action_dict)

            # Guardrails
            if force_test_after_edit and action.action_type in {"INSPECT_ERROR", "QUERY_CONTEXT"}:
                action = Action(
                    action_type="EXECUTE_UNIT_TEST", patched_code=None,
                    rationale="Auto-test after edit",
                )
            if action.action_type == "INSPECT_ERROR":
                consecutive_inspect += 1
                if consecutive_inspect >= 3:
                    action = Action(
                        action_type="EXECUTE_UNIT_TEST", patched_code=None,
                        rationale="Auto-test after repeated inspections",
                    )
                    consecutive_inspect = 0
            else:
                consecutive_inspect = 0

            if action.action_type == "EDIT_CODE":
                force_test_after_edit = True
            elif action.action_type in {"EXECUTE_UNIT_TEST", "SUBMIT_FIX"}:
                force_test_after_edit = False

            result = step_episode(task_id, action)
            obs = result.observation

            history_text += (
                f"\n[step {obs.step_count}]  {action.action_type}"
                f"  reward={result.reward.value:.2f}"
            )

        # record to leaderboard
        try:
            save_result(
                model=model_name,
                task_id=task_id,
                reward=obs.reward,
                steps=obs.step_count,
            )
        except Exception:
            pass

        passed = obs.test_results.get("passed", 0)
        total = obs.test_results.get("total", 0)
        status = (
            _status_msg(f"Score {obs.reward:.2f} in {obs.step_count} steps. All tests green.", "ok")
            if passed == total
            else _status_msg(f"Done. Score {obs.reward:.2f}, {passed}/{total} passing.")
        )

        diff = _make_diff(TASKS[task_id]["buggy_code"], obs.buggy_code)

        return (
            obs.buggy_code,
            obs.terminal_output,
            json.dumps(obs.model_dump(), indent=2),
            "",
            _reward_html(obs.reward),
            history_text,
            diff,
            conv_history,
            status,
        )

    except Exception as e:
        return *empty[:-1], _status_msg(f"Error: {e}", "err")


# ---------------------------------------------------------------------------
# Tab 2: Leaderboard
# ---------------------------------------------------------------------------

def refresh_leaderboard() -> List[List]:
    data = get_leaderboard()
    rows = []
    for m in data.get("models", []):
        rows.append([
            m["rank"],
            m["model"],
            f"{m['mean_reward']:.4f}",
            m["tasks_completed"],
            m["avg_steps"],
            m["last_run"][:19].replace("T", " "),
        ])
    return rows if rows else [["--", "No runs recorded", "--", "--", "--", "--"]]


# ---------------------------------------------------------------------------
# Tab 3: Replay
# ---------------------------------------------------------------------------

def get_replay_files() -> List[str]:
    return list_replays() or ["(none)"]


def load_replay_file(filename: str) -> Tuple[List[Dict], int, str]:
    if not filename or filename == "(none)":
        return [], 0, "Select a replay file"
    log = load_replay(filename)
    if not log:
        return [], 0, "Could not load replay"
    return log, len(log) - 1, f"Loaded {len(log)} steps from {filename}"


def get_replay_step(
    replay_log_json: str,
    step_idx: int,
) -> Tuple[str, str, str]:
    if not replay_log_json:
        return "", "", ""
    try:
        log = json.loads(replay_log_json)
        if not log or step_idx >= len(log):
            return "", "", ""
        entry = log[step_idx]
        rv = entry.get("reward", {}).get("value", 0.0)
        rr = entry.get("reward", {}).get("reason", "")
        at = entry.get("action", {}).get("action_type", "?")
        return (
            entry.get("code", ""),
            entry.get("terminal_output", ""),
            f"step {entry.get('step', step_idx + 1)}  |  "
            f"action={at}  |  reward={rv:.2f} ({rr})",
        )
    except Exception:
        return "", "", "Error loading step"


# ---------------------------------------------------------------------------
# Tab 4: Compare
# ---------------------------------------------------------------------------

def _run_episode_for_compare(
    task_id: str,
    model_name: str,
    api_key: str,
) -> Dict[str, Any]:
    try:
        obs = reset_episode(task_id)
        conv_history: List[Dict] = []
        client = _make_client(model_name, api_key)
        force_test = False
        consec_inspect = 0

        for _ in range(_MAX_STEPS):
            if obs.done:
                break
            messages = build_messages(obs, conv_history)
            try:
                resp = client.chat_completion(messages=messages, max_tokens=1500)
                llm_text = resp.choices[0].message.content or ""
            except Exception as e:
                return {"error": str(e), "model": model_name}

            conv_history += [
                {"role": "user", "content": messages[-1]["content"]},
                {"role": "assistant", "content": llm_text},
            ]
            action_dict = parse_action(llm_text)
            action = Action(**action_dict)

            if force_test and action.action_type in {"INSPECT_ERROR", "QUERY_CONTEXT"}:
                action = Action(action_type="EXECUTE_UNIT_TEST", patched_code=None, rationale="")
            if action.action_type == "INSPECT_ERROR":
                consec_inspect += 1
                if consec_inspect >= 3:
                    action = Action(action_type="EXECUTE_UNIT_TEST", patched_code=None, rationale="")
                    consec_inspect = 0
            else:
                consec_inspect = 0
            if action.action_type == "EDIT_CODE":
                force_test = True
            elif action.action_type in {"EXECUTE_UNIT_TEST", "SUBMIT_FIX"}:
                force_test = False

            result = step_episode(task_id, action)
            obs = result.observation

        return {
            "model": model_name,
            "reward": obs.reward,
            "steps": obs.step_count,
            "passed": obs.test_results.get("passed", 0),
            "total": obs.test_results.get("total", 0),
            "final_code": obs.buggy_code,
        }
    except Exception as e:
        return {"error": str(e), "model": model_name}


def run_comparison(
    task_id: str,
    model1: str, token1: str,
    model2: str, token2: str,
) -> Tuple[str, str, str]:
    if not task_id:
        return "", "", "Select a task"
    if not model1 or not token1:
        return "", "", "Model A: name and token required"
    if not model2 or not token2:
        return "", "", "Model B: name and token required"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut1 = ex.submit(_run_episode_for_compare, task_id, model1, token1)
        fut2 = ex.submit(_run_episode_for_compare, task_id, model2, token2)
        r1, r2 = fut1.result(), fut2.result()

    def _fmt(r: Dict) -> str:
        if "error" in r:
            return f"**Error:** {r['error']}"
        p, t = r["passed"], r["total"]
        verdict = "ALL PASS" if p == t else f"{p}/{t} passing"
        return (
            f"**{r['model']}**\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Reward | `{r['reward']:.4f}` |\n"
            f"| Steps | {r['steps']} |\n"
            f"| Tests | {verdict} |\n\n"
            f"```python\n{r['final_code'][:500]}\n```"
        )

    winner = ""
    if "error" not in r1 and "error" not in r2:
        if r1["reward"] > r2["reward"]:
            winner = f"Winner: {model1}"
        elif r2["reward"] > r1["reward"]:
            winner = f"Winner: {model2}"
        else:
            winner = "Tie"

    return _fmt(r1), _fmt(r2), f"Comparison complete. {winner}"


# ---------------------------------------------------------------------------
# Tab 5: Report
# ---------------------------------------------------------------------------

_last_run_results: Dict[str, Any] = {}


def generate_and_preview_report(model_name: str) -> Tuple[str, Optional[str]]:
    if not _last_run_results:
        return "No run data. Complete a full episode first.", None
    try:
        content = generate_report(_last_run_results, model_name or "unknown")
        path = save_report(_last_run_results, model_name or "unknown")
        return content, str(path)
    except Exception as e:
        return f"Error generating report: {e}", None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root ───────────────────────────────────────────────────────── */
.gradio-container {{
    background: {_BG_BASE} !important;
    font-family: {_FONT_SANS} !important;
    color: {_TEXT} !important;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 24px !important;
    margin: 0 !important;
}}

/* ── Panels / Cards ─────────────────────────────────────────────── */
.gr-panel, .gr-box, .gr-form {{
    background: {_BG_CARD} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    backdrop-filter: blur(12px) !important;
}}

/* ── Labels ─────────────────────────────────────────────────────── */
label, .gr-input-label {{
    color: {_TEXT_DIM} !important;
    font-family: {_FONT_SANS} !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}}

/* ── Inputs ─────────────────────────────────────────────────────── */
input, textarea, select, .gr-text-input, .gr-textarea {{
    background: {_BG_RAISED} !important;
    border: 1px solid {_BORDER} !important;
    color: {_TEXT} !important;
    font-family: {_FONT_MONO} !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
    transition: border-color 0.2s ease !important;
}}
input:focus, textarea:focus {{
    border-color: {_ACCENT_DIM} !important;
    outline: none !important;
    box-shadow: 0 0 0 2px {_ACCENT}20 !important;
}}

/* ── Buttons ────────────────────────────────────────────────────── */
.gr-button {{
    font-family: {_FONT_SANS} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}
button.primary, .gr-button-primary {{
    background: {_ACCENT} !important;
    color: {_BG_BASE} !important;
    border: none !important;
}}
button.primary:hover, .gr-button-primary:hover {{
    background: #00bbdd !important;
    box-shadow: 0 0 16px {_ACCENT}30 !important;
}}
button.secondary, .gr-button-secondary {{
    background: transparent !important;
    border: 1px solid {_BORDER} !important;
    color: {_TEXT_DIM} !important;
}}
button.secondary:hover, .gr-button-secondary:hover {{
    border-color: {_ACCENT_DIM} !important;
    color: {_ACCENT} !important;
}}

/* ── Tabs ──────────────────────────────────────────────────────── */
.tabs > .tab-nav > button {{
    font-family: {_FONT_SANS} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    color: {_TEXT_DIM} !important;
    padding: 10px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}}
.tabs > .tab-nav > button.selected {{
    color: {_ACCENT} !important;
    border-bottom: 2px solid {_ACCENT} !important;
}}

/* ── Code blocks ───────────────────────────────────────────────── */
.cm-editor, .cm-content, .code-component {{
    font-family: {_FONT_MONO} !important;
    font-size: 0.84rem !important;
    background: {_BG_RAISED} !important;
}}

/* ── Slider ────────────────────────────────────────────────────── */
input[type="range"] {{
    accent-color: {_ACCENT} !important;
}}

/* ── Dataframe ─────────────────────────────────────────────────── */
.dataframe th {{
    background: {_BG_RAISED} !important;
    color: {_ACCENT} !important;
    font-family: {_FONT_SANS} !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid {_BORDER} !important;
}}
.dataframe td {{
    font-family: {_FONT_MONO} !important;
    font-size: 0.84rem !important;
    color: {_TEXT} !important;
    border-bottom: 1px solid {_BORDER}80 !important;
}}

/* ── Footer hide ───────────────────────────────────────────────── */
footer {{ display: none !important; }}

/* ── Accordion ─────────────────────────────────────────────────── */
.gr-accordion {{
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {_BG_CARD} !important;
}}
"""

# ---------------------------------------------------------------------------
# Header HTML
# ---------------------------------------------------------------------------

HEADER_HTML = f"""
<div style="
    text-align: center;
    padding: 32px 20px 24px;
    border-bottom: 1px solid {_BORDER};
    margin-bottom: 4px;
    background: linear-gradient(180deg, {_BG_RAISED} 0%, {_BG_BASE} 100%);
">
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    ">
        <span style="color:{_ACCENT};opacity:0.8">{ICO.HEXAGON}</span>
        <h1 style="
            color: {_TEXT};
            font-family: {_FONT_SANS};
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: 0.08em;
        ">
            ZERO<span style="color:{_ACCENT}">TRACE</span>
        </h1>
        <span style="color:{_ACCENT};opacity:0.8">{ICO.HEXAGON}</span>
    </div>
    <p style="
        color: {_TEXT_DIM};
        margin: 0;
        font-family: {_FONT_SANS};
        font-size: 0.9rem;
        font-weight: 400;
        letter-spacing: 0.04em;
    ">
        Autonomous Code Repair Benchmark
    </p>
    <p style="
        color: {_TEXT_DIM};
        margin: 6px 0 0;
        font-family: {_FONT_MONO};
        font-size: 0.72rem;
        opacity: 0.6;
        letter-spacing: 0.1em;
    ">
        v2.0 &middot; PyTorch Hackathon
    </p>
</div>
"""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ZeroTrace") as demo:

    gr.HTML(f"<style>{CSS}</style>")
    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Benchmark ───────────────────────────────────────────────
        with gr.Tab("Benchmark"):

            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=list(TASKS.keys()),
                    label="Task",
                    value="level1_keyerror",
                    scale=3,
                )
                reset_btn = gr.Button("Reset", variant="secondary", scale=1)

            with gr.Row():
                model_name_in = gr.Textbox(
                    label="Model",
                    value=_DEFAULT_MODEL,
                    info="HuggingFace model identifier",
                    scale=2,
                )
                api_key_in = gr.Textbox(
                    label="Token",
                    value=_DEFAULT_TOKEN,
                    type="password",
                    info="HuggingFace API token",
                    scale=2,
                )

            with gr.Row():
                step_btn = gr.Button("Step", variant="primary", scale=1)
                run_btn = gr.Button("Run Full Episode", variant="primary", scale=1)

            status_display = gr.Textbox(
                label="Status", interactive=False, max_lines=2,
            )
            reward_display = gr.HTML(
                value=f'<span style="color:{_TEXT_DIM};font-family:{_FONT_MONO};font-size:0.85rem">reward: --</span>',
            )

            with gr.Row():
                with gr.Column(scale=1):
                    code_display = gr.Code(
                        label="Source", language="python",
                        interactive=False, lines=22,
                    )
                with gr.Column(scale=1):
                    reasoning_display = gr.Textbox(
                        label="Agent Output", lines=22, interactive=False,
                    )
                with gr.Column(scale=1):
                    terminal_display = gr.Textbox(
                        label="Terminal", lines=22, interactive=False,
                    )

            with gr.Accordion("Diff View", open=False):
                diff_display = gr.Textbox(
                    label="Unified Diff (original vs current)",
                    interactive=False, lines=15,
                )

            history_display = gr.Textbox(
                label="Step Log", lines=4, interactive=False,
            )

            obs_state = gr.State(None)
            conv_history_state = gr.State([])

            _bench_outputs = [
                code_display, terminal_display, obs_state, reasoning_display,
                reward_display, history_display, diff_display,
                conv_history_state, status_display,
            ]
            _bench_reset_outputs = [
                code_display, terminal_display, obs_state,
                history_display, diff_display, reward_display, status_display,
            ]

            reset_btn.click(
                fn=reset_task, inputs=[task_dropdown],
                outputs=_bench_reset_outputs,
            )
            step_btn.click(
                fn=run_agent_step,
                inputs=[task_dropdown, model_name_in, api_key_in,
                        obs_state, history_display, conv_history_state],
                outputs=_bench_outputs,
            )
            run_btn.click(
                fn=run_full_episode,
                inputs=[task_dropdown, model_name_in, api_key_in],
                outputs=_bench_outputs,
            )
            demo.load(
                fn=reset_task, inputs=[task_dropdown],
                outputs=_bench_reset_outputs,
            )

        # ── Tab 2: Rankings ────────────────────────────────────────────────
        with gr.Tab("Rankings"):
            gr.HTML(f"""
            <div style="padding:12px 0 8px;display:flex;align-items:center;gap:8px">
                <span style="color:{_ACCENT}">{ICO.AWARD}</span>
                <span style="font-family:{_FONT_SANS};font-weight:600;font-size:1rem;
                             color:{_TEXT}">Model Rankings</span>
                <span style="font-size:0.75rem;color:{_TEXT_DIM};margin-left:auto;
                             font-family:{_FONT_MONO}">auto-updated after each run</span>
            </div>
            """)
            refresh_lb_btn = gr.Button("Refresh", variant="secondary")
            leaderboard_table = gr.Dataframe(
                headers=["#", "Model", "Mean Reward", "Tasks", "Avg Steps", "Last Run"],
                datatype=["number", "str", "str", "number", "number", "str"],
                interactive=False,
                row_count=10,
            )
            refresh_lb_btn.click(fn=refresh_leaderboard, outputs=[leaderboard_table])
            demo.load(fn=refresh_leaderboard, outputs=[leaderboard_table])

        # ── Tab 3: Replay ──────────────────────────────────────────────────
        with gr.Tab("Replay"):
            gr.HTML(f"""
            <div style="padding:12px 0 8px;display:flex;align-items:center;gap:8px">
                <span style="color:{_ACCENT}">{ICO.FILM}</span>
                <span style="font-family:{_FONT_SANS};font-weight:600;font-size:1rem;
                             color:{_TEXT}">Episode Replay</span>
                <span style="font-size:0.75rem;color:{_TEXT_DIM};margin-left:auto;
                             font-family:{_FONT_MONO}">scrub through past runs step by step</span>
            </div>
            """)

            with gr.Row():
                replay_files_dd = gr.Dropdown(
                    label="Replay File", choices=get_replay_files(),
                    interactive=True, scale=3,
                )
                load_replay_btn = gr.Button("Load", variant="secondary", scale=1)
                refresh_replay_btn = gr.Button("Refresh", variant="secondary", scale=1)

            replay_status = gr.Textbox(label="Status", interactive=False, max_lines=1)
            replay_slider = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Step", interactive=True,
            )

            with gr.Row():
                replay_code = gr.Code(
                    label="Code", language="python",
                    interactive=False, lines=20,
                )
                with gr.Column():
                    replay_terminal = gr.Textbox(
                        label="Terminal", lines=10, interactive=False,
                    )
                    replay_reward = gr.Textbox(
                        label="Step Info", interactive=False, max_lines=2,
                    )

            replay_log_state = gr.State("[]")

            def _load_replay_ui(filename):
                log, max_step, status = load_replay_file(filename)
                log_json = json.dumps(log)
                code, terminal, reward = get_replay_step(log_json, 0) if log else ("", "", "")
                return (
                    log_json,
                    gr.update(maximum=max_step, value=0),
                    code, terminal, reward, status,
                )

            def _scrub_replay(log_json, step_idx):
                return get_replay_step(log_json, int(step_idx))

            load_replay_btn.click(
                fn=_load_replay_ui, inputs=[replay_files_dd],
                outputs=[replay_log_state, replay_slider,
                         replay_code, replay_terminal, replay_reward, replay_status],
            )
            refresh_replay_btn.click(
                fn=lambda: gr.update(choices=get_replay_files()),
                outputs=[replay_files_dd],
            )
            replay_slider.change(
                fn=_scrub_replay,
                inputs=[replay_log_state, replay_slider],
                outputs=[replay_code, replay_terminal, replay_reward],
            )

        # ── Tab 4: Compare ─────────────────────────────────────────────────
        with gr.Tab("Compare"):
            gr.HTML(f"""
            <div style="padding:12px 0 8px;display:flex;align-items:center;gap:8px">
                <span style="color:{_ACCENT}">{ICO.COLUMNS}</span>
                <span style="font-family:{_FONT_SANS};font-weight:600;font-size:1rem;
                             color:{_TEXT}">Model Comparison</span>
                <span style="font-size:0.75rem;color:{_TEXT_DIM};margin-left:auto;
                             font-family:{_FONT_MONO}">runs both models concurrently on the same task</span>
            </div>
            """)
            compare_task_dd = gr.Dropdown(
                choices=list(TASKS.keys()), label="Task", value="level1_keyerror",
            )
            with gr.Row():
                with gr.Column():
                    gr.HTML(f'<div style="font-family:{_FONT_SANS};font-weight:600;color:{_TEXT};padding:4px 0">Model A</div>')
                    compare_model1 = gr.Textbox(label="Model", value=_DEFAULT_MODEL)
                    compare_token1 = gr.Textbox(label="Token", type="password", value=_DEFAULT_TOKEN)
                with gr.Column():
                    gr.HTML(f'<div style="font-family:{_FONT_SANS};font-weight:600;color:{_TEXT};padding:4px 0">Model B</div>')
                    compare_model2 = gr.Textbox(label="Model", value="meta-llama/Llama-3.1-8B-Instruct")
                    compare_token2 = gr.Textbox(label="Token", type="password", value=_DEFAULT_TOKEN)

            compare_btn = gr.Button("Run Comparison", variant="primary")
            compare_status = gr.Textbox(label="Status", interactive=False, max_lines=1)

            with gr.Row():
                compare_result1 = gr.Markdown()
                compare_result2 = gr.Markdown()

            compare_btn.click(
                fn=run_comparison,
                inputs=[compare_task_dd, compare_model1, compare_token1,
                        compare_model2, compare_token2],
                outputs=[compare_result1, compare_result2, compare_status],
            )

        # ── Tab 5: Export ──────────────────────────────────────────────────
        with gr.Tab("Export"):
            gr.HTML(f"""
            <div style="padding:12px 0 8px;display:flex;align-items:center;gap:8px">
                <span style="color:{_ACCENT}">{ICO.FILE_TEXT}</span>
                <span style="font-family:{_FONT_SANS};font-weight:600;font-size:1rem;
                             color:{_TEXT}">Run Report</span>
                <span style="font-size:0.75rem;color:{_TEXT_DIM};margin-left:auto;
                             font-family:{_FONT_MONO}">complete a full episode first</span>
            </div>
            """)
            report_model_in = gr.Textbox(
                label="Model (for report header)", value=_DEFAULT_MODEL,
            )
            with gr.Row():
                gen_report_btn = gr.Button("Generate", variant="primary")
                download_btn = gr.DownloadButton(label="Download .md", visible=False)

            report_preview = gr.Markdown()
            _report_path_state = gr.State(None)

            def _gen_report(model_name):
                preview, path = generate_and_preview_report(model_name)
                return (
                    preview,
                    path,
                    gr.update(visible=path is not None, value=path),
                )

            gen_report_btn.click(
                fn=_gen_report,
                inputs=[report_model_in],
                outputs=[report_preview, _report_path_state, download_btn],
            )


# ---------------------------------------------------------------------------
# Mount Gradio on FastAPI
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(api_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860)
    except KeyboardInterrupt:
        pass
