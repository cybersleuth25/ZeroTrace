"""Thread-safe JSON-file leaderboard store for ZeroTrace.

Stores per-task results keyed by (model, task_id).
No external dependencies — uses only stdlib json + threading.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Storage path
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_LB_FILE = _DATA_DIR / "leaderboard.json"

_lock = threading.Lock()

_MAX_ENTRIES = 500  # hard cap to keep the file small


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> List[Dict[str, Any]]:
    """Load the leaderboard from disk (caller must hold _lock)."""
    if not _LB_FILE.exists():
        return []
    try:
        with open(_LB_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save(entries: List[Dict[str, Any]]) -> None:
    """Persist the leaderboard to disk (caller must hold _lock)."""
    try:
        with open(_LB_FILE, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
    except OSError:
        pass  # Non-fatal — leaderboard is best-effort


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_result(
    model: str,
    task_id: str,
    reward: float,
    steps: int,
) -> None:
    """Append one task result to the leaderboard.

    Args:
        model:   Model identifier string (e.g. "Qwen/Qwen2.5-72B-Instruct").
        task_id: Task identifier (e.g. "level1_keyerror").
        reward:  Final reward value in [-1.0, 1.0].
        steps:   Number of steps taken to complete the episode.
    """
    entry: Dict[str, Any] = {
        "model": model,
        "task_id": task_id,
        "reward": round(float(reward), 4),
        "steps": int(steps),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        # Keep only the most recent _MAX_ENTRIES records
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
        _save(entries)


def get_leaderboard() -> Dict[str, Any]:
    """Return raw entries and per-model aggregated statistics.

    Returns:
        {
            "all_results": [...],
            "models": [
                {
                    "rank": 1,
                    "model": "...",
                    "mean_reward": 1.0,
                    "tasks_completed": 6,
                    "avg_steps": 3.0,
                    "last_run": "2026-04-01T...",
                }
            ]
        }
    """
    with _lock:
        entries = _load()

    model_data: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        m = e["model"]
        if m not in model_data:
            model_data[m] = {
                "rewards": [],
                "steps": [],
                "timestamps": [],
            }
        model_data[m]["rewards"].append(e["reward"])
        model_data[m]["steps"].append(e["steps"])
        model_data[m]["timestamps"].append(e["timestamp"])

    aggregated = []
    for model, data in model_data.items():
        rewards = data["rewards"]
        steps = data["steps"]
        aggregated.append(
            {
                "model": model,
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                "tasks_completed": len(rewards),
                "avg_steps": round(sum(steps) / len(steps), 1),
                "last_run": max(data["timestamps"]),
            }
        )

    # Sort by mean_reward desc, then avg_steps asc
    aggregated.sort(key=lambda x: (-x["mean_reward"], x["avg_steps"]))
    for i, row in enumerate(aggregated, start=1):
        row["rank"] = i

    return {"all_results": entries, "models": aggregated}


def get_model_stats(model: str) -> Dict[str, Any]:
    """Return statistics for a single model.

    Args:
        model: Model identifier string.

    Returns:
        Dict with mean_reward, tasks_completed, avg_steps, per_task breakdown.
    """
    with _lock:
        entries = _load()

    model_entries = [e for e in entries if e["model"] == model]
    if not model_entries:
        return {"model": model, "tasks_completed": 0}

    per_task: Dict[str, List[float]] = {}
    for e in model_entries:
        per_task.setdefault(e["task_id"], []).append(e["reward"])

    return {
        "model": model,
        "mean_reward": round(
            sum(e["reward"] for e in model_entries) / len(model_entries), 4
        ),
        "tasks_completed": len(model_entries),
        "avg_steps": round(
            sum(e["steps"] for e in model_entries) / len(model_entries), 1
        ),
        "per_task": {
            tid: round(sum(rs) / len(rs), 4) for tid, rs in per_task.items()
        },
    }


def clear_leaderboard() -> None:
    """Wipe the leaderboard (useful for testing)."""
    with _lock:
        _save([])
