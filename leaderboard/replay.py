"""Episode replay storage for ZeroTrace.

Saves step-by-step snapshots of an episode to JSON files so they can
be scrubbed through in the Gradio Replay tab.
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPLAY_DIR = Path(__file__).parent / "data" / "replays"
_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_MAX_REPLAYS = 50  # keep last 50 replay files on disk


def save_replay(task_id: str, replay_log: List[Dict[str, Any]]) -> str:
    """Persist a replay log and return the filename (without directory).

    Args:
        task_id:    Task identifier.
        replay_log: List of step snapshots produced by EpisodeState.

    Returns:
        Filename of the saved replay (e.g. "level1_keyerror_20260401_180900.json").
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{ts}.json"
    path = _REPLAY_DIR / filename

    with _lock:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(replay_log, fh, indent=2, ensure_ascii=False)
        except OSError:
            return ""

        # Prune old files if over the limit
        files = sorted(_REPLAY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in files[:-_MAX_REPLAYS]:
            try:
                old.unlink()
            except OSError:
                pass

    return filename


def list_replays() -> List[str]:
    """Return a sorted list of available replay filenames (newest first)."""
    with _lock:
        files = sorted(
            _REPLAY_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [f.name for f in files]


def load_replay(filename: str) -> List[Dict[str, Any]]:
    """Load a replay log by filename.

    Args:
        filename: Bare filename (no directory), as returned by list_replays().

    Returns:
        List of step snapshots, or [] on error.
    """
    # Security: reject any path traversal attempt
    safe_name = Path(filename).name
    if safe_name != filename:
        return []

    path = _REPLAY_DIR / safe_name
    with _lock:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
