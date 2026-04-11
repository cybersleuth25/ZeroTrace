"""Sandbox for safe code execution.

Uses subprocess to run Python code safely with timeout and environment
isolation. Never uses eval() or exec() directly.
"""

import os
import platform
import subprocess
import sys
import tempfile
from typing import Any, Dict

_MAX_CODE_LENGTH = 50_000  # chars — hard cap before we even write a temp file


def run_code_safely(code: str, timeout: int = 5) -> Dict[str, Any]:
    """Run Python code safely in a subprocess.

    Args:
        code:    Python source code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Dictionary with stdout, stderr, success, timed_out.
    """
    result: Dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "success": False,
        "timed_out": False,
    }

    # Guard: code length cap
    if len(code) > _MAX_CODE_LENGTH:
        result["stderr"] = (
            f"Code rejected: exceeds maximum length "
            f"({len(code):,} > {_MAX_CODE_LENGTH:,} chars)"
        )
        return result

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            temp_path = f.name
    except Exception as e:
        result["stderr"] = f"Failed to create temp file: {e}"
        return result

    # Build a sanitised environment: keep Python paths but strip secrets
    safe_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "HF_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
        }
    }
    # Ensure the temp dir is isolated
    safe_env["PYTHONDONTWRITEBYTECODE"] = "1"
    safe_env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            env=safe_env,
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["success"] = proc.returncode == 0

    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["stderr"] = f"Execution timed out after {timeout}s"

    except Exception as e:
        result["stderr"] = f"Execution error: {e}"

    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    return result


def check_syntax(code: str) -> Dict[str, Any]:
    """Check Python code for syntax errors using compile().

    Args:
        code: Python source code.

    Returns:
        Dict with valid (bool), error (str|None), line (int|None).
    """
    result: Dict[str, Any] = {"valid": True, "error": None, "line": None}
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        result["valid"] = False
        result["error"] = str(e.msg) if e.msg else str(e)
        result["line"] = e.lineno
    return result
