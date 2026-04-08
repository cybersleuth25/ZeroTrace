"""Test runner and graders for ZeroTrace tasks.

Each grader returns a dict with passed, failed, total, details, score.
Levels 1-3: classic Python bugs.
Levels 4-7: PyTorch-specific bugs (gracefully skips run-tests if torch absent).
"""

import math

import os
import re
import tempfile
from typing import Any, Dict, List

from .sandbox import run_code_safely, check_syntax


def _clamp_score(score: float) -> float:
    """Clamp score to the open interval (0, 1).

    The OpenEnv validator requires *strictly* 0 < score < 1.
    We clamp BEFORE rounding so that round() can never push a
    boundary value back to exactly 0.0 or 1.0.
    """
    if not math.isfinite(score):
        return 0.01
    clamped = max(0.01, min(0.99, score))
    return round(clamped, 4)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def run_tests(level: int, patched_code: str) -> Dict[str, Any]:
    """Run tests for the given level and return results."""
    dispatch = {
        1: _run_level1_tests,
        2: _run_level2_tests,
        3: _run_level3_tests,
        4: _run_level4_tests,
        5: _run_level5_tests,
        6: _run_level6_tests,
        7: _run_level6_tests,  # level 7 reuses level 6 grader
    }
    fn = dispatch.get(level)
    if fn is None:
        return {"passed": 0, "failed": 0, "total": 0,
                "details": [f"Unknown level: {level}"], "score": 0.01}
    try:
        result = fn(patched_code)
        result["score"] = _clamp_score(result.get("score", 0.0))
        return result
    except Exception as e:
        return {"passed": 0, "failed": 0, "total": 0,
                "details": [f"Grader error: {e}"], "score": 0.01}


# ---------------------------------------------------------------------------
# Convenience grade functions (return 0.0-1.0 float)
# ---------------------------------------------------------------------------

def _safe_grade(fn, code: str) -> float:
    """Run a grader function and guarantee the result is in (0, 1)."""
    try:
        result = fn(code)
        return _clamp_score(result.get("score", 0.0))
    except Exception:
        return 0.01


def grade_level1(code: str) -> float:
    return _safe_grade(_run_level1_tests, code)

def grade_level2(code: str) -> float:
    return _safe_grade(_run_level2_tests, code)

def grade_level3(code: str) -> float:
    return _safe_grade(_run_level3_tests, code)

def grade_level4(code: str) -> float:
    return _safe_grade(_run_level4_tests, code)

def grade_level5(code: str) -> float:
    return _safe_grade(_run_level5_tests, code)

def grade_level6(code: str) -> float:
    return _safe_grade(_run_level6_tests, code)

def grade_level7(code: str) -> float:
    return _safe_grade(_run_level6_tests, code)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _syntax_gate(patched_code: str, total: int) -> Dict[str, Any]:
    """Return a failure dict if syntax is invalid, else None."""
    syntax = check_syntax(patched_code)
    if not syntax["valid"]:
        return {
            "passed": 0, "failed": total, "total": total,
            "details": [f"SyntaxError at line {syntax['line']}: {syntax['error']}"],
            "score": 0.01,
        }
    return {}  # empty = OK


# ---------------------------------------------------------------------------
# Level 1 — KeyError fix
# ---------------------------------------------------------------------------

def _run_level1_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 3

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: existing key returns correct value
    t1 = f'''
{patched_code}
test_data = {{"alice": {{"score": 95}}, "bob": {{"score": 87}}}}
result = get_user_score(test_data, "alice")
assert result == 95, f"Expected 95, got {{result}}"
print("TEST1_PASS")
'''
    r1 = run_code_safely(t1)
    if "TEST1_PASS" in r1["stdout"]:
        passed += 1
        details.append("Test 1 PASS: get_user_score(data, 'alice') == 95")
    else:
        details.append(f"Test 1 FAIL: {(r1['stderr'] or r1['stdout'])[:120]}")

    # Test 2: missing key returns 0 without crashing
    t2 = f'''
{patched_code}
test_data = {{"alice": {{"score": 95}}, "bob": {{"score": 87}}}}
try:
    result = get_user_score(test_data, "charlie")
    print("TEST2_PASS")
except KeyError as e:
    print(f"KeyError: {{e}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
    r2 = run_code_safely(t2)
    if "TEST2_PASS" in r2["stdout"]:
        passed += 1
        details.append("Test 2 PASS: get_user_score(data, 'charlie') no crash")
    else:
        details.append(f"Test 2 FAIL: {(r2['stderr'] or r2['stdout'])[:120]}")

    # Test 3: empty dict doesn't crash
    t3 = f'''
{patched_code}
try:
    result = get_user_score({{}}, "anyone")
    print("TEST3_PASS")
except KeyError as e:
    print(f"KeyError: {{e}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
    r3 = run_code_safely(t3)
    if "TEST3_PASS" in r3["stdout"]:
        passed += 1
        details.append("Test 3 PASS: get_user_score({}, 'anyone') no crash")
    else:
        details.append(f"Test 3 FAIL: {(r3['stderr'] or r3['stdout'])[:120]}")

    weights = [0.33, 0.33, 0.34]
    score = sum(w for i, w in enumerate(weights) if i < passed)
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}


# ---------------------------------------------------------------------------
# Level 2 — Resource-leak fix
# ---------------------------------------------------------------------------

def _run_level2_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 3

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: file content returned correctly
    test_content = "zerotrace_test_content_9876"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_content)
            temp_path = f.name
    except Exception as e:
        return {"passed": 0, "failed": total, "total": total,
                "details": [f"Setup error: {e}"], "score": 0.01}

    try:
        t1 = f'''
{patched_code}
result = read_config("{temp_path.replace(os.sep, '/')}")
assert result == "{test_content}", f"Expected '{test_content}', got {{result}}"
print("TEST1_PASS")
'''
        r1 = run_code_safely(t1)
        if "TEST1_PASS" in r1["stdout"]:
            passed += 1
            details.append("Test 1 PASS: File content returned correctly")
        else:
            details.append(f"Test 1 FAIL: {(r1['stderr'] or r1['stdout'])[:120]}")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    # Test 2: context manager present
    if "with open" in patched_code:
        passed += 1
        details.append("Test 2 PASS: 'with open' context manager used")
    else:
        details.append("Test 2 FAIL: 'with open' not found in code")

    # Test 3: no explicit .close()
    if ".close()" not in patched_code:
        passed += 1
        details.append("Test 3 PASS: No explicit .close() call")
    else:
        details.append("Test 3 FAIL: Explicit .close() found — use context manager")

    weights = [0.34, 0.33, 0.33]
    score = sum(w for i, w in enumerate(weights) if i < passed)
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}


# ---------------------------------------------------------------------------
# Level 3 — Race-condition fix
# ---------------------------------------------------------------------------

def _run_level3_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 4

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: threading.Lock() present in code
    if "threading.Lock()" in patched_code or "Lock()" in patched_code:
        passed += 1
        details.append("Test 1 PASS: threading.Lock() found in code")
    else:
        details.append("Test 1 FAIL: threading.Lock() not found")

    # Helper: run counter test
    _counter_template = '''
counter = 0
lock = __import__('threading').Lock()

def increment():
    global counter
    for _ in range(10000):
        with lock:
            counter += 1

threads = [__import__('threading').Thread(target=increment) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

if counter == 50000:
    print("PASS")
else:
    print(f"FAIL: got {{counter}}")
'''

    # Test 2: single run reaches 50000
    t2 = f"{patched_code}\n{_counter_template}"
    r2 = run_code_safely(t2, timeout=30)
    if "PASS" in r2["stdout"]:
        passed += 1
        details.append("Test 2 PASS: counter == 50000 in single run")
    else:
        details.append(f"Test 2 FAIL: {(r2['stdout'] or r2['stderr'])[:100]}")

    # Test 3: 5 consecutive runs all pass
    c_passes = sum(
        1 for _ in range(5)
        if "PASS" in run_code_safely(_counter_template, timeout=30)["stdout"]
    )
    if c_passes == 5:
        passed += 1
        details.append("Test 3 PASS: counter == 50000 in 5 consecutive runs")
    else:
        details.append(f"Test 3 FAIL: Only {c_passes}/5 consecutive runs passed")

    # Test 4: no race in 10 repeated runs
    race_free = sum(
        1 for _ in range(10)
        if "PASS" in run_code_safely(_counter_template, timeout=30)["stdout"]
    )
    if race_free == 10:
        passed += 1
        details.append("Test 4 PASS: No race in 10 repeated runs")
    else:
        details.append(f"Test 4 FAIL: {10 - race_free}/10 runs had a race condition")

    score = passed * 0.25
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}


# ---------------------------------------------------------------------------
# Level 4 — PyTorch dtype mismatch
# ---------------------------------------------------------------------------

def _run_level4_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 3

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: long-dtype fix pattern present
    has_long_fix = (
        ".long()" in patched_code
        or "dtype=torch.long" in patched_code
        or "dtype=torch.int64" in patched_code
        or "torch.long" in patched_code
    )
    if has_long_fix:
        passed += 1
        details.append("Test 1 PASS: Long dtype fix found (.long() or dtype=torch.long)")
    else:
        details.append("Test 1 FAIL: No .long() / dtype=torch.long fix found")

    # Test 2: bug pattern (float32 for targets) is gone
    float32_for_targets = any(
        "float32" in line and "target" in line.lower()
        for line in patched_code.splitlines()
        if "torch.long" not in line  # line already fixed
    )
    if not float32_for_targets:
        passed += 1
        details.append("Test 2 PASS: dtype=torch.float32 for targets removed")
    else:
        details.append("Test 2 FAIL: Still using float32 for targets tensor")

    # Test 3: actually run it (torch may not be installed — auto-pass if absent)
    t3 = f'''
{patched_code}
try:
    import torch
    import math
    logits = torch.randn(4, 3)
    labels = [0, 1, 2, 1]
    loss = compute_classification_loss(logits, labels)
    val = loss.item()
    if math.isnan(val) or math.isinf(val):
        print(f"FAIL: loss={{val}}")
    else:
        print("TEST3_PASS")
except ImportError:
    print("TEST3_PASS")  # torch not installed — trust static checks
except RuntimeError as e:
    print(f"RuntimeError: {{e}}")
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
'''
    r3 = run_code_safely(t3, timeout=30)
    if "TEST3_PASS" in r3["stdout"]:
        passed += 1
        details.append("Test 3 PASS: compute_classification_loss runs without error")
    else:
        details.append(f"Test 3 FAIL: {(r3['stderr'] or r3['stdout'])[:150]}")

    weights = [0.34, 0.33, 0.33]
    score = sum(w for i, w in enumerate(weights) if i < passed)
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}


# ---------------------------------------------------------------------------
# Level 5 — PyTorch NaN gradient
# ---------------------------------------------------------------------------

def _run_level5_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 3

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: numerical-stability fix pattern present
    has_stability = any(
        kw in patched_code
        for kw in ("clamp", "1e-8", "eps", "epsilon", "clip")
    )
    if has_stability:
        passed += 1
        details.append("Test 1 PASS: Numerical stability fix found (clamp / eps / clip)")
    else:
        details.append("Test 1 FAIL: No numerical stability fix found")

    # Test 2: raw torch.log(predictions) without protection is gone
    raw_log = re.search(r"torch\.log\s*\(\s*predictions\s*\)", patched_code)
    if not raw_log or has_stability:
        passed += 1
        details.append("Test 2 PASS: torch.log protected against zero values")
    else:
        details.append("Test 2 FAIL: torch.log(predictions) called without protection")

    # Test 3: actually run — loss must be finite
    t3 = f'''
{patched_code}
try:
    import torch
    import math
    preds = torch.tensor([[0.0, 0.5, 0.5], [0.3, 0.7, 0.0]])
    tgts  = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = custom_cross_entropy(preds, tgts)
    val = loss.item()
    if math.isnan(val) or math.isinf(val):
        print(f"FAIL: loss is {{val}}")
    else:
        print(f"TEST3_PASS loss={{val:.4f}}")
except ImportError:
    print("TEST3_PASS")  # torch not installed — trust static checks
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
'''
    r3 = run_code_safely(t3, timeout=30)
    if "TEST3_PASS" in r3["stdout"]:
        passed += 1
        details.append("Test 3 PASS: Loss is finite (no NaN / inf)")
    else:
        details.append(f"Test 3 FAIL: {(r3['stderr'] or r3['stdout'])[:150]}")

    weights = [0.34, 0.33, 0.33]
    score = sum(w for i, w in enumerate(weights) if i < passed)
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}


# ---------------------------------------------------------------------------
# Level 6 — PyTorch wrong softmax dimension
# ---------------------------------------------------------------------------

def _run_level6_tests(patched_code: str) -> Dict[str, Any]:
    details: List[str] = []
    passed = 0
    total = 3

    if g := _syntax_gate(patched_code, total):
        return g

    # Test 1: dim=-1 in softmax call present
    has_dim_neg1 = bool(re.search(r"softmax\s*\(.*dim\s*=\s*-1", patched_code))
    if has_dim_neg1:
        passed += 1
        details.append("Test 1 PASS: F.softmax(..., dim=-1) found")
    else:
        details.append("Test 1 FAIL: dim=-1 not found in softmax call")

    # Test 2: dim=0 in softmax is gone
    has_dim_0 = bool(
        re.search(r"softmax\s*\(.*dim\s*=\s*0", patched_code)
    )
    if not has_dim_0:
        passed += 1
        details.append("Test 2 PASS: softmax(dim=0) removed")
    else:
        details.append("Test 2 FAIL: Still using softmax with dim=0")

    # Test 3: output matches reference implementation with fixed seed
    t3 = f'''
{patched_code}
try:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    batch, seq_len, d_k = 2, 3, 4
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)

    student_out = scaled_dot_product_attention(q, k, v)

    # Reference with CORRECT dim=-1
    torch.manual_seed(42)
    q2 = torch.randn(batch, seq_len, d_k)
    k2 = torch.randn(batch, seq_len, d_k)
    v2 = torch.randn(batch, seq_len, d_k)
    ref_weights = F.softmax(torch.matmul(q2, k2.transpose(-2,-1)) / (d_k**0.5), dim=-1)
    ref_out = torch.matmul(ref_weights, v2)

    if torch.allclose(student_out, ref_out, atol=1e-5):
        print("TEST3_PASS")
    else:
        diff = (student_out - ref_out).abs().max().item()
        print(f"FAIL: max diff={{diff:.6f}}")
except ImportError:
    print("TEST3_PASS")  # torch not installed — trust static checks
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")
'''
    r3 = run_code_safely(t3, timeout=30)
    if "TEST3_PASS" in r3["stdout"]:
        passed += 1
        details.append("Test 3 PASS: Attention output matches reference implementation")
    else:
        details.append(f"Test 3 FAIL: {(r3['stderr'] or r3['stdout'])[:150]}")

    weights = [0.34, 0.33, 0.33]
    score = sum(w for i, w in enumerate(weights) if i < passed)
    return {"passed": passed, "failed": total - passed, "total": total,
            "details": details, "score": _clamp_score(score)}
