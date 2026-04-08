#!/usr/bin/env python3
"""Exhaustively test every grader function to find any score leaks."""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from environment.test_runner import (
    grade_level1, grade_level2, grade_level3,
    grade_level4, grade_level5, grade_level6, grade_level7,
    run_tests, _clamp_score,
)
from tasks import TASKS

def check_score(label: str, score):
    """Check a single score value."""
    ok = isinstance(score, float) and 0.0 < score < 1.0
    status = "OK" if ok else "FAIL"
    print(f"  {status}: {label} = {score!r} (type={type(score).__name__})")
    return ok


def main():
    all_ok = True

    # Test 1: _clamp_score edge cases
    print("=" * 60)
    print("TEST: _clamp_score edge cases")
    print("=" * 60)
    edge_cases = [
        (0.0, "zero"),
        (1.0, "one"),
        (-1.0, "negative"),
        (0.5, "half"),
        (float('nan'), "NaN"),
        (float('inf'), "inf"),
        (float('-inf'), "-inf"),
        (0.001, "tiny"),
        (0.999, "near-one"),
        (2.0, "over-one"),
    ]
    for val, desc in edge_cases:
        result = _clamp_score(val)
        if not check_score(f"_clamp_score({val}) [{desc}]", result):
            all_ok = False

    # Test 2: grade_level* with buggy code
    print("\n" + "=" * 60)
    print("TEST: Graders with BUGGY code")
    print("=" * 60)

    grader_map = {
        "level1_keyerror": (grade_level1, 1),
        "level2_resource_leak": (grade_level2, 2),
        "level3_race_condition": (grade_level3, 3),
        "torch_dtype_mismatch": (grade_level4, 4),
        "torch_nan_gradient": (grade_level5, 5),
        "torch_wrong_dim": (grade_level6, 6),
        "torch_ddp_batch": (grade_level7, 7),
    }

    for task_id, (grader_fn, level) in grader_map.items():
        task = TASKS.get(task_id)
        if not task:
            print(f"  SKIP: {task_id} not in TASKS")
            continue

        buggy = task["buggy_code"]
        correct = task["correct_code"]

        # Grade buggy code via grade_level* function
        try:
            score = grader_fn(buggy)
            if not check_score(f"{task_id} buggy (grade_fn)", score):
                all_ok = False
        except Exception as e:
            print(f"  FAIL: {task_id} buggy (grade_fn) EXCEPTION: {e}")
            all_ok = False

        # Grade buggy code via run_tests
        try:
            result = run_tests(level, buggy)
            s = result.get("score")
            if not check_score(f"{task_id} buggy (run_tests).score", s):
                all_ok = False
        except Exception as e:
            print(f"  FAIL: {task_id} buggy (run_tests) EXCEPTION: {e}")
            all_ok = False

    # Test 3: grade_level* with correct code
    print("\n" + "=" * 60)
    print("TEST: Graders with CORRECT code")
    print("=" * 60)

    for task_id, (grader_fn, level) in grader_map.items():
        task = TASKS.get(task_id)
        if not task:
            continue

        correct = task["correct_code"]

        # Grade correct code via grade_level* function
        try:
            score = grader_fn(correct)
            if not check_score(f"{task_id} correct (grade_fn)", score):
                all_ok = False
        except Exception as e:
            print(f"  FAIL: {task_id} correct (grade_fn) EXCEPTION: {e}")
            all_ok = False

        # Grade correct code via run_tests
        try:
            result = run_tests(level, correct)
            s = result.get("score")
            if not check_score(f"{task_id} correct (run_tests).score", s):
                all_ok = False
        except Exception as e:
            print(f"  FAIL: {task_id} correct (run_tests) EXCEPTION: {e}")
            all_ok = False

    # Test 4: edge cases for graders
    print("\n" + "=" * 60)
    print("TEST: Graders with edge-case inputs")
    print("=" * 60)
    edge_inputs = [
        ("empty string", ""),
        ("None-like", "None"),
        ("syntax error", "def foo(:\n  pass"),
        ("just a comment", "# nothing here"),
    ]
    for desc, code in edge_inputs:
        for task_id, (grader_fn, level) in grader_map.items():
            try:
                score = grader_fn(code)
                if not check_score(f"{task_id} [{desc}]", score):
                    all_ok = False
            except Exception as e:
                print(f"  FAIL: {task_id} [{desc}] EXCEPTION: {e}")
                all_ok = False

    # Final verdict
    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — scores out of (0, 1) detected!")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
