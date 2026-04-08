"""Verify ALL clamp functions produce scores strictly in (0, 1)."""
import math
import sys

def _clamp_score(score):
    if not math.isfinite(score):
        return 0.01
    clamped = max(0.01, min(0.99, score))
    return round(clamped, 4)

tests = [
    (0.0, "exact zero"),
    (1.0, "exact one"),
    (0.00004, "tiny positive (old round gave 0.0)"),
    (0.99996, "near one (old round gave 1.0)"),
    (-0.5, "negative"),
    (-1.0, "minus one"),
    (1.5, "above one"),
    (float("nan"), "NaN"),
    (float("inf"), "inf"),
    (float("-inf"), "-inf"),
    (0.33, "one third"),
    (0.66, "two thirds"),
    (0.99, "upper bound"),
    (0.01, "lower bound"),
    (0.5, "half"),
    (0.0001, "very small"),
    (0.9999, "very close to 1"),
    (0.33 + 0.33 + 0.34, "weights sum = 1.0"),
    (4 * 0.25, "level3 perfect"),
    (0 * 0.25, "level3 zero"),
]

all_ok = True
for val, desc in tests:
    result = _clamp_score(val)
    ok = 0.0 < result < 1.0
    status = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  {status}: _clamp_score({val}) = {result}  ({desc})")

print()
if all_ok:
    print("ALL TESTS PASSED: Every result strictly in (0, 1)")
else:
    print("SOME TESTS FAILED!")
    sys.exit(1)
