"""Test that all score/reward values are force-clamped to (0, 1)."""
from environment.models import Observation, Reward, StepResult

print("=== Reward model tests ===")
for val, label in [(0.0, "zero"), (1.0, "one"), (-5.0, "neg"), (0.5, "mid"),
                   (0.99, "max"), (float("nan"), "nan"), (float("inf"), "inf")]:
    r = Reward(value=val, reason="test")
    ok = 0.0 < r.value < 1.0
    print(f"  Reward({label:4s}) -> {r.value}  {'OK' if ok else 'FAIL'}")
    assert ok, f"Reward value {r.value} out of range for input {val}"

print("\n=== Observation model tests ===")
for val in [0.0, 1.0, -1.0, 0.5, 0.99]:
    o = Observation(task_id="t", level=1, buggy_code="x", reward=val)
    ok = 0.0 < o.reward < 1.0
    score = o.test_results.get("score", "MISSING")
    print(f"  Obs(reward={val}) -> reward={o.reward} score={score}  {'OK' if ok else 'FAIL'}")
    assert ok, f"Observation reward {o.reward} out of range for input {val}"

print("\n=== test_results.score clamping ===")
for score_val in [0.0, 1.0, -1.0, 0.5, 0.99]:
    o = Observation(
        task_id="t", level=1, buggy_code="x",
        test_results={"passed": 0, "total": 0, "score": score_val}
    )
    s = o.test_results["score"]
    ok = 0.0 < s < 1.0
    print(f"  test_results(score={score_val}) -> {s}  {'OK' if ok else 'FAIL'}")
    assert ok, f"test_results score {s} out of range for input {score_val}"

print("\n=== Full episode flow test ===")
from environment.state_machine import reset_episode, step_episode
from environment.models import Action
from tasks import TASKS

for task_id in list(TASKS.keys()):
    obs = reset_episode(task_id)
    ok = 0.0 < obs.reward < 1.0
    sok = 0.0 < obs.test_results.get("score", 0) < 1.0
    print(f"  reset({task_id:30s}) reward={obs.reward:.4f} score={obs.test_results.get('score'):.4f}  {'OK' if ok and sok else 'FAIL'}")
    assert ok, f"Reset reward {obs.reward} out of range for {task_id}"
    assert sok, f"Reset score {obs.test_results.get('score')} out of range for {task_id}"

    # Test SUBMIT_FIX with correct code
    correct = TASKS[task_id].get("correct_code", "")
    if correct:
        action = Action(action_type="SUBMIT_FIX", patched_code=correct, rationale="test")
        result = step_episode(task_id, action)
        rv = result.reward.value
        ov = result.observation.reward
        sv = result.observation.test_results.get("score", 0)
        ok = 0.0 < rv < 1.0 and 0.0 < ov < 1.0 and 0.0 < sv < 1.0
        print(f"  step ({task_id:30s}) reward.val={rv:.4f} obs.reward={ov:.4f} score={sv:.4f}  {'OK' if ok else 'FAIL'}")
        assert 0.0 < rv < 1.0, f"Step reward {rv} out of range"
        assert 0.0 < ov < 1.0, f"Obs reward {ov} out of range"
        assert 0.0 < sv < 1.0, f"Score {sv} out of range"

print("\n=== Grader function tests ===")
from environment.test_runner import (
    grade_level1, grade_level2, grade_level3,
    grade_level4, grade_level5, grade_level6, grade_level7
)

graders = [
    ("grade_level1", grade_level1),
    ("grade_level2", grade_level2),
    ("grade_level3", grade_level3),
    ("grade_level4", grade_level4),
    ("grade_level5", grade_level5),
    ("grade_level6", grade_level6),
    ("grade_level7", grade_level7),
]

for name, fn in graders:
    for code_label, code in [("empty", ""), ("none_str", "None"), ("garbage", "aslkdjf")]:
        score = fn(code)
        ok = 0.0 < score < 1.0
        print(f"  {name}({code_label:10s}) -> {score:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, f"{name}({code_label}) returned {score} out of range"

    for task_id, task in TASKS.items():
        for variant in ["buggy_code", "correct_code"]:
            code = task.get(variant, "")
            if code:
                score = fn(code)
                ok = 0.0 < score < 1.0
                if not ok:
                    print(f"  {name}({task_id}/{variant}) -> {score:.4f}  FAIL!")
                    assert False, f"{name}({task_id}/{variant}) returned {score}"

print("\n=== ALL TESTS PASSED ===")
