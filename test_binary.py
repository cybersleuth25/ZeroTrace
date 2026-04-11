from environment.test_runner import grade_level1, grade_level2, grade_level3, grade_level4, grade_level5, grade_level6, grade_level7
from tasks import TASKS

graders = [
    ("L1", grade_level1, "level1_keyerror"),
    ("L2", grade_level2, "level2_resource_leak"),
    ("L3", grade_level3, "level3_race_condition"),
    ("L4", grade_level4, "torch_dtype_mismatch"),
    ("L5", grade_level5, "torch_nan_gradient"),
    ("L6", grade_level6, "torch_wrong_dim"),
    ("L7", grade_level7, "torch_ddp_batch"),
]

print("Grader outputs (must be 0 or 1 only):")
for name, fn, tid in graders:
    buggy = fn(TASKS[tid]["buggy_code"])
    fixed = fn(TASKS[tid]["correct_code"])
    ok_b = buggy in (0, 1)
    ok_f = fixed in (0, 1)
    print(f"  {name}: buggy={buggy} {'OK' if ok_b else 'BAD'}  fixed={fixed} {'OK' if ok_f else 'BAD'}")

print("\nReward from state machine:")
from environment.state_machine import reset_episode, step_episode
from environment.models import Action

for tid in TASKS:
    obs = reset_episode(tid)
    action = Action(action_type="SUBMIT_FIX", patched_code=TASKS[tid]["correct_code"], rationale="test")
    result = step_episode(tid, action)
    rw = result.reward.value
    ok = rw in (0, 1, 0.0, 1.0)
    print(f"  {tid:30s} reward={rw} {'OK' if ok else 'BAD'}")

print("\nDONE")
