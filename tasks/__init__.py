"""Task definitions for ZeroTrace.

Levels 1-3: Classic Python bugs (easy → hard).
Levels 4-6: PyTorch-specific bugs (easy → hard).
"""

from . import level1, level2, level3
from . import torch_dtype, torch_nan_grad, torch_wrong_dim

TASKS = {
    # ── Classic Python ──────────────────────────────────────────────────────
    "level1_keyerror": {
        "name": "Fix KeyError Bug",
        "level": 1,
        "difficulty": "easy",
        "description": level1.DESCRIPTION,
        "buggy_code": level1.BUGGY_CODE,
        "correct_code": level1.CORRECT_CODE,
    },
    "level2_resource_leak": {
        "name": "Fix Resource Leak",
        "level": 2,
        "difficulty": "medium",
        "description": level2.DESCRIPTION,
        "buggy_code": level2.BUGGY_CODE,
        "correct_code": level2.CORRECT_CODE,
    },
    "level3_race_condition": {
        "name": "Fix Race Condition",
        "level": 3,
        "difficulty": "hard",
        "description": level3.DESCRIPTION,
        "buggy_code": level3.BUGGY_CODE,
        "correct_code": level3.CORRECT_CODE,
    },
    # ── PyTorch ─────────────────────────────────────────────────────────────
    "torch_dtype_mismatch": {
        "name": "Fix PyTorch Dtype Mismatch",
        "level": 4,
        "difficulty": "pytorch-easy",
        "description": torch_dtype.DESCRIPTION,
        "buggy_code": torch_dtype.BUGGY_CODE,
        "correct_code": torch_dtype.CORRECT_CODE,
    },
    "torch_nan_gradient": {
        "name": "Fix PyTorch NaN Gradient",
        "level": 5,
        "difficulty": "pytorch-medium",
        "description": torch_nan_grad.DESCRIPTION,
        "buggy_code": torch_nan_grad.BUGGY_CODE,
        "correct_code": torch_nan_grad.CORRECT_CODE,
    },
    "torch_wrong_dim": {
        "name": "Fix PyTorch Softmax Dimension",
        "level": 6,
        "difficulty": "pytorch-hard",
        "description": torch_wrong_dim.DESCRIPTION,
        "buggy_code": torch_wrong_dim.BUGGY_CODE,
        "correct_code": torch_wrong_dim.CORRECT_CODE,
    },
}

__all__ = [
    "TASKS",
    "level1", "level2", "level3",
    "torch_dtype", "torch_nan_grad", "torch_wrong_dim",
]
