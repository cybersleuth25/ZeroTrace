"""Level 4: PyTorch Dtype Mismatch Bug (PyTorch / Easy)"""

DESCRIPTION = """Fix a dtype mismatch bug in a PyTorch cross-entropy loss computation.

torch.nn.CrossEntropyLoss expects class indices as Long (int64) tensors.
The buggy code creates the target tensor with dtype=torch.float32, which
causes a RuntimeError: "expected scalar type Long but found Float".

Your task: fix the target tensor dtype so the loss computes correctly.
"""

BUGGY_CODE = '''import torch
import torch.nn as nn


def compute_classification_loss(logits, targets):
    """Compute cross-entropy loss for a classification task.

    Args:
        logits:  (batch_size, num_classes) float tensor of raw model outputs.
        targets: List of integer class indices.

    Returns:
        Scalar loss tensor.
    """
    criterion = nn.CrossEntropyLoss()
    # Bug: CrossEntropyLoss expects Long (int64) targets, not Float.
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    return criterion(logits, targets_tensor)


if __name__ == "__main__":
    logits = torch.randn(4, 3)
    labels = [0, 1, 2, 1]
    loss = compute_classification_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}")
'''

CORRECT_CODE = '''import torch
import torch.nn as nn


def compute_classification_loss(logits, targets):
    """Compute cross-entropy loss for a classification task.

    Args:
        logits:  (batch_size, num_classes) float tensor of raw model outputs.
        targets: List of integer class indices.

    Returns:
        Scalar loss tensor.
    """
    criterion = nn.CrossEntropyLoss()
    # Fix: dtype=torch.long (int64) is required by CrossEntropyLoss.
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return criterion(logits, targets_tensor)


if __name__ == "__main__":
    logits = torch.randn(4, 3)
    labels = [0, 1, 2, 1]
    loss = compute_classification_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}")
'''
