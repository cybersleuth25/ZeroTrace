"""Level 5: PyTorch NaN Gradient Bug (PyTorch / Medium)"""

DESCRIPTION = """Fix a NaN gradient bug in a custom cross-entropy implementation.

The buggy code calls torch.log(predictions) directly. When any prediction
is exactly 0.0, log(0) = -inf, which propagates as NaN through the loss
and corrupts model weights during backpropagation.

Your task: add numerical stability (epsilon clamping) so log is never
called on zero or negative values.
"""

BUGGY_CODE = '''import torch


def custom_cross_entropy(predictions, targets):
    """Compute a custom cross-entropy loss.

    Args:
        predictions: Probability tensor of shape (batch, num_classes).
                     Values are in [0, 1] and may include exact zeros.
        targets:     One-hot encoded target tensor, same shape.

    Returns:
        Scalar loss value.
    """
    # Bug: torch.log(0) = -inf, which causes NaN during backward().
    log_probs = torch.log(predictions)
    loss = -torch.sum(targets * log_probs) / predictions.shape[0]
    return loss


if __name__ == "__main__":
    predictions = torch.tensor([[0.0, 0.5, 0.5], [0.3, 0.7, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = custom_cross_entropy(predictions, targets)
    print(f"Loss: {loss}")  # Prints nan or inf — broken!
'''

CORRECT_CODE = '''import torch


def custom_cross_entropy(predictions, targets):
    """Compute a numerically stable custom cross-entropy loss.

    Args:
        predictions: Probability tensor of shape (batch, num_classes).
                     Values are in [0, 1] and may include exact zeros.
        targets:     One-hot encoded target tensor, same shape.

    Returns:
        Scalar loss value.
    """
    # Fix: clamp predictions to a small positive value before taking log.
    eps = 1e-8
    log_probs = torch.log(predictions.clamp(min=eps))
    loss = -torch.sum(targets * log_probs) / predictions.shape[0]
    return loss


if __name__ == "__main__":
    predictions = torch.tensor([[0.0, 0.5, 0.5], [0.3, 0.7, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = custom_cross_entropy(predictions, targets)
    print(f"Loss: {loss}")  # Prints a valid finite number — fixed!
'''
