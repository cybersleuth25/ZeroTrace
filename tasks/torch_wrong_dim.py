"""Level 6: PyTorch Wrong Softmax Dimension Bug (PyTorch / Hard)"""

DESCRIPTION = """Fix a wrong softmax dimension bug in a scaled dot-product attention layer.

The buggy code applies softmax over dim=0 (the batch dimension) instead of
dim=-1 (the key/sequence dimension). This causes attention weights to sum
to 1.0 across the batch rather than across the keys, breaking the entire
attention mechanism.

Your task: fix the softmax call to normalise over the correct dimension.
"""

BUGGY_CODE = '''import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value):
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len, d_k).
        key:   Key tensor   of shape (batch, seq_len, d_k).
        value: Value tensor of shape (batch, seq_len, d_v).

    Returns:
        Context tensor of shape (batch, seq_len, d_v).
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    # Bug: dim=0 normalises across the batch, not across keys.
    weights = F.softmax(scores, dim=0)
    return torch.matmul(weights, value)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_k = 2, 3, 4
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)
    out = scaled_dot_product_attention(q, k, v)
    # Row sums should all be 1.0 if softmax is on dim=-1, but they\'re not.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=0)
    print(f"Output shape: {out.shape}")
    print(f"Row sums (SHOULD all be 1.0):\\n{weights.sum(dim=-1)}")
'''

CORRECT_CODE = '''import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value):
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len, d_k).
        key:   Key tensor   of shape (batch, seq_len, d_k).
        value: Value tensor of shape (batch, seq_len, d_v).

    Returns:
        Context tensor of shape (batch, seq_len, d_v).
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    # Fix: dim=-1 normalises across the key dimension (last axis).
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_k = 2, 3, 4
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)
    out = scaled_dot_product_attention(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    print(f"Output shape: {out.shape}")
    print(f"Row sums (all ~1.0):\\n{weights.sum(dim=-1)}")
'''
