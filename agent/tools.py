"""Agent tools for ZeroTrace — offline doc search and code snippet runner.

These tools extend the agent's capabilities without requiring any network
calls. The doc index is a curated, bundled dict of common Python / PyTorch
patterns relevant to the benchmark tasks.
"""

from typing import Any, Dict, List
from environment.sandbox import run_code_safely

# ---------------------------------------------------------------------------
# Offline documentation index
# ---------------------------------------------------------------------------
_DOCS: Dict[str, str] = {
    "crossentropyloss": (
        "torch.nn.CrossEntropyLoss — expected dtypes\n"
        "  input  : (N, C) FloatTensor (raw logits, NOT softmaxed)\n"
        "  target : (N,)   LongTensor  (class indices, dtype=torch.long / int64)\n"
        "Common bug: passing float32 targets → RuntimeError.\n"
        "Fix: torch.tensor(labels, dtype=torch.long)"
    ),
    "log": (
        "torch.log(x) — numerical stability\n"
        "  log(0) = -inf  →  NaN gradients during backprop.\n"
        "Fix options:\n"
        "  1. x.clamp(min=1e-8)     → torch.log(x.clamp(min=1e-8))\n"
        "  2. torch.clamp(x, 1e-8)  → same\n"
        "  3. torch.log(x + 1e-8)   → less precise but common"
    ),
    "softmax": (
        "torch.nn.functional.softmax(input, dim)\n"
        "  dim : which axis is normalised (values sum to 1 along that axis).\n"
        "  For attention weights of shape (batch, seq, keys): use dim=-1.\n"
        "  dim=0 → normalises across the batch dimension (WRONG for attention).\n"
        "Fix: F.softmax(scores, dim=-1)"
    ),
    "keyerror": (
        "Python dict — safe key access\n"
        "  dict[key]           → KeyError if key missing.\n"
        "  dict.get(key, default) → returns default (None by default) safely.\n"
        "  'key' in dict       → boolean membership check.\n"
        "Fix: use dict.get(user_id, {}).get('score', 0)"
    ),
    "contextmanager": (
        "Python file handling — context manager\n"
        "  Bad:  f = open(path); data = f.read(); f.close()  ← leaks on exception\n"
        "  Good: with open(path) as f: data = f.read()       ← always closes\n"
        "The 'with' statement guarantees __exit__ is called even on exceptions."
    ),
    "threading": (
        "Python threading — race condition fix\n"
        "  Race condition: multiple threads read-modify-write a shared variable\n"
        "  without synchronisation → non-deterministic results.\n"
        "Fix:\n"
        "  import threading\n"
        "  lock = threading.Lock()\n"
        "  def increment():\n"
        "      global counter\n"
        "      for _ in range(10000):\n"
        "          with lock:\n"
        "              counter += 1"
    ),
    "gradients": (
        "PyTorch gradient basics\n"
        "  model.zero_grad()   → clear accumulated gradients before backward.\n"
        "  loss.backward()     → compute gradients.\n"
        "  optimizer.step()    → update parameters.\n"
        "  NaN gradients are caused by inf/nan in the forward pass (e.g. log(0))."
    ),
    "dtype": (
        "PyTorch tensor dtypes\n"
        "  torch.float32 / torch.float  → default for model weights, inputs.\n"
        "  torch.long    / torch.int64  → required for class indices in losses.\n"
        "  torch.bool                   → masks.\n"
        "Cast: tensor.float(), tensor.long(), tensor.to(dtype=torch.long)"
    ),
}


def search_docs(query: str, top_k: int = 3) -> str:
    """Search the offline documentation index.

    Uses simple keyword matching — no network calls.

    Args:
        query: Free-text search query (e.g. "torch log nan").
        top_k: Maximum number of results to return.

    Returns:
        Formatted string with matching documentation snippets.
    """
    query_lower = query.lower()
    scores: List[tuple] = []

    for key, doc in _DOCS.items():
        hits = sum(
            1 for word in query_lower.split()
            if word in key or word in doc.lower()
        )
        if hits > 0:
            scores.append((hits, key, doc))

    scores.sort(key=lambda x: -x[0])
    top = scores[:top_k]

    if not top:
        return (
            f"No documentation found for query: '{query}'.\n"
            "Try keywords: crossentropyloss, log, softmax, keyerror, "
            "contextmanager, threading, gradients, dtype"
        )

    parts = []
    for _, key, doc in top:
        parts.append(f"[{key.upper()}]\n{doc}")

    return "\n\n".join(parts)


def run_code_snippet(code: str, timeout: int = 5) -> Dict[str, Any]:
    """Run a short code snippet in the sandbox and return stdout/stderr.

    This is a restricted execution — designed for quick verification
    snippets, NOT for running the full patched code (use EXECUTE_UNIT_TEST
    for that).

    Args:
        code:    Python code string to execute.
        timeout: Maximum execution time in seconds (capped at 5).

    Returns:
        Dict with keys: stdout, stderr, success, timed_out.
    """
    safe_timeout = min(int(timeout), 5)
    return run_code_safely(code, timeout=safe_timeout)
