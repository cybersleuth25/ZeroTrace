"""Static code safety scanner for ZeroTrace.

Checks agent-submitted patched_code for dangerous patterns before it
is executed in the sandbox. This is defence-in-depth: the sandbox
already has subprocess isolation and timeouts, but preventing known-bad
patterns from ever running is better.

No external dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ScanResult:
    """Result of a code safety scan."""

    safe: bool
    violations: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.safe:
            return "Code passed all safety checks."
        return "; ".join(self.violations)


# ---------------------------------------------------------------------------
# Patterns deemed dangerous in agent-submitted patches.
# Each tuple is (regex_pattern, human_readable_description).
# ---------------------------------------------------------------------------
_DANGEROUS: List[Tuple[str, str]] = [
    # Shell execution
    (r"\bos\.system\s*\(", "os.system() — direct shell execution"),
    (r"\bsubprocess\.(run|call|check_output|check_call|Popen)\s*\(", "subprocess execution"),
    # Dynamic evaluation
    (r"(?<!\w)eval\s*\(", "eval() — arbitrary code evaluation"),
    (r"(?<!\w)exec\s*\(", "exec() — arbitrary code execution"),
    # Dynamic import tricks  (legitimate __import__ occurs in test harness, NOT in patches)
    (r"__import__\s*\(", "__import__() — dynamic import"),
    # Sensitive file paths
    (r"""open\s*\(\s*['"/](?:etc|proc|sys)/""", "open() on sensitive system path"),
    # Network
    (r"\bsocket\.socket\s*\(", "socket.socket — raw network access"),
    (r"\burllib\.request\.(urlopen|urlretrieve)\s*\(", "urllib network request"),
    # Destructive file operations
    (r"\bshutil\.rmtree\s*\(", "shutil.rmtree — recursive directory deletion"),
    (r"\bos\.remove\s*\(", "os.remove — file deletion"),
    (r"\bos\.unlink\s*\(", "os.unlink — file deletion"),
    # Code object tricks
    (r"\bcompile\s*\(.*exec", "compile()+exec — indirect code execution"),
    (r"\b__builtins__\b", "__builtins__ manipulation"),
    (r"\bctypes\b", "ctypes — low-level C interop"),
]

# Pre-compile all patterns for speed
_COMPILED = [(re.compile(pat, re.IGNORECASE), desc) for pat, desc in _DANGEROUS]

# Maximum code length we're willing to scan / execute
MAX_CODE_LENGTH = 50_000


def scan_code(code: str) -> ScanResult:
    """Scan a code string for dangerous patterns.

    Args:
        code: The Python source code to check.

    Returns:
        ScanResult with safe=True if no violations were found.
    """
    if not isinstance(code, str):
        return ScanResult(safe=False, violations=["patched_code is not a string"])

    if len(code) > MAX_CODE_LENGTH:
        return ScanResult(
            safe=False,
            violations=[
                f"Code exceeds maximum allowed length "
                f"({len(code):,} > {MAX_CODE_LENGTH:,} chars)"
            ],
        )

    violations: List[str] = []
    for pattern, description in _COMPILED:
        if pattern.search(code):
            violations.append(description)

    return ScanResult(safe=len(violations) == 0, violations=violations)
