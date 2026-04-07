"""
Safety guardrails for task execution.
Validates requests before routing and post-results before surfacing.
"""
from __future__ import annotations

import re

import structlog

log = structlog.get_logger()

# Patterns that indicate potentially dangerous shell commands in diffs
_DANGEROUS_DIFF_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"dd\s+if=.*of=/dev/",
    r"mkfs\.",
    r":(){ :|:& };:",       # fork bomb
    r"chmod\s+-R\s+777\s+/",
    r">\s*/etc/passwd",
    r">\s*/etc/shadow",
    r"curl\s+.*\|\s*(?:bash|sh)",
    r"wget\s+.*\|\s*(?:bash|sh)",
]

_COMPILED = [re.compile(p) for p in _DANGEROUS_DIFF_PATTERNS]


def validate_diff(diff: str) -> list[str]:
    """
    Check a unified diff for dangerous patterns.
    Returns list of warnings (empty = clean).
    """
    warnings = []
    for pattern in _COMPILED:
        if pattern.search(diff):
            warnings.append(f"Dangerous pattern detected: {pattern.pattern}")
    return warnings


def validate_request(request: str) -> list[str]:
    """
    Light validation on the task request string.
    Returns list of warnings.
    """
    warnings = []
    # Warn if request contains paths to sensitive system files
    if re.search(r"/etc/(passwd|shadow|sudoers)", request):
        warnings.append("Request references sensitive system files")
    if len(request) > 10_000:
        warnings.append("Request is unusually long (>10k chars) — consider summarizing")
    return warnings


def log_safety_check(task_id: str, diff: str) -> None:
    """Run safety check and log any warnings."""
    warnings = validate_diff(diff)
    if warnings:
        log.warning("safety.diff_warnings", task_id=task_id, warnings=warnings)
