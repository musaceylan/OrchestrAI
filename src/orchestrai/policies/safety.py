"""
Safety guardrails for task execution.
Validates requests before routing and post-results before surfacing.
"""
from __future__ import annotations

import re

import structlog

from orchestrai.observability.metrics import safety_violations_total

log = structlog.get_logger()


class SafetyViolationError(Exception):
    """Raised when a diff contains a dangerous pattern."""

    def __init__(self, patterns: list[str]) -> None:
        super().__init__(f"Safety violation — dangerous patterns detected: {patterns}")
        self.patterns = patterns


# Patterns that indicate potentially dangerous shell commands in diffs
_DANGEROUS_DIFF_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"dd\s+if=.*of=/dev/",
    r"mkfs\.",
    r":(){ :|:& };:",                          # fork bomb
    r"chmod\s+-R\s+777\s+/",
    r">\s*/etc/passwd",
    r">\s*/etc/shadow",
    r"curl\s+.*\|\s*(?:bash|sh)",
    r"wget\s+.*\|\s*(?:bash|sh)",
    r"sudo\s+(?:rm|chmod|chown|dd|mkfs|fdisk)",  # privileged destructive ops
    r"base64\s+-d\s*.*\|\s*(?:bash|sh)",          # encoded payload execution
    r"eval\s*\$\(.*\)",                            # subshell eval
    r">\s*/etc/crontab",                           # crontab poisoning
    r"ssh-keygen.*>>.*authorized_keys",            # SSH backdoor
]

_COMPILED = [re.compile(p) for p in _DANGEROUS_DIFF_PATTERNS]

# PII / secret patterns: (pattern, replacement_label)
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[OPENAI_KEY]"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "[GH_PAT]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[AWS_KEY]"),
    (re.compile(r"(?i)api[_\-]?key\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{20,}[\"']?"), "[API_KEY]"),
]


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


def enforce_diff_safety(diff: str) -> None:
    """
    Check a unified diff for dangerous patterns and raise if any are found.

    Raises SafetyViolationError listing all matched patterns.
    """
    violations = validate_diff(diff)
    if violations:
        log.error("safety.diff_violation", violations=violations)
        for violation in violations:
            safety_violations_total.labels(kind="diff").inc()
        raise SafetyViolationError(violations)


def mask_pii(text: str) -> str:
    """Replace known PII / secret patterns with safe placeholders."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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
        warnings.append("Request is unusually long (>10k chars) — consider summarising")
    return warnings


def log_safety_check(task_id: str, diff: str) -> None:
    """Run safety check and log any warnings."""
    warnings = validate_diff(diff)
    if warnings:
        log.warning("safety.diff_warnings", task_id=task_id, warnings=warnings)
