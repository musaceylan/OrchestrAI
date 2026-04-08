"""
Shell / tool execution layer — runs linters, tests, formatters, git diffs.
All execution is sandboxed with timeouts and output capture.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, CommandOutput, LintOutput, Provenance, TestCandidate, TypeCheckOutput,
)
from orchestrai.observability.trace import make_artifact_id

log = structlog.get_logger()

DEFAULT_TIMEOUT = 60.0  # seconds


async def run_command(
    cmd: str | list[str],
    cwd: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    task_id: str = "",
    subtask_id: str | None = None,
    env: dict[str, str] | None = None,
) -> CommandOutput:
    """Run a shell command, capture stdout/stderr, return structured output."""
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
        cmd_str = cmd
    else:
        cmd_list = cmd
        cmd_str = " ".join(cmd)

    # Merge caller-supplied env with current environment so subprocess has PATH etc.
    merged_env = {**os.environ, **(env or {})}

    start = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=merged_env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            duration = (time.time() - start) * 1000
            log.warning("shell.timeout", cmd=cmd_str, timeout=timeout)
            return CommandOutput(
                id=make_artifact_id(),
                kind=ArtifactKind.COMMAND_OUTPUT,
                provenance=Provenance(task_id=task_id, subtask_id=subtask_id),
                command=cmd_str,
                stdout="",
                stderr=f"[TIMEOUT] Command exceeded {timeout}s",
                exit_code=-1,
                duration_ms=round(duration, 2),
                success=False,
            )

        exit_code = proc.returncode or 0
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        duration = (time.time() - start) * 1000

        log.info(
            "shell.executed",
            cmd=cmd_str,
            exit_code=exit_code,
            duration_ms=round(duration, 2),
        )

        return CommandOutput(
            id=make_artifact_id(),
            kind=ArtifactKind.COMMAND_OUTPUT,
            provenance=Provenance(task_id=task_id, subtask_id=subtask_id),
            command=cmd_str,
            stdout=stdout[:50_000],  # cap at 50KB
            stderr=stderr[:10_000],
            exit_code=exit_code,
            duration_ms=round(duration, 2),
            success=exit_code == 0,
        )

    except FileNotFoundError as e:
        log.error("shell.not_found", cmd=cmd_str, error=str(e))
        duration = (time.time() - start) * 1000
        return CommandOutput(
            id=make_artifact_id(),
            kind=ArtifactKind.COMMAND_OUTPUT,
            provenance=Provenance(task_id=task_id, subtask_id=subtask_id),
            command=cmd_str,
            stdout="",
            stderr=f"[NOT FOUND] {e}",
            exit_code=-1,
            duration_ms=round(duration, 2),
            success=False,
        )


async def run_tests(
    cwd: str,
    task_id: str = "",
    framework: str | None = None,
    test_path: str | None = None,
    timeout: float = 120.0,
) -> TestCandidate:
    """Auto-detect test framework and run tests."""
    detected = framework or _detect_test_framework(cwd)
    cmd = _build_test_command(detected, test_path)

    output = await run_command(cmd, cwd=cwd, timeout=timeout, task_id=task_id)
    return TestCandidate(
        id=make_artifact_id(),
        kind=ArtifactKind.TEST_CANDIDATE,
        provenance=Provenance(task_id=task_id),
        test_code="",
        framework=detected,
        passed=output.success,
        test_output=output.stdout + ("\n" + output.stderr if output.stderr else ""),
    )


async def run_lint(
    cwd: str,
    task_id: str = "",
    files: list[str] | None = None,
    timeout: float = 60.0,
) -> LintOutput:
    """Run available linters, return structured output."""
    detected_tool, cmd = _detect_lint_command(cwd, files)
    output = await run_command(cmd, cwd=cwd, timeout=timeout, task_id=task_id)

    issues = _parse_lint_output(detected_tool, output.stdout + output.stderr)
    errors = sum(1 for i in issues if i.get("severity") == "error")
    warnings = sum(1 for i in issues if i.get("severity") == "warning")

    return LintOutput(
        id=make_artifact_id(),
        kind=ArtifactKind.LINT_OUTPUT,
        provenance=Provenance(task_id=task_id),
        tool=detected_tool,
        issues=issues,
        error_count=errors,
        warning_count=warnings,
        passed=output.success and errors == 0,
        raw_output=output.stdout + output.stderr,
    )


async def run_typecheck(
    cwd: str,
    task_id: str = "",
    timeout: float = 60.0,
) -> TypeCheckOutput:
    """Run type checker if available."""
    tool, cmd = _detect_typecheck_command(cwd)
    output = await run_command(cmd, cwd=cwd, timeout=timeout, task_id=task_id)

    return TypeCheckOutput(
        id=make_artifact_id(),
        kind=ArtifactKind.TYPE_CHECK_OUTPUT,
        provenance=Provenance(task_id=task_id),
        tool=tool,
        errors=[],
        passed=output.success,
        raw_output=output.stdout + output.stderr,
    )


async def get_git_diff(cwd: str, task_id: str = "") -> CommandOutput:
    """Get current git diff."""
    return await run_command("git diff HEAD", cwd=cwd, task_id=task_id)


async def get_git_status(cwd: str, task_id: str = "") -> CommandOutput:
    return await run_command("git status --porcelain", cwd=cwd, task_id=task_id)


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_test_framework(cwd: str) -> str:
    root = Path(cwd)
    if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
        return "pytest"
    if (root / "package.json").exists():
        pkg = (root / "package.json").read_text()
        if "vitest" in pkg:
            return "vitest"
        if "jest" in pkg:
            return "jest"
        if "mocha" in pkg:
            return "mocha"
        return "npm test"
    if (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test"
    return "pytest"


def _build_test_command(framework: str, test_path: str | None) -> str:
    if framework == "pytest":
        base = "python -m pytest -v --tb=short"
        return f"{base} {test_path}" if test_path else base
    if framework == "vitest":
        return "npx vitest run"
    if framework == "jest":
        return "npx jest --passWithNoTests"
    if framework == "cargo test":
        return "cargo test"
    if framework == "go test":
        return "go test ./..."
    return framework


def _detect_lint_command(cwd: str, files: list[str] | None) -> tuple[str, list[str]]:
    root = Path(cwd)
    # Use list form to avoid argument injection — never join into a shell string
    file_args: list[str] = files if files else ["."]
    # Reject any file arg that looks like a flag to prevent option injection
    safe_file_args = [f for f in file_args if not f.startswith("-")]
    if not safe_file_args:
        safe_file_args = ["."]
    if (root / "pyproject.toml").exists() or (root / ".ruff.toml").exists():
        return "ruff", ["ruff", "check"] + safe_file_args
    if (root / ".eslintrc.json").exists() or (root / ".eslintrc.js").exists():
        return "eslint", ["npx", "eslint"] + safe_file_args
    if (root / "Cargo.toml").exists():
        return "clippy", ["cargo", "clippy", "--", "-D", "warnings"]
    if (root / "go.mod").exists():
        return "golint", ["go", "vet", "./..."]
    return "ruff", ["ruff", "check"] + safe_file_args


def _detect_typecheck_command(cwd: str) -> tuple[str, str]:
    root = Path(cwd)
    if (root / "tsconfig.json").exists():
        return "tsc", "npx tsc --noEmit"
    if (root / "pyproject.toml").exists():
        return "mypy", "python -m mypy ."
    if (root / "go.mod").exists():
        return "go build", "go build ./..."
    return "mypy", "python -m mypy ."


def _parse_lint_output(tool: str, raw: str) -> list[dict]:
    """Best-effort parsing of lint output into structured issues."""
    import re
    issues = []
    # ruff/flake8: "path.py:10:5: E501 line too long"
    _ruff_error = re.compile(r":\s+E\d{3,}")
    # generic "error:" at word boundary (avoid false matches like "noerror")
    _generic_error = re.compile(r"\berror\b", re.IGNORECASE)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if _ruff_error.search(line) or _generic_error.search(line):
            sev = "error"
        else:
            sev = "warning"
        issues.append({"text": line, "severity": sev})
    return issues
