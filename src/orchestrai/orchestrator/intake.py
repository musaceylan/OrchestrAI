"""
Task intake and classification.
Converts a raw user request into a typed TaskBrief artifact.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, Provenance, RepoSummary, TaskBrief, TaskType,
)
from orchestrai.observability.trace import make_artifact_id

log = structlog.get_logger()

# Keyword patterns for task classification
_BUGFIX_KEYWORDS = re.compile(
    r"\b(bug|fix|broken|error|exception|crash|failing|regression|wrong output|"
    r"doesn.?t work|not working|issue|problem|TypeError|AttributeError)\b",
    re.IGNORECASE,
)
_FEATURE_KEYWORDS = re.compile(
    r"\b(add|implement|build|create|new feature|support|integrate|extend|"
    r"feature|functionality|endpoint|route|page|component|module)\b",
    re.IGNORECASE,
)
_REFACTOR_KEYWORDS = re.compile(
    r"\b(refactor|clean up|restructure|simplify|reorganize|split|extract|"
    r"rename|move|dedup|deduplicate|technical debt|improve structure)\b",
    re.IGNORECASE,
)
_REVIEW_KEYWORDS = re.compile(
    r"\b(review|critique|analyse|analyze|assess|check|evaluate|audit|"
    r"look at|what do you think|feedback)\b",
    re.IGNORECASE,
)
_TEST_KEYWORDS = re.compile(
    r"\b(test|tests|unit test|integration test|coverage|spec|assert|"
    r"pytest|jest|vitest|write tests|add tests)\b",
    re.IGNORECASE,
)
_DOCS_KEYWORDS = re.compile(
    r"\b(doc|docs|documentation|readme|comment|docstring|explain|describe|"
    r"changelog|api doc|openapi)\b",
    re.IGNORECASE,
)


def classify_task(request: str) -> TaskType:
    """Heuristically classify a task from the user's raw request."""
    scores: dict[TaskType, int] = {t: 0 for t in TaskType}

    if _BUGFIX_KEYWORDS.search(request):
        scores[TaskType.BUGFIX] += 3
    if _FEATURE_KEYWORDS.search(request):
        scores[TaskType.FEATURE] += 2
    if _REFACTOR_KEYWORDS.search(request):
        scores[TaskType.REFACTOR] += 3
    if _REVIEW_KEYWORDS.search(request):
        scores[TaskType.REVIEW] += 3
    if _TEST_KEYWORDS.search(request):
        scores[TaskType.TEST_GENERATION] += 3
    if _DOCS_KEYWORDS.search(request):
        scores[TaskType.DOCS] += 3

    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return TaskType.GENERAL
    return best


def scan_repo(root: str) -> RepoSummary:
    """Quick static scan of the repo to build context."""
    path = Path(root)
    if not path.exists():
        return RepoSummary(
            id=make_artifact_id(),
            kind=ArtifactKind.REPO_SUMMARY,
            provenance=Provenance(task_id=""),
            summary_text=f"Repo path not found: {root}",
        )

    language = "unknown"
    frameworks: list[str] = []
    test_framework: str | None = None
    lint_tools: list[str] = []
    key_files: list[str] = []
    total = 0

    # Language detection
    if (path / "pyproject.toml").exists() or any(path.glob("**/*.py")):
        language = "python"
        if (path / "pyproject.toml").exists():
            key_files.append("pyproject.toml")
    elif (path / "package.json").exists():
        language = "typescript/javascript"
        key_files.append("package.json")
        pkg = (path / "package.json").read_text(errors="ignore")
        if "react" in pkg.lower():
            frameworks.append("react")
        if "next" in pkg.lower():
            frameworks.append("nextjs")
        if "vitest" in pkg:
            test_framework = "vitest"
        elif "jest" in pkg:
            test_framework = "jest"
    elif (path / "Cargo.toml").exists():
        language = "rust"
        key_files.append("Cargo.toml")
        test_framework = "cargo test"
    elif (path / "go.mod").exists():
        language = "go"
        key_files.append("go.mod")
        test_framework = "go test"

    # Test framework for Python
    if language == "python":
        if (path / "pytest.ini").exists() or (path / "pyproject.toml").exists():
            test_framework = "pytest"

    # Lint tools
    if (path / ".ruff.toml").exists() or (path / "pyproject.toml").exists():
        lint_tools.append("ruff")
    if (path / ".eslintrc.json").exists() or (path / ".eslintrc.js").exists():
        lint_tools.append("eslint")

    # Rough file count (don't recurse infinitely)
    try:
        total = sum(1 for _ in path.rglob("*") if _.is_file() and ".git" not in str(_))
    except Exception:
        total = 0

    # README
    for readme in ["README.md", "README.rst", "README"]:
        if (path / readme).exists():
            key_files.append(readme)
            break

    summary = (
        f"Language: {language}, "
        f"Frameworks: {frameworks or 'none detected'}, "
        f"Test framework: {test_framework or 'unknown'}, "
        f"Lint: {lint_tools or 'none detected'}, "
        f"Files: {total}"
    )

    return RepoSummary(
        id=make_artifact_id(),
        kind=ArtifactKind.REPO_SUMMARY,
        provenance=Provenance(task_id=""),
        language=language,
        frameworks=frameworks,
        test_framework=test_framework,
        lint_tools=lint_tools,
        key_files=key_files,
        total_files=total,
        summary_text=summary,
    )


def build_task_brief(
    request: str,
    task_id: str,
    repo_root: str | None = None,
    target_files: list[str] | None = None,
    extra_context: str | None = None,
) -> TaskBrief:
    task_type = classify_task(request)
    context_snippets: list[str] = []
    if extra_context:
        context_snippets.append(extra_context)

    log.info("intake.classified", task_id=task_id, task_type=task_type.value)

    return TaskBrief(
        id=make_artifact_id(),
        kind=ArtifactKind.TASK_BRIEF,
        provenance=Provenance(task_id=task_id),
        task_type=task_type,
        description=request,
        repo_root=repo_root,
        target_files=target_files or [],
        context_snippets=context_snippets,
        raw_request=request,
    )
