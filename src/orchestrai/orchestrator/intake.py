"""
Task intake and classification.
Converts a raw user request into a typed TaskBrief artifact.
"""
from __future__ import annotations

import re

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind,
    Provenance,
    RepoSummary,
    TaskBrief,
    TaskType,
)
from orchestrai.observability.trace import make_artifact_id
from orchestrai.policies.paths import MAX_PACKAGE_METADATA_BYTES, PathPolicy, PathPolicyError

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
    r"\b(review|critique|assess|check|evaluate|audit|"
    r"look at|what do you think|feedback)\b",
    re.IGNORECASE,
)
_TEST_KEYWORDS = re.compile(
    r"\b(test|tests|unit test|integration test|coverage|spec|assert|"
    r"pytest|jest|vitest|write tests|add tests)\b",
    re.IGNORECASE,
)
_DOCS_KEYWORDS = re.compile(
    r"\b(document|doc|docs|documentation|readme|comment|docstring|explain|describe|"
    r"changelog|api doc|openapi)\b",
    re.IGNORECASE,
)
_RESEARCH_KEYWORDS = re.compile(
    r"\b(analyse|analyze|investigate|profile|benchmark|bottleneck|performance|"
    r"research|explore|study|understand|diagnose|why is|how does)\b",
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
    if _RESEARCH_KEYWORDS.search(request):
        scores[TaskType.RESEARCH] += 3

    best = max(scores, key=lambda t: scores[t])
    if scores[best] == 0:
        return TaskType.GENERAL
    return best


def scan_repo(root: str) -> RepoSummary:
    """Quick static scan of the repo to build context."""
    paths = PathPolicy.current()
    path = paths.repository(root)
    files = {entry.relative_to(path).as_posix() for entry in paths.iter_files(path)}

    language = "unknown"
    frameworks: list[str] = []
    test_framework: str | None = None
    lint_tools: list[str] = []
    key_files: list[str] = []
    total = len(files)

    # Language detection
    if "pyproject.toml" in files or any(name.endswith(".py") for name in files):
        language = "python"
        if "pyproject.toml" in files:
            key_files.append("pyproject.toml")
    elif "package.json" in files:
        language = "typescript/javascript"
        key_files.append("package.json")
        content = paths.read_bytes("package.json", path, max_bytes=MAX_PACKAGE_METADATA_BYTES)
        try:
            pkg = content.decode("utf-8")
        except UnicodeError:
            raise PathPolicyError from None
        if "react" in pkg.lower():
            frameworks.append("react")
        if "next" in pkg.lower():
            frameworks.append("nextjs")
        if "vitest" in pkg:
            test_framework = "vitest"
        elif "jest" in pkg:
            test_framework = "jest"
    elif "Cargo.toml" in files:
        language = "rust"
        key_files.append("Cargo.toml")
        test_framework = "cargo test"
    elif "go.mod" in files:
        language = "go"
        key_files.append("go.mod")
        test_framework = "go test"

    # Test framework for Python
    if language == "python" and ("pytest.ini" in files or "pyproject.toml" in files):
        test_framework = "pytest"

    # Lint tools
    if ".ruff.toml" in files or "pyproject.toml" in files:
        lint_tools.append("ruff")
    if ".eslintrc.json" in files or ".eslintrc.js" in files:
        lint_tools.append("eslint")

    # README
    for readme in ["README.md", "README.rst", "README"]:
        if readme in files:
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
    """Build descriptive artifacts only; filesystem admission belongs to submit.

    Standalone callers may supply target names without a repository. No file is
    read and an omitted repo_root remains None.
    """
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
