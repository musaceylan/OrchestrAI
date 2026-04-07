"""Unit tests for task intake and classification."""
from __future__ import annotations

import pytest

from orchestrai.artifacts.schemas import TaskType
from orchestrai.orchestrator.intake import classify_task, build_task_brief


@pytest.mark.parametrize("request,expected", [
    ("Fix the NullPointerException in auth.py", TaskType.BUGFIX),
    ("Add pagination to the users API endpoint", TaskType.FEATURE),
    ("Refactor the database layer to use repository pattern", TaskType.REFACTOR),
    ("Write unit tests for the payment service", TaskType.TEST_GENERATION),
    ("Review this pull request for security issues", TaskType.REVIEW),
    ("Document the public API surface", TaskType.DOCS),
    ("Analyse performance bottlenecks in the query pipeline", TaskType.RESEARCH),
])
def test_classify_task(request, expected):
    result = classify_task(request)
    assert result == expected, f"Expected {expected} for '{request}', got {result}"


def test_build_task_brief_basic():
    brief = build_task_brief(
        request="Fix the login bug",
        task_id="task-123",
    )
    assert brief.id == "task-123"
    assert brief.description == "Fix the login bug"
    assert brief.task_type == TaskType.BUGFIX
    assert brief.repo_root is None
    assert brief.target_files == []


def test_build_task_brief_with_files():
    brief = build_task_brief(
        request="Add tests",
        task_id="task-456",
        target_files=["src/auth.py", "tests/test_auth.py"],
    )
    assert "src/auth.py" in brief.target_files
    assert brief.task_type == TaskType.TEST_GENERATION


def test_build_task_brief_unknown_falls_back():
    brief = build_task_brief(
        request="Do something vague with the system",
        task_id="task-789",
    )
    # Should not raise; falls back to a default type
    assert brief.task_type in list(TaskType)
