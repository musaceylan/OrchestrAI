"""
Built-in MCP prompts for OrchestrAI.
Each prompt is a reusable workflow template for common software engineering tasks.
"""
from __future__ import annotations

from typing import Any

BUILTIN_PROMPTS = [
    {
        "name": "bugfix",
        "description": "Orchestrate a multi-model bugfix: analyse → plan → patch → test → review",
        "arguments": [
            {"name": "description", "description": "Bug description and reproduction steps", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "error_message", "description": "Error message or stack trace", "required": False},
            {"name": "target_files", "description": "Comma-separated list of suspect files", "required": False},
        ],
    },
    {
        "name": "feature",
        "description": "Orchestrate a new feature: plan → implement → test → review",
        "arguments": [
            {"name": "description", "description": "Feature description and acceptance criteria", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Comma-separated files to modify", "required": False},
        ],
    },
    {
        "name": "refactor",
        "description": "Orchestrate a refactor: analyse → plan → change → verify tests still pass",
        "arguments": [
            {"name": "description", "description": "What to refactor and why", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Comma-separated files to refactor", "required": False},
        ],
    },
    {
        "name": "review",
        "description": "Run an independent multi-model code review on a diff or file",
        "arguments": [
            {"name": "description", "description": "What is being reviewed and review goals", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Files to review", "required": False},
        ],
    },
    {
        "name": "test_generation",
        "description": "Generate comprehensive tests for existing code using TDD-style workflow",
        "arguments": [
            {"name": "description", "description": "What to test and desired coverage goals", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Files to generate tests for", "required": False},
            {"name": "framework", "description": "Test framework (pytest, jest, vitest, etc.)", "required": False},
        ],
    },
    {
        "name": "docs",
        "description": "Generate or improve documentation for code",
        "arguments": [
            {"name": "description", "description": "What to document and target audience", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Files to document", "required": False},
            {"name": "doc_format", "description": "Documentation format (markdown, rst, docstring)", "required": False},
        ],
    },
    {
        "name": "local_only",
        "description": "Run any task using only local models (privacy-preserving, no cloud API calls)",
        "arguments": [
            {"name": "description", "description": "Task description", "required": True},
            {"name": "repo_root", "description": "Path to repository root", "required": False},
            {"name": "target_files", "description": "Comma-separated files", "required": False},
        ],
    },
]

_PROMPT_TEMPLATE = """\
You are working with OrchestrAI, a multi-model orchestration system.

Task: {description}
{extra}

Use the `submit_task` tool to execute this task with the following parameters:
- request: "{description}"
{submit_params}

After submission, use `get_task_result` with the returned task_id to retrieve the result.

If you want to inspect progress or artifacts, use:
- `inspect_agents` — see which models are assigned
- `inspect_artifacts` — browse generated patches and tests
- `inspect_trace` — full execution timeline
"""


def render_prompt(name: str, arguments: dict[str, str]) -> dict[str, Any]:
    """Render a built-in prompt with provided arguments."""
    prompt_def = next((p for p in BUILTIN_PROMPTS if p["name"] == name), None)
    if prompt_def is None:
        return {
            "description": f"Unknown prompt: {name}",
            "content": f"Error: prompt '{name}' not found.",
        }

    description = arguments.get("description", "")
    repo_root = arguments.get("repo_root", "")
    target_files = arguments.get("target_files", "")

    extra_lines = []
    submit_params = []

    if repo_root:
        extra_lines.append(f"Repository: {repo_root}")
        submit_params.append(f'- repo_root: "{repo_root}"')
    if target_files:
        files = [f.strip() for f in target_files.split(",") if f.strip()]
        extra_lines.append(f"Target files: {', '.join(files)}")
        submit_params.append(f"- target_files: {files}")

    # Mode-specific parameters
    mode_map = {
        "bugfix": "bugfix",
        "feature": "planner_coder_reviewer",
        "refactor": "refactor",
        "review": "impl_tester",
        "test_generation": "impl_tester",
        "docs": "docs",
        "local_only": None,  # no mode override, but add privacy preference
    }
    mode = mode_map.get(name)
    if mode:
        submit_params.append(f'- mode: "{mode}"')

    # local_only adds user_preferences
    if name == "local_only":
        submit_params.append('- user_preferences: {"privacy_level": "secret"}')

    # test_generation adds framework hint
    if name == "test_generation":
        framework = arguments.get("framework", "pytest")
        extra_lines.append(f"Test framework: {framework}")
        submit_params.append(f'- user_preferences: {{"test_framework": "{framework}"}}')

    # error message for bugfix
    if name == "bugfix" and arguments.get("error_message"):
        extra_lines.append(f"Error: {arguments['error_message']}")
        description = f"{description}\n\nError/stacktrace:\n{arguments['error_message']}"

    # doc format
    if name == "docs" and arguments.get("doc_format"):
        extra_lines.append(f"Documentation format: {arguments['doc_format']}")

    extra = "\n".join(extra_lines)
    params_str = "\n".join(submit_params)

    content = _PROMPT_TEMPLATE.format(
        description=description,
        extra=extra,
        submit_params=params_str,
    )

    return {
        "description": prompt_def["description"],
        "content": content,
    }
