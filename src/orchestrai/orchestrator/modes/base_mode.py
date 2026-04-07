"""Base class shared by all orchestration modes."""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, CodePatch, CompletionRequest as _CompReq, OrchestratedTask,
    Provenance, ReviewComments, RoutingDecision, RoleType, SubTask, TestCandidate,
)
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.observability.trace import Tracer, make_agent_run_id, make_artifact_id, make_subtask_id
from orchestrai.providers.base import BaseProvider, CompletionRequest, ProviderError
from orchestrai.registry.registry import CapabilityRegistry

log = structlog.get_logger()

# System prompts tuned per role
ROLE_SYSTEM_PROMPTS: dict[RoleType, str] = {
    RoleType.PLANNER: (
        "You are an expert software architect and technical lead. "
        "Your job is to analyse the task and create a clear, actionable implementation plan. "
        "Be concrete: list files to change, data structures to modify, APIs to add. "
        "Output valid JSON with keys: steps (list), risks (list), estimated_complexity (str)."
    ),
    RoleType.CODER: (
        "You are a world-class software engineer. "
        "Implement the requested change with production-quality code. "
        "Output a unified diff (diff -u format) preceded by a brief description. "
        "Make the smallest correct change. Prefer existing patterns in the codebase."
    ),
    RoleType.TESTER: (
        "You are a senior QA engineer and testing expert. "
        "Write comprehensive tests for the described change. "
        "Cover happy paths, edge cases, and failure modes. "
        "Output runnable test code in the appropriate framework."
    ),
    RoleType.REVIEWER: (
        "You are a senior code reviewer. Be thorough but constructive. "
        "Identify bugs, security issues, performance problems, and style violations. "
        "Output JSON: {overall_verdict, comments: [{file, line, severity, message}], key_concerns, praise}."
    ),
    RoleType.DEBUGGER: (
        "You are an expert debugger. "
        "Analyse the error, find the root cause, and propose a minimal, correct fix. "
        "Show your reasoning step by step."
    ),
    RoleType.ANALYZER: (
        "You are a code analysis expert. "
        "Deeply analyse the provided code/repo for the specified concern. "
        "Be specific about what you find — file paths, line numbers, patterns."
    ),
    RoleType.REFACTOR: (
        "You are a refactoring expert. "
        "Propose and implement a refactor that improves structure without changing behaviour. "
        "Output a unified diff and explain the rationale."
    ),
    RoleType.DOCUMENTER: (
        "You are a technical writer specialising in developer documentation. "
        "Write clear, accurate, and helpful documentation for the described component or change."
    ),
    RoleType.JUDGE: (
        "You are a senior engineering lead acting as judge. "
        "Compare the candidate solutions and select the best one. "
        "Consider: correctness, safety, simplicity, test coverage, and maintainability. "
        "Output JSON: {winner_index, rationale, scores: {index: float}, concerns}."
    ),
    RoleType.RESEARCHER: (
        "You are a deep code researcher. "
        "Read and analyse provided context thoroughly. "
        "Summarise key findings relevant to the task."
    ),
}


class BaseMode(ABC):
    def __init__(
        self,
        registry: CapabilityRegistry,
        store: ArtifactStore,
        tracer: Tracer,
    ) -> None:
        self._registry = registry
        self._store = store
        self._tracer = tracer

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def run(self, task: OrchestratedTask) -> OrchestratedTask:
        ...

    async def _call_agent(
        self,
        task: OrchestratedTask,
        role: RoleType,
        provider_name: str,
        model_id: str,
        user_prompt: str,
        extra_context: str = "",
    ) -> tuple[str, SubTask]:
        """
        Call a single agent (provider/model) for a given role.
        Returns (content, subtask).
        """
        subtask_id = make_subtask_id()
        agent_run_id = make_agent_run_id()
        system = ROLE_SYSTEM_PROMPTS.get(role, "You are a helpful software engineering assistant.")

        subtask = SubTask(
            id=subtask_id,
            role=role,
            description=user_prompt[:200],
            provider=provider_name,
            model=model_id,
            status="running",
            started_at=time.time(),
        )
        task.subtasks.append(subtask)

        self._tracer.subtask_started(subtask_id, role.value, provider_name, model_id)

        provider = self._registry.get_provider(provider_name)
        if provider is None:
            subtask.status = "failed"
            subtask.error = f"Provider '{provider_name}' not found in registry"
            subtask.finished_at = time.time()
            self._tracer.subtask_finished(
                subtask_id, role.value, success=False,
                duration_ms=(subtask.finished_at - subtask.started_at) * 1000,
                error=subtask.error,
            )
            return "", subtask

        messages = [{"role": "user", "content": user_prompt}]
        if extra_context:
            messages.insert(0, {"role": "user", "content": f"Context:\n{extra_context}"})

        start = time.time()
        try:
            request = CompletionRequest(
                messages=messages,
                system=system,
                model=model_id,
                max_tokens=4096,
                temperature=0.2,
                role=role,
                task_id=task.id,
                subtask_id=subtask_id,
                agent_run_id=agent_run_id,
            )
            response = await provider.complete(request)
            duration_ms = (time.time() - start) * 1000

            subtask.status = "done"
            subtask.finished_at = time.time()
            self._tracer.subtask_finished(
                subtask_id, role.value, success=True,
                duration_ms=duration_ms,
                tokens={"input": response.input_tokens, "output": response.output_tokens},
            )
            return response.content, subtask

        except ProviderError as e:
            duration_ms = (time.time() - start) * 1000
            subtask.status = "failed"
            subtask.error = str(e)
            subtask.finished_at = time.time()
            self._tracer.subtask_finished(
                subtask_id, role.value, success=False,
                duration_ms=duration_ms, error=str(e),
            )
            log.error(
                "agent.call_failed",
                role=role.value,
                provider=provider_name,
                model=model_id,
                error=str(e),
            )
            return "", subtask

    def _find_assignment(
        self, routing: RoutingDecision, role: RoleType
    ) -> dict[str, Any] | None:
        for a in routing.assignments:
            if a["role_type"] == role.value:
                return a
        return None

    def _find_all_assignments(
        self, routing: RoutingDecision, role: RoleType
    ) -> list[dict[str, Any]]:
        return [a for a in routing.assignments if a["role_type"] == role.value]

    def _extract_diff(self, content: str) -> str:
        """Extract unified diff from model output."""
        # Look for ```diff blocks first
        import re
        diff_block = re.search(r"```diff\n(.*?)```", content, re.DOTALL)
        if diff_block:
            return diff_block.group(1)
        # Fall back to raw diff heuristic
        lines = content.split("\n")
        diff_lines = [l for l in lines if l.startswith(("---", "+++", "@@", "+", "-", " "))]
        if len(diff_lines) > 3:
            return "\n".join(diff_lines)
        return content  # return full content if no diff found

    def _make_patch(
        self,
        content: str,
        provenance: Provenance,
        confidence: float = 0.7,
    ) -> CodePatch:
        diff = self._extract_diff(content)
        return CodePatch(
            id=make_artifact_id(),
            kind=ArtifactKind.CODE_PATCH,
            provenance=provenance,
            unified_diff=diff,
            description=content[:500] if len(content) > 500 else content,
            confidence=confidence,
        )
