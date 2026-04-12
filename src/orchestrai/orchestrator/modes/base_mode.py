"""Base class shared by all orchestration modes."""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, CodePatch, FinalDecision, OrchestratedTask,
    Provenance, ReviewComments, RoutingDecision, RoleType, SubTask, TestCandidate,
)
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.config.settings import get_settings
from orchestrai.observability.trace import Tracer, make_agent_run_id, make_artifact_id, make_subtask_id
from orchestrai.observability.metrics import agent_calls_total, cost_usd_total
from orchestrai.policies.costs import estimate_cost
from orchestrai.policies.safety import enforce_diff_safety, mask_pii
from orchestrai.providers.base import BaseProvider, CompletionRequest, ProviderError
from orchestrai.registry.registry import CapabilityRegistry

log = structlog.get_logger()

def _extract_json(text: str) -> dict | None:
    """
    Extract the first syntactically complete JSON object from model output.

    Unlike a greedy regex (which grabs first-{ to last-}), this walks the
    string character-by-character tracking brace depth, stopping exactly when
    the outermost object closes. That handles:
      - explanatory text before the JSON block
      - trailing commentary after the closing brace
      - nested objects / arrays inside the JSON
    """
    import json
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


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

    def _build_base_context(self, brief: Any) -> str:
        """Standard context block shared across all modes."""
        return (
            f"Task: {brief.description}\n"
            f"Task type: {brief.task_type.value}\n"
            f"Repo: {brief.repo_root or 'unspecified'}\n"
            f"Target files: {', '.join(brief.target_files) or 'none'}\n"
            f"Context: {chr(10).join(brief.context_snippets)}"
        )

    def _finalize(
        self,
        task: "OrchestratedTask",
        final: "FinalDecision",
    ) -> None:
        """Store the final artifact and mark task done."""
        self._store.put(final)
        task.final = final
        task.status = "done"

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
        _max_retries: int = 3,
        _retry_base_delay: float = 1.0,
    ) -> tuple[str, SubTask]:
        """
        Call a single agent (provider/model) for a given role.

        Returns (content, subtask).  Retries up to _max_retries times on
        transient ProviderErrors (retryable=True) with exponential backoff.
        Also accumulates token counts and cost onto task.tokens_used /
        task.cost_usd, and enforces policy.max_cost_usd before each call.
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

        # ── Pre-flight budget check ───────────────────────────────────────
        settings = get_settings()
        if (
            settings.policy.max_cost_usd is not None
            and task.cost_usd >= settings.policy.max_cost_usd
        ):
            msg = (
                f"Budget ${settings.policy.max_cost_usd:.4f} USD already reached "
                f"(spent ${task.cost_usd:.4f}); skipping {role.value} call"
            )
            log.warning("agent.budget_preflight_skip", task_id=task.id, role=role.value,
                        cost_usd=task.cost_usd, max_cost_usd=settings.policy.max_cost_usd)
            subtask.status = "failed"
            subtask.error = msg
            subtask.finished_at = time.time()
            self._tracer.subtask_finished(
                subtask_id, role.value, success=False,
                duration_ms=0.0, error=msg,
            )
            return "", subtask

        # Merge extra_context into a single user message to avoid consecutive
        # user turn errors (OpenAI rejects [user, user] message sequences).
        full_prompt = (
            f"<context>\n{extra_context}\n</context>\n\n{user_prompt}"
            if extra_context
            else user_prompt
        )
        messages = [{"role": "user", "content": full_prompt}]

        # Respect the model's published output limit; fall back to 4096.
        cap = self._registry.get_capability(provider_name, model_id)
        max_tokens = min(cap.max_output_tokens, 8192) if cap else 4096

        start = time.time()
        last_error: ProviderError | None = None

        for attempt in range(_max_retries):
            try:
                request = CompletionRequest(
                    messages=messages,
                    system=system,
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    role=role,
                    task_id=task.id,
                    subtask_id=subtask_id,
                    agent_run_id=agent_run_id,
                )
                response = await provider.complete(request)
                duration_ms = (time.time() - start) * 1000

                # ── Accumulate tokens and cost ────────────────────────────
                cost = estimate_cost(response.model, response.input_tokens, response.output_tokens)
                task.cost_usd += cost
                token_key = f"{provider_name}/{response.model}"
                task.tokens_used[token_key] = (
                    task.tokens_used.get(token_key, 0) + response.total_tokens
                )

                agent_calls_total.labels(role=role.value, provider=provider_name, status="success").inc()
                if cost > 0:
                    cost_usd_total.labels(provider=provider_name, model=response.model).inc(cost)

                subtask.status = "done"
                subtask.finished_at = time.time()
                self._tracer.subtask_finished(
                    subtask_id, role.value, success=True,
                    duration_ms=duration_ms,
                    tokens={"input": response.input_tokens, "output": response.output_tokens},
                )
                return response.content, subtask

            except ProviderError as e:
                last_error = e
                if not e.retryable or attempt == _max_retries - 1:
                    break
                delay = _retry_base_delay * (2 ** attempt)
                log.warning(
                    "agent.retrying",
                    role=role.value,
                    provider=provider_name,
                    model=model_id,
                    attempt=attempt + 1,
                    max_retries=_max_retries,
                    delay_s=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

        # All retries exhausted (or non-retryable error on first attempt)
        duration_ms = (time.time() - start) * 1000
        error_msg = str(last_error) if last_error else "Unknown error"
        subtask.status = "failed"
        subtask.error = error_msg
        subtask.finished_at = time.time()
        agent_calls_total.labels(role=role.value, provider=provider_name, status="failed").inc()
        self._tracer.subtask_finished(
            subtask_id, role.value, success=False,
            duration_ms=duration_ms, error=error_msg,
        )
        log.error(
            "agent.call_failed",
            role=role.value,
            provider=provider_name,
            model=model_id,
            error=error_msg,
            attempts=attempt + 1,
        )
        return "", subtask

    def _parse_review(self, text: str) -> tuple[str, list[str], list[str]]:
        """Parse a model's JSON review response into (verdict, concerns, praise)."""
        data = _extract_json(text)
        if data:
            return (
                data.get("overall_verdict", "needs_discussion"),
                data.get("key_concerns", []),
                data.get("praise", []),
            )
        verdict = "approve" if "looks good" in text.lower() else "needs_discussion"
        return verdict, [], []

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
        enforce_diff_safety(diff)     # raises SafetyViolationError on dangerous patterns
        diff = mask_pii(diff)          # scrub emails / API keys before storing
        return CodePatch(
            id=make_artifact_id(),
            kind=ArtifactKind.CODE_PATCH,
            provenance=provenance,
            unified_diff=diff,
            description=mask_pii(content[:500] if len(content) > 500 else content),
            confidence=confidence,
        )
