"""
Core Orchestrator — entry point for all task orchestration.
Wires together: intake → routing → mode execution → trace → result.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from orchestrai.artifacts.schemas import OrchestratedTask, TaskType
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.config.settings import get_settings
from orchestrai.observability.metrics import task_duration_seconds, tasks_total
from orchestrai.observability.trace import Tracer, make_task_id, make_trace_id
from orchestrai.orchestrator.intake import build_task_brief, scan_repo
from orchestrai.orchestrator.modes.impl_tester import ImplTesterMode
from orchestrai.orchestrator.modes.parallel_draft import ParallelDraftMode
from orchestrai.orchestrator.modes.planner_coder_reviewer import PlannerCoderReviewerMode
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.registry.router import RoutingEngine

log = structlog.get_logger()

MODE_MAP = {
    "parallel_draft": ParallelDraftMode,
    "impl_tester": ImplTesterMode,
    "planner_coder_reviewer": PlannerCoderReviewerMode,
    "bugfix": PlannerCoderReviewerMode,      # alias
    "refactor": PlannerCoderReviewerMode,    # alias
    "docs": ImplTesterMode,                  # alias
}


_MAX_FINISHED = 100  # cap recent-completed cache to avoid unbounded memory growth


class Orchestrator:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry
        self._settings = get_settings()
        self._router = RoutingEngine(registry, self._settings.policy)
        self._active: dict[str, OrchestratedTask] = {}
        self._finished: dict[str, OrchestratedTask] = {}  # recently completed tasks
        self._bg_tasks: dict[str, asyncio.Task[None]] = {}  # background asyncio tasks

    async def submit(
        self,
        request: str,
        repo_root: str | None = None,
        target_files: list[str] | None = None,
        mode: str | None = None,
        user_preferences: dict[str, Any] | None = None,
        wait: bool = True,
    ) -> OrchestratedTask:
        """
        Main entrypoint: intake a task and orchestrate it end-to-end.

        When wait=True (default) blocks until the task finishes and returns the
        completed OrchestratedTask.  When wait=False, fires the execution as a
        background asyncio task and returns immediately with status="running".
        """
        task, store, tracer, executor = await self._prepare(
            request=request,
            repo_root=repo_root,
            target_files=target_files,
            mode=mode,
            user_preferences=user_preferences,
        )

        if wait:
            await self._run(task, store, tracer, executor)
        else:
            bg = asyncio.create_task(self._run(task, store, tracer, executor))
            self._bg_tasks[task.id] = bg
            # Remove from bg_tasks when done; errors are handled inside _run
            bg.add_done_callback(lambda _: self._bg_tasks.pop(task.id, None))

        return task

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a background task by task_id.

        Returns True if a running background task was found and cancelled,
        False if the task is not in the background queue (already finished or
        was a synchronous submit).
        """
        bg = self._bg_tasks.get(task_id)
        if bg is None or bg.done():
            return False
        bg.cancel()
        try:
            await bg
        except (asyncio.CancelledError, Exception):
            pass
        # Mark the task as failed in both caches
        task = self._active.get(task_id) or self._finished.get(task_id)
        if task and task.status == "running":
            task.status = "failed"
            task.error = "Cancelled by client"
        return True

    async def _prepare(
        self,
        request: str,
        repo_root: str | None,
        target_files: list[str] | None,
        mode: str | None,
        user_preferences: dict[str, Any] | None,
    ) -> tuple[OrchestratedTask, ArtifactStore, Tracer, Any]:
        """Set up all state for a task without running it. Returns (task, store, tracer, executor)."""
        task_id = make_task_id()
        trace_id = make_trace_id()
        store = ArtifactStore(task_id)
        tracer = Tracer(trace_id=trace_id, task_id=task_id)

        # Intake
        brief = build_task_brief(
            request=request,
            task_id=task_id,
            repo_root=repo_root,
            target_files=target_files,
        )
        store.put(brief)

        # Scan repo for context and propagate detected framework into brief metadata
        if repo_root:
            repo_summary = scan_repo(repo_root)
            repo_summary.provenance.task_id = task_id
            store.put(repo_summary)
            brief.context_snippets.append(f"Repo: {repo_summary.summary_text}")
            if repo_summary.test_framework:
                brief.metadata["test_framework"] = repo_summary.test_framework
            if repo_summary.language:
                brief.metadata["language"] = repo_summary.language

        # Resolve mode
        if mode is None:
            mode = self._router.recommended_mode(brief.task_type)

        # Route
        routing = self._router.route(
            task_type=brief.task_type,
            mode=mode,
            task_id=task_id,
            user_preferences=user_preferences,
        )
        store.put(routing)

        # Build task object
        task = OrchestratedTask(
            id=task_id,
            trace_id=trace_id,
            brief=brief,
            mode=mode,
            routing=routing,
            status="running",
        )
        self._active[task_id] = task

        tracer.task_started(mode=mode, task_type=brief.task_type.value)
        tracer.routing_decided(assignments=routing.assignments, rationale=routing.rationale)

        # Build mode executor
        mode_cls = MODE_MAP.get(mode, PlannerCoderReviewerMode)
        executor = mode_cls(
            registry=self._registry,
            store=store,
            tracer=tracer,
        )

        return task, store, tracer, executor

    async def _run(
        self,
        task: OrchestratedTask,
        store: ArtifactStore,
        tracer: Tracer,
        executor: Any,
    ) -> None:
        """Execute a prepared task to completion, handling timeout and errors."""
        start = time.time()
        executor_task = asyncio.create_task(executor.run(task))
        try:
            await asyncio.wait_for(
                executor_task,
                timeout=self._settings.orchestrator.timeout_budget_sec,
            )
        except asyncio.TimeoutError:
            # wait_for has already issued a cancel; wait for it to propagate so
            # all child coroutines (asyncio.gather subtasks) are fully cancelled.
            try:
                await executor_task
            except (asyncio.CancelledError, Exception):
                pass
            task.status = "failed"
            task.error = f"Task exceeded timeout of {self._settings.orchestrator.timeout_budget_sec}s"
            log.error("orchestrator.timeout", task_id=task.id)
        except asyncio.CancelledError:
            task.status = "failed"
            task.error = "Cancelled by client"
            executor_task.cancel()
            try:
                await executor_task
            except (asyncio.CancelledError, Exception):
                pass
            raise  # re-raise so the background task wrapper records it correctly
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            log.exception("orchestrator.error", task_id=task.id, error=str(e))
        finally:
            task.finished_at = time.time()
            # Propagate accumulated cost/tokens into the final artifact
            if task.final is not None:
                task.final.tokens_used = task.tokens_used
                task.final.cost_usd = task.cost_usd
            tracer.task_finished(
                success=task.status == "done",
                total_tokens=task.tokens_used,
                cost_usd=task.cost_usd if task.cost_usd > 0 else None,
                error=task.error,
            )
            tracer.save()
            self._active.pop(task.id, None)
            # Keep in finished cache for inspect tools; evict oldest if over cap
            self._finished[task.id] = task
            if len(self._finished) > _MAX_FINISHED:
                oldest = next(iter(self._finished))
                del self._finished[oldest]

        _duration = time.time() - start
        tasks_total.labels(status=task.status, mode=task.mode).inc()
        task_duration_seconds.labels(mode=task.mode).observe(_duration)
        log.info(
            "orchestrator.complete",
            task_id=task.id,
            status=task.status,
            duration_s=round(_duration, 2),
        )

    def get_active_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": t.id,
                "status": t.status,
                "mode": t.mode,
                "task_type": t.brief.task_type.value,
                "subtasks": len(t.subtasks),
            }
            for t in self._active.values()
        ]

    def get_recent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recently finished tasks, newest first."""
        finished = list(self._finished.values())
        finished.sort(key=lambda t: t.finished_at or 0, reverse=True)
        return [
            {
                "id": t.id,
                "status": t.status,
                "mode": t.mode,
                "task_type": t.brief.task_type.value,
                "subtasks": len(t.subtasks),
                "finished_at": t.finished_at,
                "cost_usd": t.cost_usd,
            }
            for t in finished[:limit]
        ]
