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


class Orchestrator:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry
        self._settings = get_settings()
        self._router = RoutingEngine(registry, self._settings.policy)
        self._active: dict[str, OrchestratedTask] = {}

    async def submit(
        self,
        request: str,
        repo_root: str | None = None,
        target_files: list[str] | None = None,
        mode: str | None = None,
        user_preferences: dict[str, Any] | None = None,
    ) -> OrchestratedTask:
        """
        Main entrypoint: intake a task and orchestrate it end-to-end.
        Returns the completed OrchestratedTask.
        """
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

        # Scan repo for context
        if repo_root:
            repo_summary = scan_repo(repo_root)
            repo_summary.provenance.task_id = task_id
            store.put(repo_summary)
            brief.context_snippets.append(f"Repo: {repo_summary.summary_text}")

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

        # Execute mode
        mode_cls = MODE_MAP.get(mode, PlannerCoderReviewerMode)
        executor = mode_cls(
            registry=self._registry,
            store=store,
            tracer=tracer,
        )

        start = time.time()
        try:
            await asyncio.wait_for(
                executor.run(task),
                timeout=self._settings.orchestrator.timeout_budget_sec,
            )
        except asyncio.TimeoutError:
            task.status = "failed"
            task.error = f"Task exceeded timeout of {self._settings.orchestrator.timeout_budget_sec}s"
            log.error("orchestrator.timeout", task_id=task_id)
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            log.exception("orchestrator.error", task_id=task_id, error=str(e))
        finally:
            task.finished_at = time.time()
            tracer.task_finished(
                success=task.status == "done",
                total_tokens={},
                cost_usd=None,
                error=task.error,
            )
            tracer.save()
            self._active.pop(task_id, None)

        log.info(
            "orchestrator.complete",
            task_id=task_id,
            status=task.status,
            duration_s=round(time.time() - start, 2),
        )
        return task

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
