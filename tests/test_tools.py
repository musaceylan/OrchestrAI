"""
Tests for MCP tool handlers and the new async-submit features.

Coverage:
- role_strengths property on ModelCapability (AttributeError regression)
- inspect_artifacts enum no longer accepts "tool_result"
- list_tasks / get_task_status / cancel_task handlers
- submit_task wait=false (background / fire-and-forget)
- cancel_task for running vs finished tasks
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrai.artifacts.schemas import (
    ArtifactKind,
    CostTier,
    FinalDecision,
    LatencyTier,
    OrchestratedTask,
    PrivacyLevel,
    Provenance,
    RoutingDecision,
    RoleType,
    TaskBrief,
    TaskType,
)
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.observability.trace import make_artifact_id, make_task_id, make_trace_id
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelCapability,
    ProviderError,
    ProviderKind,
)
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.tools import handle_tool


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_capability(
    provider: str = "mock",
    model_id: str = "mock-model",
    planning_strength: float = 0.9,
    coding_strength: float = 0.85,
) -> ModelCapability:
    return ModelCapability(
        provider=provider,
        provider_kind=ProviderKind.OPENAI,
        model_id=model_id,
        display_name="Mock Model",
        planning_strength=planning_strength,
        coding_strength=coding_strength,
        latency_tier=LatencyTier.FAST,
        cost_tier=CostTier.MEDIUM,
        privacy_level=PrivacyLevel.PUBLIC,
        preferred_roles=[RoleType.PLANNER, RoleType.CODER],
    )


class _MockProvider(BaseProvider):
    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.OPENAI

    async def probe(self) -> bool:
        return True

    async def list_models(self) -> list[ModelCapability]:
        return [_make_capability(self._name)]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            content="mock response",
            model=request.model or "mock-model",
            provider=self._name,
            input_tokens=10,
            output_tokens=20,
        )


@pytest.fixture
async def registry() -> CapabilityRegistry:
    provider = _MockProvider()
    return await CapabilityRegistry.build([provider])


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORCHESTRAI__OBSERVABILITY__ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ORCHESTRAI__OBSERVABILITY__TRACE_DIR", str(tmp_path / "traces"))
    return tmp_path


# ─── ModelCapability.role_strengths ──────────────────────────────────────────

class TestRoleStrengthsProperty:
    """role_strengths was missing; list_available_models crashed with AttributeError."""

    def test_role_strengths_returns_dict(self):
        cap = _make_capability()
        rs = cap.role_strengths
        assert isinstance(rs, dict)
        assert RoleType.PLANNER in rs
        assert RoleType.CODER in rs

    def test_role_strengths_values_match_strength_for_role(self):
        cap = _make_capability(planning_strength=0.9, coding_strength=0.75)
        rs = cap.role_strengths
        assert rs[RoleType.PLANNER] == cap.strength_for_role(RoleType.PLANNER)
        assert rs[RoleType.CODER] == cap.strength_for_role(RoleType.CODER)

    def test_role_strengths_covers_all_roles(self):
        cap = _make_capability()
        rs = cap.role_strengths
        assert set(rs.keys()) == set(RoleType)

    def test_role_strengths_iterable_for_tool(self):
        """Simulate exactly what _list_available_models does."""
        cap = _make_capability()
        # This line crashed before the fix
        strengths = {r.value: round(s, 2) for r, s in cap.role_strengths.items()}
        assert "planner" in strengths
        assert "coder" in strengths
        assert isinstance(strengths["planner"], float)


# ─── inspect_artifacts schema ─────────────────────────────────────────────────

class TestInspectArtifactsSchema:
    """'tool_result' was listed in the enum but doesn't exist in ArtifactKind."""

    def test_tool_result_not_in_artifact_kind(self):
        with pytest.raises(ValueError):
            ArtifactKind("tool_result")

    def test_valid_artifact_kinds_work(self):
        for value in [
            "task_brief", "code_patch", "test_candidate", "review_comments",
            "judge_verdict", "final_decision", "implementation_plan",
            "repo_summary", "routing_decision",
        ]:
            kind = ArtifactKind(value)
            assert kind.value == value


# ─── list_tasks tool ─────────────────────────────────────────────────────────

class TestListTasksTool:
    async def test_list_tasks_empty(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("list_tasks", {}, orch, registry)
        assert "active" in result
        assert "recent_finished" in result
        assert result["active"] == []
        assert result["recent_finished"] == []

    async def test_list_tasks_custom_limit(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("list_tasks", {"limit": 5}, orch, registry)
        assert "active" in result
        assert "recent_finished" in result

    async def test_list_tasks_shows_finished(self, registry: CapabilityRegistry, tmp_dirs: Path):
        """A task submitted with wait=True appears in recent_finished after completion."""
        orch = Orchestrator(registry)

        # Inject a finished task directly into _finished cache
        task_id = make_task_id()
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
            description="test task",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="done",
            finished_at=1000.0,
        )
        orch._finished[task_id] = task

        result = await handle_tool("list_tasks", {}, orch, registry)
        ids = [t["id"] for t in result["recent_finished"]]
        assert task_id in ids


# ─── get_task_status tool ─────────────────────────────────────────────────────

class TestGetTaskStatusTool:
    async def test_unknown_task_returns_error(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("get_task_status", {"task_id": "nonexistent-id"}, orch, registry)
        assert "error" in result

    async def test_invalid_task_id_returns_error(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("get_task_status", {"task_id": "../etc/passwd"}, orch, registry)
        assert "error" in result

    async def test_running_task_status(self, registry: CapabilityRegistry, tmp_dirs: Path):
        """A task in _active returns status='running'."""
        orch = Orchestrator(registry)
        task_id = make_task_id()
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.BUGFIX,
            description="fix the bug",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.BUGFIX,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="planner_coder_reviewer",
            routing=routing,
            status="running",
        )
        orch._active[task_id] = task

        result = await handle_tool("get_task_status", {"task_id": task_id}, orch, registry)
        assert result["status"] == "running"
        assert result["task_id"] == task_id
        assert result["mode"] == "planner_coder_reviewer"

    async def test_finished_task_includes_summary(self, registry: CapabilityRegistry, tmp_dirs: Path):
        """A done task includes summary and confidence in get_task_status."""
        orch = Orchestrator(registry)
        task_id = make_task_id()
        provenance = Provenance(task_id=task_id)
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=provenance,
            task_type=TaskType.FEATURE,
            description="add feature",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=provenance,
            task_type=TaskType.FEATURE,
        )
        final = FinalDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.FINAL_DECISION,
            provenance=provenance,
            summary="Feature implemented",
            confidence=0.9,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="done",
            final=final,
            finished_at=2000.0,
        )
        orch._finished[task_id] = task

        result = await handle_tool("get_task_status", {"task_id": task_id}, orch, registry)
        assert result["status"] == "done"
        assert result["summary"] == "Feature implemented"
        assert result["confidence"] == 0.9


# ─── cancel_task tool ─────────────────────────────────────────────────────────

class TestCancelTaskTool:
    async def test_cancel_nonexistent_task(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("cancel_task", {"task_id": "no-such-task"}, orch, registry)
        assert "error" in result

    async def test_cancel_invalid_id(self, registry: CapabilityRegistry, tmp_dirs: Path):
        orch = Orchestrator(registry)
        result = await handle_tool("cancel_task", {"task_id": "../bad"}, orch, registry)
        assert "error" in result

    async def test_cancel_finished_task_not_cancellable(
        self, registry: CapabilityRegistry, tmp_dirs: Path
    ):
        orch = Orchestrator(registry)
        task_id = make_task_id()
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
            description="done",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="done",
        )
        orch._finished[task_id] = task

        result = await handle_tool("cancel_task", {"task_id": task_id}, orch, registry)
        assert result["cancelled"] is False

    async def test_cancel_background_task(self, registry: CapabilityRegistry, tmp_dirs: Path):
        """Background task in _bg_tasks is cancelled and cancel_task returns True."""
        orch = Orchestrator(registry)
        task_id = make_task_id()

        # Plant a long-running fake asyncio task in _bg_tasks
        async def _never_finish() -> None:
            await asyncio.sleep(3600)

        bg = asyncio.create_task(_never_finish())
        orch._bg_tasks[task_id] = bg

        # Also put a running task in _active so cancel_task can find it
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
            description="slow task",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="running",
        )
        orch._active[task_id] = task

        result = await handle_tool("cancel_task", {"task_id": task_id}, orch, registry)
        assert result["cancelled"] is True
        assert bg.cancelled()


# ─── submit_task wait=false ───────────────────────────────────────────────────

class TestSubmitTaskAsync:
    async def test_wait_false_returns_running(
        self, registry: CapabilityRegistry, tmp_dirs: Path
    ):
        """submit_task with wait=false returns immediately with status running."""
        orch = Orchestrator(registry)

        # Patch _run to suspend indefinitely so we can inspect mid-flight state
        async def _slow_run(*args, **kwargs) -> None:
            await asyncio.sleep(3600)

        with patch.object(orch, "_run", side_effect=_slow_run):
            result = await handle_tool(
                "submit_task",
                {"request": "add a feature", "wait": False},
                orch,
                registry,
            )

        assert result["status"] == "running"
        assert "task_id" in result
        # Cleanup any lingering tasks
        for bg in list(orch._bg_tasks.values()):
            bg.cancel()
            try:
                await bg
            except (asyncio.CancelledError, Exception):
                pass

    async def test_wait_true_blocks_to_completion(
        self, registry: CapabilityRegistry, tmp_dirs: Path
    ):
        """submit_task with wait=true (default) returns a completed result."""
        orch = Orchestrator(registry)

        # Patch _run to finish immediately with a done status
        async def _instant_run(
            task: OrchestratedTask, store: ArtifactStore, tracer: object, executor: object
        ) -> None:
            task.status = "done"
            orch._finished[task.id] = task
            orch._active.pop(task.id, None)

        with patch.object(orch, "_run", side_effect=_instant_run):
            result = await handle_tool(
                "submit_task",
                {"request": "fix the bug", "wait": True},
                orch,
                registry,
            )

        assert result["status"] == "done"
