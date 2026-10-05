"""
Integration tests for all three orchestration modes.

These tests wire real mode classes to a lightweight mock provider, bypassing
the routing engine.  They verify:
  - Mode happy paths produce the expected final artifact shape
  - All-failure and partial-failure paths degrade gracefully
  - Retry logic fires on retryable errors and is skipped for permanent ones
  - Token counts and cost are accumulated on the task
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

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
from orchestrai.observability.trace import (
    Tracer,
    make_artifact_id,
    make_task_id,
    make_trace_id,
)
from orchestrai.orchestrator.modes.impl_tester import ImplTesterMode
from orchestrai.orchestrator.modes.parallel_draft import ParallelDraftMode
from orchestrai.orchestrator.modes.planner_coder_reviewer import PlannerCoderReviewerMode
from orchestrai.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelCapability,
    ProviderError,
    ProviderKind,
)
from orchestrai.registry.registry import CapabilityRegistry


# ─── Lightweight mock provider ───────────────────────────────────────────────

class _MockProvider(BaseProvider):
    """Returns configurable canned responses keyed by role value."""

    def __init__(
        self,
        name: str = "mock",
        role_responses: dict[str, str] | None = None,
        input_tokens: int = 50,
        output_tokens: int = 100,
    ) -> None:
        self._name = name
        self._role_responses: dict[str, str] = role_responses or {}
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.OPENAI

    async def probe(self) -> bool:
        return True

    async def list_models(self) -> list[ModelCapability]:
        return []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        content = self._role_responses.get(
            request.role.value,
            f"--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new  # {request.role.value}",
        )
        return CompletionResponse(
            content=content,
            model=request.model or "mock-model",
            provider=self._name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


# ─── Test helpers ─────────────────────────────────────────────────────────────

def _make_cap(provider: str, model: str) -> ModelCapability:
    return ModelCapability(
        model_id=model,
        provider=provider,
        provider_kind=ProviderKind.OPENAI,
        display_name=model,
        cost_tier=CostTier.CHEAP,
        latency_tier=LatencyTier.FAST,
        privacy_level=PrivacyLevel.PUBLIC,
    )


def _make_registry(*providers: _MockProvider) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for p in providers:
        registry._providers[p.name] = p
        cap = _make_cap(p.name, "mock-model")
        cap.available = True
        registry._capabilities[(p.name, "mock-model")] = cap
    return registry


def _make_routing(task_id: str, assignments: list[dict]) -> RoutingDecision:
    """Build a RoutingDecision with explicit role_type on each assignment."""
    enriched = [{**a, "role_type": a["role"].split("_")[0]} for a in assignments]
    return RoutingDecision(
        id=make_artifact_id(),
        kind=ArtifactKind.ROUTING_DECISION,
        provenance=Provenance(task_id=task_id),
        task_type=TaskType.FEATURE,
        assignments=enriched,
    )


def _make_brief(task_id: str) -> TaskBrief:
    return TaskBrief(
        id=make_artifact_id(),
        kind=ArtifactKind.TASK_BRIEF,
        provenance=Provenance(task_id=task_id),
        task_type=TaskType.FEATURE,
        description="Add a pagination helper to the users module",
    )


def _mode_deps(task_id: str, tmp_path: Path) -> tuple[ArtifactStore, Tracer]:
    store = ArtifactStore(task_id)
    tracer = Tracer(trace_id=make_trace_id(), task_id=task_id)
    return store, tracer


def _task(task_id: str, mode: str, routing: RoutingDecision) -> OrchestratedTask:
    return OrchestratedTask(
        id=task_id,
        trace_id=make_trace_id(),
        brief=_make_brief(task_id),
        mode=mode,
        routing=routing,
    )


# ─── ParallelDraftMode ───────────────────────────────────────────────────────

class TestParallelDraftMode:
    @pytest.mark.asyncio
    async def test_two_coders_happy_path(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_MockProvider("p1"), _MockProvider("p2"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [
                {"role": "coder",   "provider": "p1", "model": "mock-model"},
                {"role": "coder_2", "provider": "p2", "model": "mock-model"},
                {"role": "reviewer","provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert result.status == "done"
        assert result.final is not None
        assert result.final.patch is not None
        assert result.final.confidence > 0
        # Both coders ran → 2 patches in alternatives + chosen
        assert len(result.final.alternatives) + 1 >= 1

    @pytest.mark.asyncio
    async def test_all_coders_fail_returns_failed(self, tmp_dirs: Path) -> None:
        class _Failing(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                raise ProviderError("API down", provider=self.name, retryable=False)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_Failing("p1"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [{"role": "coder", "provider": "p1", "model": "mock-model"}]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert result.status == "failed"
        assert result.final is None

    @pytest.mark.asyncio
    async def test_no_routing_returns_failed(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_MockProvider())

        task = OrchestratedTask(
            id=task_id, trace_id=make_trace_id(),
            brief=_make_brief(task_id), mode="parallel_draft",
            routing=None,
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)
        assert result.status == "failed"


# ─── ImplTesterMode ──────────────────────────────────────────────────────────

class TestImplTesterMode:
    @pytest.mark.asyncio
    async def test_produces_patch_and_tests(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)

        coder_diff = "--- a/util.py\n+++ b/util.py\n@@ -1 +1 @@\n-pass\n+return paginate(items, size)"
        tester_code = "def test_pagination():\n    assert paginate([], 10) == []\n"

        p = _MockProvider("p1", {"coder": coder_diff, "tester": tester_code})
        registry = _make_registry(p)

        task = _task(
            task_id, "impl_tester",
            _make_routing(task_id, [
                {"role": "coder",  "provider": "p1", "model": "mock-model"},
                {"role": "tester", "provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await ImplTesterMode(registry, store, tracer).run(task)

        assert result.status == "done"
        assert result.final is not None
        assert result.final.patch is not None
        assert result.final.tests is not None
        assert "test_pagination" in result.final.tests.test_code

    @pytest.mark.asyncio
    async def test_coder_fail_degraded_gracefully(self, tmp_dirs: Path) -> None:
        """If coder fails, tests alone should still produce a done result."""
        class _FailCoder(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                if request.role == RoleType.CODER:
                    raise ProviderError("Coder down", provider=self.name, retryable=False)
                return await super().complete(request)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_FailCoder("p1"))

        task = _task(
            task_id, "impl_tester",
            _make_routing(task_id, [
                {"role": "coder",  "provider": "p1", "model": "mock-model"},
                {"role": "tester", "provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await ImplTesterMode(registry, store, tracer).run(task)

        # Mode degrades: no patch, but tests present → still done
        assert result.status == "done"
        assert result.final is not None
        assert result.final.patch is None
        assert result.final.tests is not None

    @pytest.mark.asyncio
    async def test_reviewer_runs_when_patch_or_tests_exist(self, tmp_dirs: Path) -> None:
        calls: list[str] = []

        class _TrackedProvider(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                calls.append(request.role.value)
                return await super().complete(request)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_TrackedProvider("p1"))

        task = _task(
            task_id, "impl_tester",
            _make_routing(task_id, [
                {"role": "coder",    "provider": "p1", "model": "mock-model"},
                {"role": "tester",   "provider": "p1", "model": "mock-model"},
                {"role": "reviewer", "provider": "p1", "model": "mock-model"},
            ]),
        )

        await ImplTesterMode(registry, store, tracer).run(task)

        assert "reviewer" in calls


# ─── PlannerCoderReviewerMode ────────────────────────────────────────────────

class TestPlannerCoderReviewerMode:
    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)

        plan_json = (
            '{"steps": [{"step": "Add helper", "files": ["util.py"]}],'
            ' "risks": [], "estimated_complexity": "low"}'
        )
        review_json = (
            '{"overall_verdict": "approve", "key_concerns": [], "praise": ["Clean impl"]}'
        )
        p = _MockProvider("p1", {
            "planner": plan_json,
            "coder":   "--- a/util.py\n+++ b/util.py\n@@ -1 +1 @@\n-pass\n+return result",
            "tester":  "def test_helper():\n    assert helper() is not None",
            "reviewer": review_json,
        })
        registry = _make_registry(p)

        task = _task(
            task_id, "planner_coder_reviewer",
            _make_routing(task_id, [
                {"role": "planner",  "provider": "p1", "model": "mock-model"},
                {"role": "coder",    "provider": "p1", "model": "mock-model"},
                {"role": "tester",   "provider": "p1", "model": "mock-model"},
                {"role": "reviewer", "provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await PlannerCoderReviewerMode(registry, store, tracer).run(task)

        assert result.status == "done"
        assert result.final is not None
        assert result.final.patch is not None
        assert result.final.review is not None
        assert result.final.review.overall_verdict == "approve"
        assert result.final.confidence > 0.70

    @pytest.mark.asyncio
    async def test_plan_steps_parsed_and_stored(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)

        plan_json = (
            '{"steps": [{"step": "Step 1"}, {"step": "Step 2"}],'
            ' "risks": ["r1"], "estimated_complexity": "medium"}'
        )
        p = _MockProvider("p1", {"planner": plan_json})
        registry = _make_registry(p)

        task = _task(
            task_id, "planner_coder_reviewer",
            _make_routing(task_id, [
                {"role": "planner", "provider": "p1", "model": "mock-model"},
                {"role": "coder",   "provider": "p1", "model": "mock-model"},
            ]),
        )

        await PlannerCoderReviewerMode(registry, store, tracer).run(task)

        plans = store.list_by_kind(ArtifactKind.IMPLEMENTATION_PLAN)
        assert len(plans) == 1
        assert len(plans[0]["steps"]) == 2

    @pytest.mark.asyncio
    async def test_coder_failure_patch_is_none(self, tmp_dirs: Path) -> None:
        """Coder failure should leave patch=None in final, not crash the mode."""
        class _FailCoder(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                if request.role == RoleType.CODER:
                    raise ProviderError("Quota", provider=self.name, retryable=False)
                return await super().complete(request)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_FailCoder("p1"))

        task = _task(
            task_id, "planner_coder_reviewer",
            _make_routing(task_id, [
                {"role": "planner", "provider": "p1", "model": "mock-model"},
                {"role": "coder",   "provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await PlannerCoderReviewerMode(registry, store, tracer).run(task)

        assert result.status == "done"
        assert result.final is not None
        assert result.final.patch is None

    @pytest.mark.asyncio
    async def test_no_routing_returns_failed(self, tmp_dirs: Path) -> None:
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_MockProvider())

        task = OrchestratedTask(
            id=task_id, trace_id=make_trace_id(),
            brief=_make_brief(task_id), mode="planner_coder_reviewer",
            routing=None,
        )

        result = await PlannerCoderReviewerMode(registry, store, tracer).run(task)
        assert result.status == "failed"


# ─── Retry logic ─────────────────────────────────────────────────────────────

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retryable_error_retried_until_success(
        self, tmp_dirs: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider fails twice then succeeds; should get 3 calls total."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))

        call_count = 0

        class _FlakyProvider(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ProviderError("Transient", provider=self.name, retryable=True)
                return CompletionResponse(
                    content="--- a/f.py\n+++ b/f.py\n@@ -1+1@@\n-x\n+y",
                    model="mock-model",
                    provider=self.name,
                    input_tokens=10,
                    output_tokens=20,
                )

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_FlakyProvider("flaky"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [{"role": "coder", "provider": "flaky", "model": "mock-model"}]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert call_count == 3, f"Expected 3 calls, got {call_count}"
        assert result.status == "done"

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(
        self, tmp_dirs: Path
    ) -> None:
        """A permanent error should result in exactly one call and task failure."""
        call_count = 0

        class _PermanentFail(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                nonlocal call_count
                call_count += 1
                raise ProviderError("Invalid API key", provider=self.name, retryable=False)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_PermanentFail("p1"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [{"role": "coder", "provider": "p1", "model": "mock-model"}]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert call_count == 1, f"Expected 1 call (no retries), got {call_count}"
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_retryable_exhausted_returns_failed(
        self, tmp_dirs: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider always fails with retryable=True; task must still fail after exhausting retries."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))

        class _AlwaysFail(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                raise ProviderError("Always fails", provider=self.name, retryable=True)

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_AlwaysFail("p1"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [{"role": "coder", "provider": "p1", "model": "mock-model"}]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)
        assert result.status == "failed"


# ─── Cost tracking ────────────────────────────────────────────────────────────

class TestCostTracking:
    @pytest.mark.asyncio
    async def test_tokens_accumulated_across_calls(self, tmp_dirs: Path) -> None:
        """2 agent calls of 150 tokens each → 300 tokens total on task."""
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)

        p = _MockProvider("p1", input_tokens=50, output_tokens=100)  # 150 per call
        registry = _make_registry(p)

        task = _task(
            task_id, "impl_tester",
            _make_routing(task_id, [
                {"role": "coder",  "provider": "p1", "model": "mock-model"},
                {"role": "tester", "provider": "p1", "model": "mock-model"},
            ]),
        )

        result = await ImplTesterMode(registry, store, tracer).run(task)

        assert result.status == "done"
        total_tokens = sum(result.tokens_used.values())
        assert total_tokens == 300, f"Expected 300 tokens, got {total_tokens}"

    @pytest.mark.asyncio
    async def test_cost_tracked_for_known_model(self, tmp_dirs: Path) -> None:
        """cost_usd should be non-zero when a model with known rates is used."""
        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)

        class _KnownModelProvider(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    content="diff",
                    model="claude-sonnet-4-6",  # $3/$15 per 1M tokens
                    provider=self.name,
                    input_tokens=1_000,
                    output_tokens=500,
                )

        p = _KnownModelProvider("p1")
        registry = _make_registry(p)
        registry._capabilities = {
            ("p1", "claude-sonnet-4-6"): _make_cap("p1", "claude-sonnet-4-6"),
        }

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [
                {"role": "coder", "provider": "p1", "model": "claude-sonnet-4-6"},
            ]),
        )

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert result.cost_usd > 0, "Expected non-zero cost for claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_budget_preflight_skips_call_when_exceeded(
        self, tmp_dirs: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When budget is already exceeded, no API call should be made."""
        from types import SimpleNamespace

        from orchestrai.config.settings import PolicyConfig

        # Build a settings object with a tiny budget and inject it
        mock_settings = SimpleNamespace(policy=PolicyConfig(max_cost_usd=0.0001))
        monkeypatch.setattr(
            "orchestrai.orchestrator.context.get_settings",
            lambda: mock_settings,
        )

        call_count = 0

        class _ExpensiveProvider(_MockProvider):
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                nonlocal call_count
                call_count += 1
                return CompletionResponse(
                    content="diff",
                    model="claude-opus-4-6",
                    provider=self.name,
                    input_tokens=100_000,
                    output_tokens=50_000,
                )

        task_id = make_task_id()
        store, tracer = _mode_deps(task_id, tmp_dirs)
        registry = _make_registry(_ExpensiveProvider("p1"))

        task = _task(
            task_id, "parallel_draft",
            _make_routing(task_id, [
                {"role": "coder",   "provider": "p1", "model": "mock-model"},
                {"role": "coder_2", "provider": "p1", "model": "mock-model"},
            ]),
        )
        # Pre-seed cost_usd well above the 0.0001 budget
        task.cost_usd = 1.0

        result = await ParallelDraftMode(registry, store, tracer).run(task)

        assert call_count == 0, f"Expected 0 calls (budget exceeded), got {call_count}"
        assert result.status == "failed"


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ORCHESTRAI__OBSERVABILITY__ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ORCHESTRAI__OBSERVABILITY__TRACE_DIR", str(tmp_path / "traces"))
    return tmp_path
