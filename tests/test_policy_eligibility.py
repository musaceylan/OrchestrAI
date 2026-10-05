"""Regression tests for monotonic, server-owned task admission."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from orchestrai.artifacts.schemas import (
    CodePatch,
    CostTier,
    OrchestratedTask,
    PrivacyLevel,
    Provenance,
    ProviderKind,
    RoleType,
    TaskType,
)
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.config import settings as config
from orchestrai.config.settings import PolicyConfig
from orchestrai.orchestrator.context import task_context
from orchestrai.orchestrator.judge import run_judge
from orchestrai.orchestrator.modes.base_mode import BaseMode
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelCapability,
    ProviderError,
)
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.registry.router import RoutingEngine
from orchestrai.server.tools import handle_tool


def inventory() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for name, privacy, strength, cost in (
        ("cloud", PrivacyLevel.PUBLIC, 0.99, CostTier.EXPENSIVE),
        ("enterprise", PrivacyLevel.CONFIDENTIAL, 0.9, CostTier.MEDIUM),
        ("local", PrivacyLevel.SECRET, 0.8, CostTier.CHEAP),
        ("other-local", PrivacyLevel.SECRET, 0.7, CostTier.CHEAP),
    ):
        cap = ModelCapability(
            provider=name, model_id="org/model", provider_kind=ProviderKind.OPENAI_COMPAT,
            privacy_level=privacy, cost_tier=cost,
            role_strengths={role: strength for role in RoleType},
        )
        registry._capabilities[(name, cap.model_id)] = cap
    return registry


def routed(policy: PolicyConfig, preferences: dict[str, Any]) -> list[str]:
    decision = RoutingEngine(inventory(), policy).route(
        TaskType.FEATURE, "impl_tester", "policy-test", preferences,
    )
    return [assignment["provider"] for assignment in decision.assignments]


@pytest.mark.parametrize(
    ("configured", "requested", "expected"),
    [("secret", "public", {"local", "other-local"}),
     ("confidential", "internal", {"enterprise", "local", "other-local"}),
     ("public", "secret", {"local", "other-local"})],
)
def test_strictest_privacy_wins(configured: str, requested: str, expected: set[str]) -> None:
    result = routed(PolicyConfig(privacy_level=configured), {"privacy_level": requested})
    assert result and set(result) <= expected


@pytest.mark.parametrize(("configured", "requested"), [(True, False), (False, True)])
def test_local_only_ors_tighter(configured: bool, requested: bool) -> None:
    result = routed(PolicyConfig(local_only_mode=configured), {"local_only": requested})
    assert result and set(result) <= {"local", "other-local"}


@pytest.mark.parametrize(
    ("allowed", "requested", "expected"),
    [(('cloud', 'local'), ['local', 'enterprise'], {'local'}),
     (('cloud',), ['local'], set()),
     ((), [], set()),
     ((), ['local'], {'local'})],
)
def test_allowlists_intersect_including_empty_denial(
    allowed: tuple[str, ...], requested: list[str], expected: set[str],
) -> None:
    assert set(routed(PolicyConfig(allowed_providers=allowed), {
        "allowed_providers": requested,
    })) == expected


def test_direct_registry_empty_allowlist_is_unrestricted() -> None:
    registry = inventory()
    unrestricted = registry.capabilities_for_role(RoleType.CODER)
    assert unrestricted
    assert registry.capabilities_for_role(
        RoleType.CODER, provider_allowlist=[],
    ) == unrestricted


def test_local_only_preserves_configured_allowlist() -> None:
    assert set(routed(PolicyConfig(allowed_providers=("local",)), {
        "local_only": True,
    })) == {"local"}


def test_denylists_union() -> None:
    assert set(routed(PolicyConfig(denied_providers=("cloud",)), {
        "denied_providers": ["enterprise", "other-local"],
    })) == {"local"}


def test_preferred_provider_ranks_but_never_grants_access() -> None:
    assert routed(PolicyConfig(), {"preferred_providers": ["local"]})[0] == "local"
    assert set(routed(PolicyConfig(allowed_providers=("local",)), {
        "preferred_providers": ["cloud"],
    })) == {"local"}


def test_cost_tier_alias_uses_stricter_minimum() -> None:
    assert set(routed(PolicyConfig(), {
        "cost_max": "expensive", "cost_tier": "cheap",
    })) == {"local", "other-local"}


class RecordingProvider(BaseProvider):
    def __init__(self, cap: ModelCapability) -> None:
        self.cap = cap
        self.identity = cap.provider
        self.calls: list[CompletionRequest] = []
        self.on_call: Callable[[], None] | None = None

    @property
    def name(self) -> str:
        return self.identity

    @property
    def kind(self) -> ProviderKind:
        return self.cap.provider_kind

    async def probe(self) -> bool:
        return True

    async def list_models(self) -> list[ModelCapability]:
        return [self.cap]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if self.on_call:
            self.on_call()
        return CompletionResponse(
            '{"winner_index": 0}', request.model or "", self.name,
        )


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> config.SettingsValues:
    settings = config.SettingsValues(observability=config.ObservabilityConfig(
        artifact_dir=str(tmp_path / "artifacts"), trace_dir=str(tmp_path / "traces"),
    ))
    monkeypatch.setattr(config, "_settings", settings)
    return settings


async def prepared(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    policy: PolicyConfig | None = None, prefs: dict[str, Any] | None = None,
    mode: str = "impl_tester",
) -> tuple[Orchestrator, OrchestratedTask, ArtifactStore, BaseMode,
           list[RecordingProvider], list[CodePatch]]:
    settings = runtime.model_copy(update={"policy": policy or PolicyConfig()})
    monkeypatch.setattr(config, "_settings", settings)
    providers = [RecordingProvider(cap) for cap in inventory().all_capabilities()]
    registry = await CapabilityRegistry.build(list(providers))
    orchestrator = Orchestrator(registry)
    task, store, tracer, executor = await orchestrator._prepare(
        "Implement a helper", None, None, mode, prefs,
    )
    patches = [CodePatch(id=f"patch-{i}", provenance=Provenance(task_id=task.id)) for i in range(2)]
    for patch in patches:
        store.put(patch)
    return orchestrator, task, store, executor, providers, patches


async def test_router_and_judge_share_original_snapshot(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefs = {"allowed_providers": ["local"], "privacy_level": "secret"}
    orch, task, _, _, providers, patches = await prepared(runtime, monkeypatch, prefs=prefs)
    assert {a["provider"] for a in task.routing.assignments} == {"local"}
    prefs["allowed_providers"].append("cloud")
    prefs["privacy_level"] = "public"
    monkeypatch.setattr(config, "_settings", runtime)
    verdict = await run_judge(task, patches, orch._registry)
    assert verdict.evidence_used == ["judge model comparison"]
    assert [p.name for p in providers if p.calls] == ["local"]


@pytest.mark.parametrize("count", [0, 1, 2])
async def test_explicit_judge_override_cannot_bypass(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    orch, task, _, _, providers, patches = await prepared(
        runtime, monkeypatch, policy=PolicyConfig(local_only_mode=True),
    )
    with pytest.raises(ValueError, match="eligib|policy"):
        await run_judge(task, patches[:count], orch._registry, "cloud/org/model")
    assert not any(p.calls for p in providers)


async def test_compare_candidates_cannot_bypass(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, task, store, _, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"denied_providers": ["cloud"]},
    )
    # Isolate admission from Task 7's known fresh-store reopening defect.
    monkeypatch.setattr(ArtifactStore, "list_by_kind", lambda self, kind: [
        value for value in store.all().values() if value["kind"] == kind.value
    ])
    result = await handle_tool("compare_candidates", {
        "task_id": task.id, "judge_model": "cloud/org/model",
    }, orch, orch._registry)
    assert "error" in result
    assert not any(p.calls for p in providers)


@pytest.mark.parametrize("mode", [
    "parallel_draft", "impl_tester", "planner_coder_reviewer", "bugfix", "refactor", "docs",
])
async def test_all_modes_recheck_policy_before_execution(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    _, task, _, executor, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"allowed_providers": ["local"]}, mode=mode,
    )
    # A stale/forged routing assignment is not authority to send task data.
    for assignment in task.routing.assignments:
        assignment["provider"] = "cloud"
    # Every mode also exposes the shared dispatch path, including the docs
    # alias whose role sequence is a separate Task 8 defect.
    _, subtask = await executor._call_agent(
        task, RoleType.CODER, "cloud", "org/model", "task data",
    )
    await executor.run(task)
    assert not any(p.calls for p in providers)
    assert subtask.status == "failed"


@pytest.mark.parametrize("mutation", ["provider", "model", "kind", "unavailable", "missing"])
async def test_identity_rechecked_before_transmission(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    orch, task, _, executor, providers, _ = await prepared(runtime, monkeypatch)
    cap = orch._registry.get_capability("cloud", "org/model")
    if mutation == "provider":
        providers[0].identity = "impostor"
    elif mutation == "model":
        cap.model_id = "different/model"
    elif mutation == "kind":
        cap.provider_kind = ProviderKind.UNKNOWN
    elif mutation == "unavailable":
        cap.available = False
    else:
        del orch._registry._capabilities[("cloud", "org/model")]
    content, subtask = await executor._call_agent(
        task, RoleType.CODER, "cloud", "org/model", "task data",
    )
    assert not providers[0].calls
    assert content == "" and subtask.status == "failed"


async def test_retry_rechecks_capability_after_backoff(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, task, _, executor, providers, _ = await prepared(runtime, monkeypatch)
    cap = orch._registry.get_capability("cloud", "org/model")

    def revoke() -> None:
        cap.available = False
        raise ProviderError("retry", "cloud", retryable=True)

    providers[0].on_call = revoke
    await executor._call_agent(
        task, RoleType.CODER, "cloud", "org/model", "task data", _retry_base_delay=0,
    )
    assert len(providers[0].calls) == 1


@pytest.mark.parametrize("timing", ["initial", "retry"])
@pytest.mark.parametrize("mutation", ["removed", "reclassified", "unavailable"])
async def test_active_task_uses_published_registry(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    timing: str, mutation: str,
) -> None:
    from orchestrai.server.main import publish_runtime

    orch, task, _, executor, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"privacy_level": "secret"},
    )
    original = orch._registry
    replacement = await CapabilityRegistry.build(list(providers))
    cap = replacement.get_capability("local", "org/model")
    assert cap is not None
    if mutation == "removed":
        del replacement._providers["local"]
        del replacement._capabilities[("local", "org/model")]
    elif mutation == "reclassified":
        cap.privacy_level = PrivacyLevel.PUBLIC
    else:
        cap.available = False

    def publish_and_retry() -> None:
        publish_runtime(orch, orch._settings, replacement)
        raise ProviderError("retry after publication", "local", retryable=True)

    if timing == "initial":
        publish_runtime(orch, orch._settings, replacement)
    else:
        providers[2].on_call = publish_and_retry
    content, subtask = await executor._call_agent(
        task, RoleType.CODER, "local", "org/model", "task data", _retry_base_delay=0,
    )
    assert len(providers[2].calls) == (0 if timing == "initial" else 1)
    assert content == "" and subtask.status == "failed"
    assert executor._registry is original  # Publication must not mutate active executors.


@pytest.mark.parametrize(
    ("model", "count", "ceiling", "expected_calls"),
    [("gpt-4o", 1, 0.00035, 0), ("gpt-4o", 2, 0.00105, 1), ("unpriced", 1, 1.0, 0)],
    ids=["single", "concurrent", "unknown-cloud-price"],
)
async def test_cost_reserved_before_transmission(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    model: str, count: int, ceiling: float, expected_calls: int,
) -> None:
    orch, task, _, executor, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"max_cost_usd": ceiling},
    )
    provider = providers[0]
    provider.cap = replace(
        provider.cap, model_id=model, context_window=100, max_output_tokens=10,
    )
    orch._registry = await CapabilityRegistry.build([provider])

    async def complete(request: CompletionRequest) -> CompletionResponse:
        provider.calls.append(request)
        await asyncio.sleep(0)  # Let concurrent requests attempt admission before accounting.
        return CompletionResponse("valid", model, provider.name, 100, 10)

    monkeypatch.setattr(provider, "complete", complete)
    results = await asyncio.gather(*(
        executor._call_agent(task, RoleType.CODER, provider.name, model, "task data")
        for _ in range(count)
    ))
    assert len(provider.calls) == expected_calls
    assert sum(subtask.status == "done" for _, subtask in results) == expected_calls
    assert task.cost_usd <= ceiling


@pytest.mark.parametrize(("ceiling", "spent"), [(0.0, 0.0), (1.0, 1.0)])
async def test_zero_cost_call_allowed_at_ceiling(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    ceiling: float, spent: float,
) -> None:
    _, task, _, executor, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"max_cost_usd": ceiling},
    )
    local = next(provider for provider in providers if provider.name == "local")
    task.cost_usd = spent

    content, subtask = await executor._call_agent(
        task, RoleType.CODER, local.name, local.cap.model_id, "task data",
    )

    assert content
    assert subtask.status == "done"
    assert len(local.calls) == 1
    assert task.cost_usd == spent


async def test_budget_reconciles_without_serializing_provider_calls(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrai.policies.costs import estimate_cost

    orch, task, _, executor, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"max_cost_usd": 0.0015},
    )
    provider = providers[0]
    provider.cap = replace(
        provider.cap, model_id="gpt-4o", context_window=100, max_output_tokens=10,
    )
    orch._registry = await CapabilityRegistry.build([provider])
    both_started = asyncio.Event()

    async def complete(request: CompletionRequest) -> CompletionResponse:
        provider.calls.append(request)
        if len(provider.calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return CompletionResponse("valid", "gpt-4o", provider.name, 1, 1)

    monkeypatch.setattr(provider, "complete", complete)
    results = await asyncio.gather(*(
        executor._call_agent(task, RoleType.CODER, provider.name, "gpt-4o", "task data")
        for _ in range(2)
    ))
    results.append(await executor._call_agent(
        task, RoleType.CODER, provider.name, "gpt-4o", "task data",
    ))
    assert len(provider.calls) == 3
    assert all(subtask.status == "done" for _, subtask in results)
    assert task.cost_usd == pytest.approx(3 * estimate_cost("gpt-4o", 1, 1))
    assert task.tokens_used == {"cloud/gpt-4o": 6}


@pytest.mark.parametrize("cancelled", [False, True])
async def test_uncertain_call_keeps_budget_reserved(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    from orchestrai.orchestrator.context import EligibilityError, task_context

    orch, task, _, _, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"max_cost_usd": 0.0008},
    )
    provider = providers[0]
    provider.cap = replace(
        provider.cap, model_id="gpt-4o", context_window=100, max_output_tokens=10,
    )
    orch._registry = await CapabilityRegistry.build([provider])

    async def complete(request: CompletionRequest) -> CompletionResponse:
        provider.calls.append(request)
        if cancelled:
            raise asyncio.CancelledError
        raise ProviderError("usage unknown", "cloud", retryable=True)

    monkeypatch.setattr(provider, "complete", complete)
    context = task_context(task)
    request = CompletionRequest(messages=[], model="gpt-4o", max_tokens=10)
    with pytest.raises(asyncio.CancelledError if cancelled else ProviderError):
        await context.complete(task, orch._registry, provider.name, request)
    with pytest.raises(EligibilityError, match="cost ceiling"):
        await context.complete(task, orch._registry, provider.name, request)
    assert len(provider.calls) == 1
    assert task.cost_usd == 0 and task.tokens_used == {}


@pytest.mark.parametrize("field", ["provider", "model"])
async def test_response_identity_fails_closed(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    _, task, _, executor, providers, _ = await prepared(runtime, monkeypatch)
    task.cost_usd = 0.5
    task.tokens_used = {"prior/model": 7}

    async def complete(request: CompletionRequest) -> CompletionResponse:
        providers[0].calls.append(request)
        response = CompletionResponse("untrusted", "org/model", "cloud", 10, 10)
        setattr(response, field, "impostor")
        return response

    monkeypatch.setattr(providers[0], "complete", complete)
    content, subtask = await executor._call_agent(
        task, RoleType.CODER, "cloud", "org/model", "task data",
    )
    assert content == "" and subtask.status == "failed"
    assert len(providers[0].calls) == 1  # Invalid responses must not be retried.
    assert task.cost_usd == 0.5 and task.tokens_used == {"prior/model": 7}


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize(
    "value", [-1, 0.5, float("nan"), float("inf"), True, "1", None, 10**400],
    ids=["negative", "fractional", "nan", "infinite", "boolean", "string", "null", "overflow"],
)
async def test_response_accounting_fails_closed(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    field: str, value: object,
) -> None:
    orch, task, _, executor, providers, _ = await prepared(runtime, monkeypatch)
    provider = providers[0]
    provider.cap = replace(provider.cap, model_id="gpt-4o")
    orch._registry = await CapabilityRegistry.build([provider])
    task.cost_usd = 0.5
    task.tokens_used = {"prior/model": 7}

    async def complete(request: CompletionRequest) -> CompletionResponse:
        provider.calls.append(request)
        response = CompletionResponse("untrusted", "gpt-4o", provider.name, 1, 1)
        setattr(response, field, value)
        return response

    monkeypatch.setattr(provider, "complete", complete)
    content, subtask = await executor._call_agent(
        task, RoleType.CODER, provider.name, "gpt-4o", "task data",
    )
    assert content == "" and subtask.status == "failed"
    assert len(provider.calls) == 1
    assert task.cost_usd == 0.5 and task.tokens_used == {"prior/model": 7}


@pytest.mark.parametrize(("configured", "requested"), [(1.0, 10.0), (10.0, 1.0), (None, 0)])
@pytest.mark.parametrize("path", ["mode", "judge"])
async def test_cost_ceiling_minimum_survives_reload(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
    configured: float | None, requested: float, path: str,
) -> None:
    orch, task, _, executor, providers, patches = await prepared(
        runtime, monkeypatch, policy=PolicyConfig(max_cost_usd=configured),
        prefs={"max_cost_usd": requested},
    )
    task.cost_usd = 2.0
    monkeypatch.setattr(config, "_settings", runtime)  # later permissive reload
    if path == "mode":
        await executor._call_agent(task, RoleType.CODER, "cloud", "org/model", "task data")
    else:
        await run_judge(task, patches, orch._registry)
    assert not any(p.calls for p in providers)


@pytest.mark.parametrize("dimension", ["privacy", "allowlist", "denylist", "local", "cost"])
async def test_rerun_cannot_weaken_original_or_admin_policy(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch, dimension: str,
) -> None:
    initial = {
        "privacy": {"privacy_level": "secret"},
        "allowlist": {"allowed_providers": ["local"]},
        "denylist": {"denied_providers": ["cloud", "enterprise"]},
        "local": {"local_only": True},
        "cost": {"cost_tier": "cheap", "max_cost_usd": 0},
    }[dimension]
    orch, task, store, _, providers, _ = await prepared(runtime, monkeypatch, prefs=initial)
    # Rerun must intersect new admin policy too.
    orch._settings = runtime.model_copy(update={"policy": PolicyConfig(
        denied_providers=("other-local",),
    )})
    orch._router = RoutingEngine(orch._registry, orch._settings.policy)
    monkeypatch.setattr(ArtifactStore, "list_by_kind", lambda self, kind: [
        value for value in store.all().values() if value["kind"] == kind.value
    ])
    result = await handle_tool("rerun_with_policy", {
        "task_id": task.id,
        "policy_overrides": {
            "privacy_level": "public", "allowed_providers": ["cloud", "local", "other-local"],
            "denied_providers": [], "local_only": False, "cost_tier": "expensive",
            "max_cost_usd": 100,
        },
    }, orch, orch._registry)
    assert "error" not in result
    assert result["new_task_id"] != task.id
    rerun = orch._finished[result["new_task_id"]]
    if dimension == "cost":
        assert task_context(rerun).eligibility.max_cost_usd == 0
    called = {p.name for p in providers if p.calls}
    assert called == {"local"}


@pytest.mark.parametrize("adapter", ["ollama", "vllm", "compatible"])
@pytest.mark.parametrize("base_url", [
    "https://models.example.com", "http://192.168.1.10:8000", "http://[fd00::1]:8000",
    "http://models.localhost:8000", "http://127.42.7.9:8000", "http://[::1]:8000",
])
async def test_native_endpoint_trust_and_private_routing(adapter: str, base_url: str) -> None:
    import respx

    from orchestrai.config.settings import LocalProviderEndpoint
    from orchestrai.providers.ollama import OllamaProvider
    from orchestrai.providers.openai_compat import OpenAICompatProvider
    from orchestrai.providers.vllm import VLLMProvider

    endpoint = LocalProviderEndpoint(name="endpoint", base_url=base_url)
    provider = {
        "ollama": lambda: OllamaProvider(base_url, name="endpoint"),
        "vllm": lambda: VLLMProvider(base_url, name="endpoint"),
        "compatible": lambda: OpenAICompatProvider(endpoint),
    }[adapter]()
    with respx.mock as mock:
        mock.get(f"{base_url}/api/version").respond(200, json={"version": "test"})
        mock.get(f"{base_url}/api/tags").respond(200, json={"models": [{"name": "org/model"}]})
        mock.post(f"{base_url}/api/show").respond(200, json={})
        mock.get(f"{base_url}/health").respond(200)
        mock.get(f"{base_url}/v1/models").respond(200, json={"data": [{"id": "org/model"}]})
        # Each adapter uses only its own discovery routes.
        mock.assert_all_called = False
        caps = await provider._discover_models() if adapter != "compatible" else (
            provider._build_capabilities([{"id": "org/model"}])
        )
    assert len(caps) == 1 and caps[0].model_id == "org/model"
    assert caps[0].privacy_level == (
        PrivacyLevel.SECRET if endpoint.is_loopback else PrivacyLevel.PUBLIC
    )
    registry = CapabilityRegistry()
    registry._providers[provider.name] = provider
    registry._capabilities[(provider.name, caps[0].model_id)] = caps[0]
    for policy in (PolicyConfig(local_only_mode=True), PolicyConfig(privacy_level="secret")):
        result = RoutingEngine(registry, policy).route(TaskType.FEATURE, "impl_tester", "endpoint")
        assert bool(result.assignments) is endpoint.is_loopback


async def test_snapshot_is_immutable_and_not_request_serializable(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import FrozenInstanceError

    from orchestrai.orchestrator.context import task_context

    prefs = {"allowed_providers": ["local"], "denied_providers": ["cloud"],
             "preferred_providers": ["local"], "max_cost_usd": 1}
    _, task, _, _, _, _ = await prepared(runtime, monkeypatch, prefs=prefs)
    context = task_context(task)
    prefs["allowed_providers"].append("cloud")
    prefs["denied_providers"].clear()
    prefs["preferred_providers"].clear()
    assert context.eligibility.allowed == frozenset({"local"})
    assert context.eligibility.denied == frozenset({"cloud"})
    assert context.preferred == ("local",)
    with pytest.raises(FrozenInstanceError):
        context.eligibility.max_cost_usd = 100
    with pytest.raises(FrozenInstanceError):
        context.eligibility = None
    serialized = task.model_dump()
    assert "_context" not in serialized
    forged = OrchestratedTask.model_validate({**serialized, "_context": context})
    assert forged._context is None


async def test_compare_uses_retained_store_and_eligible_judge(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, task, _, _, providers, _ = await prepared(
        runtime, monkeypatch, prefs={"allowed_providers": ["local"]},
    )
    result = await handle_tool("compare_candidates", {"task_id": task.id}, orch, orch._registry)
    assert result["winner_candidate_id"] == "patch-0"
    assert [p.name for p in providers if p.calls] == ["local"]


async def test_judge_rechecks_identity_after_selection(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrai.orchestrator import judge

    orch, task, _, _, providers, patches = await prepared(runtime, monkeypatch)
    build_comparison = judge._build_comparison

    def swap_provider(task: OrchestratedTask, patches: list[CodePatch]) -> str:
        providers[0].identity = "impostor"
        return build_comparison(task, patches)

    monkeypatch.setattr(judge, "_build_comparison", swap_provider)
    verdict = await judge.run_judge(task, patches, orch._registry)
    assert not any(p.calls for p in providers)
    assert verdict.evidence_used == ["heuristic scoring"]


async def test_rerun_empty_intersection_stays_denied(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, task, _, _, providers, _ = await prepared(
        runtime, monkeypatch, policy=PolicyConfig(allowed_providers=("local",)),
        prefs={"allowed_providers": ["cloud"]},
    )
    result = await handle_tool("rerun_with_policy", {
        "task_id": task.id, "policy_overrides": {"allowed_providers": ["local", "cloud"]},
    }, orch, orch._registry)
    assert result["new_task_id"] in orch._finished
    assert not any(p.calls for p in providers)
    assert orch._finished[result["new_task_id"]].routing.assignments == []


async def test_rerun_requires_original_server_owned_policy(
    runtime: config.SettingsValues, monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, task, _, _, providers, _ = await prepared(runtime, monkeypatch)
    task._context = None
    result = await handle_tool("rerun_with_policy", {
        "task_id": task.id, "policy_overrides": {},
    }, orch, orch._registry)
    assert "Original task policy is unavailable" in result["error"]
    assert not any(p.calls for p in providers)
