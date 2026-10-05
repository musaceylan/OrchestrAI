"""Provider identity, atomic registry construction, and judge selection regressions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from orchestrai.artifacts.schemas import (
    CodePatch,
    OrchestratedTask,
    Provenance,
    ProviderKind,
    RoleType,
    TaskBrief,
    TaskType,
)
from orchestrai.config import settings as config
from orchestrai.orchestrator.judge import run_judge
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.providers import discovery
from orchestrai.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelCapability,
)
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.tools import _probe_providers, _reload_config


class StubProvider(BaseProvider):
    def __init__(self, name: str, *model_ids: str, strength: float = 0.5) -> None:
        self._name = name
        self.models = [
            ModelCapability(
                provider=name, model_id=model, role_strengths={RoleType.JUDGE: strength}
            )
            for model in model_ids
        ]
        self.requests: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.UNKNOWN

    async def probe(self) -> bool:
        return True

    async def list_models(self) -> list[ModelCapability]:
        return self.models

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        return CompletionResponse(
            content='{"winner_index": 1, "rationale": "Selected by stub", "scores": {"1": 0.9}}',
            provider=self.name,
            model=request.model or "",
        )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    with patch.dict("os.environ", {}, clear=True):
        monkeypatch.chdir(tmp_path)
        yield


async def test_duplicate_provider_names_are_rejected() -> None:
    providers = [StubProvider("local", "first"), StubProvider("local", "second")]
    with pytest.raises(ValueError, match="[Dd]uplicate provider"):
        await CapabilityRegistry.build(list(providers))


async def test_duplicate_model_ids_within_provider_are_rejected() -> None:
    with pytest.raises(ValueError, match="[Dd]uplicate.*(model|identity)"):
        await CapabilityRegistry.build([StubProvider("local", "model", "model")])


@pytest.mark.parametrize("name", ["", " ", "local/org", " local", "local ", "lo cal", "lo\tcal"])
async def test_empty_or_ambiguous_provider_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="[Pp]rovider"):
        await CapabilityRegistry.build([StubProvider(name, "model")])


@pytest.mark.parametrize("model_id", ["", " ", "\t\n"])
async def test_empty_model_ids_are_rejected(model_id: str) -> None:
    with pytest.raises(ValueError, match="[Mm]odel"):
        await CapabilityRegistry.build([StubProvider("local", model_id)])


@pytest.mark.parametrize("spoofed_name", ["trusted", "unknown", ""])
async def test_capability_provider_mismatch_is_rejected(spoofed_name: str) -> None:
    trusted = StubProvider("trusted", "model")
    attacker = StubProvider("attacker", "model")
    attacker.models[0].provider = spoofed_name
    with pytest.raises(ValueError, match="[Pp]rovider"):
        await CapabilityRegistry.build([trusted, attacker])


async def test_slash_model_ids_are_preserved_without_lookup_collisions() -> None:
    model_ids = ("Llama-3.3-70B", "meta-llama/Llama-3.3-70B", "org/team/model")
    registry = await CapabilityRegistry.build([
        StubProvider("gpu", *model_ids), StubProvider("backup", *model_ids),
    ])
    assert len(registry.all_capabilities()) == 6
    for provider in ("gpu", "backup"):
        for model_id in model_ids:
            cap = registry.get_capability(provider, model_id)
            assert cap is not None
            assert (cap.provider, cap.model_id) == (provider, model_id)
    assert registry.get_capability("gpu/meta-llama", "Llama-3.3-70B") is None
    assert registry.get_capability("gpu/org", "team/model") is None
    serialized = json.loads(json.dumps(registry.to_dict()))
    assert set(serialized["models"]) == {
        f"{provider}/{model_id}" for provider in ("gpu", "backup") for model_id in model_ids
    }
    assert serialized["models"]["gpu/org/team/model"]["model_id"] == "org/team/model"


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, asyncio.CancelledError])
async def test_failed_build_does_not_return_partial_state_or_mutate_source(
    failure: type[BaseException],
) -> None:
    first = StubProvider("first", "model")
    first.models[0].available = False
    broken = StubProvider("broken")
    with (
        patch.object(broken, "list_models", AsyncMock(side_effect=failure)),
        pytest.raises(failure),
    ):
        await CapabilityRegistry.build([first, broken])
    assert first.models[0].available is False


async def test_successful_build_detaches_mutable_capability_fields() -> None:
    provider = StubProvider("local", "model", strength=0.7)
    source = provider.models[0]
    source.preferred_roles.append(RoleType.JUDGE)

    registry = await CapabilityRegistry.build([provider])
    published = registry.get_capability("local", "model")
    assert published is not None

    source.role_strengths[RoleType.JUDGE] = 0.01
    source.preferred_roles.append(RoleType.CODER)
    assert published.role_strengths[RoleType.JUDGE] == 0.7
    assert published.preferred_roles == [RoleType.JUDGE]

    published.role_strengths[RoleType.JUDGE] = 0.9
    published.preferred_roles.append(RoleType.REVIEWER)
    assert source.role_strengths[RoleType.JUDGE] == 0.01
    assert source.preferred_roles == [RoleType.JUDGE, RoleType.CODER]


@pytest.mark.parametrize("action", ["probe", "reload"])
@pytest.mark.parametrize("failure", ["duplicate-provider", "duplicate-model", "spoof", "load"])
async def test_identity_failure_retains_published_runtime(action: str, failure: str) -> None:
    from orchestrai.server import main

    provider = StubProvider("original", "model")
    registry = await CapabilityRegistry.build([provider])
    cap = registry.get_capability("original", "model")
    assert cap is not None
    cap.available = False
    provider.models = [cap]
    orch = Orchestrator(registry)
    settings = config.get_settings()
    snapshot = registry.to_dict()
    broken = StubProvider("broken", "model")
    error: type[Exception] = ValueError
    if failure == "duplicate-provider":
        broken = StubProvider("original", "other")
    elif failure == "duplicate-model":
        broken = StubProvider("broken", "model", "model")
    elif failure == "spoof":
        broken.models[0].provider = "original"
    else:
        error = RuntimeError
    list_models = (
        AsyncMock(side_effect=error) if failure == "load" else AsyncMock(return_value=broken.models)
    )

    with (
        patch.object(main, "_orchestrator", orch),
        patch.object(main, "_registry", registry),
        patch.object(discovery, "discover_providers", AsyncMock(return_value=[provider, broken])),
        patch.object(broken, "list_models", list_models),
    ):
        if action == "probe":
            with pytest.raises(error):
                await _probe_providers({}, orch, registry)
        else:
            assert "error" in await _reload_config({}, orch, registry)
        assert main._registry is registry
        assert orch._registry is orch._router._registry is registry
        assert config.get_settings() is orch._settings is settings
        assert orch._router._policy is settings.policy
        assert registry.to_dict() == snapshot


@pytest.mark.parametrize(
    "names",
    [("custom", "custom"), ("openai",), ("ollama", "ollama"), ("vllm/team",), ("",)],
)
async def test_discovery_rejects_identity_collisions_before_probing(
    monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...],
) -> None:
    endpoints = tuple(config.LocalProviderEndpoint(name=name) for name in names)
    monkeypatch.setattr(
        discovery, "get_settings", lambda: SimpleNamespace(local_providers=endpoints)
    )
    for constructor, name in (
        ("AnthropicProvider", "anthropic"), ("OpenAIProvider", "openai"),
        ("GeminiProvider", "gemini"), ("OllamaProvider", "ollama"), ("VLLMProvider", "vllm"),
    ):
        monkeypatch.setattr(discovery, constructor, lambda name=name: StubProvider(name))
    monkeypatch.setattr(
        discovery, "OpenAICompatProvider", lambda endpoint: StubProvider(endpoint.name)
    )
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(discovery, "_probe_one", probe)
    with pytest.raises(ValueError, match="[Pp]rovider"):
        await discovery.discover_providers()
    probe.assert_not_awaited()


@pytest.fixture
def judge_inputs() -> tuple[OrchestratedTask, list[CodePatch]]:
    provenance = Provenance(task_id="identity-test")
    task = OrchestratedTask(
        id="identity-test", trace_id="identity-trace", mode="parallel_draft",
        brief=TaskBrief(
            id="brief", provenance=provenance, task_type=TaskType.FEATURE,
            description="Compare two implementations",
        ),
    )
    candidates = [CodePatch(id=f"candidate-{i}", provenance=provenance) for i in range(2)]
    return task, candidates


@pytest.mark.parametrize("reference", ["beta/shared", "beta/org/team/model", "unique"])
async def test_judge_override_selects_exact_identity(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]], reference: str,
) -> None:
    alpha = StubProvider("alpha", "shared", "org/team/model", strength=0.9)
    beta = StubProvider("beta", "shared", "org/team/model", "unique", strength=0.1)
    registry = await CapabilityRegistry.build([alpha, beta])
    verdict = await run_judge(*judge_inputs, registry, judge_model_override=reference)
    assert not alpha.requests
    assert len(beta.requests) == 1
    assert beta.requests[0].model == reference.removeprefix("beta/")
    assert verdict.winner_candidate_id == "candidate-1"
    assert verdict.evidence_used == ["judge model comparison"]


@pytest.mark.parametrize(
    "reference", ["shared", "missing", "missing/model", "beta/missing", "", " "]
)
@pytest.mark.parametrize("candidate_count", [0, 1, 2])
async def test_invalid_judge_override_fails_closed_without_provider_calls(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]], reference: str, candidate_count: int,
) -> None:
    providers = [StubProvider("alpha", "shared"), StubProvider("beta", "shared")]
    registry = await CapabilityRegistry.build(list(providers))
    task, candidates = judge_inputs
    with pytest.raises(ValueError, match="[Jj]udge|[Mm]odel|[Pp]rovider"):
        await run_judge(
            task, candidates[:candidate_count], registry, judge_model_override=reference
        )
    assert all(not provider.requests for provider in providers)


@pytest.mark.parametrize("reference", ["shared", "alpha/shared", "missing"])
async def test_override_does_not_fall_back_when_models_are_unavailable(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]], reference: str,
) -> None:
    provider = StubProvider("alpha", "shared")
    registry = await CapabilityRegistry.build([provider])
    for cap in registry.all_capabilities():
        cap.available = False
    with pytest.raises(ValueError, match="[Jj]udge|[Mm]odel"):
        await run_judge(*judge_inputs, registry, judge_model_override=reference)
    assert not provider.requests


async def test_bare_override_is_ambiguous_even_if_one_identity_is_unavailable(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]],
) -> None:
    providers = [StubProvider("alpha", "shared"), StubProvider("beta", "shared")]
    registry = await CapabilityRegistry.build(list(providers))
    cap = registry.get_capability("beta", "shared")
    assert cap is not None
    cap.available = False
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        await run_judge(*judge_inputs, registry, judge_model_override="shared")
    assert all(not provider.requests for provider in providers)


async def test_no_override_preserves_automatic_and_heuristic_selection(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]],
) -> None:
    alpha = StubProvider("alpha", "model", strength=0.9)
    beta = StubProvider("beta", "model", strength=0.1)
    registry = await CapabilityRegistry.build([beta, alpha])
    verdict = await run_judge(*judge_inputs, registry)
    assert len(alpha.requests) == 1
    assert not beta.requests
    assert verdict.evidence_used == ["judge model comparison"]
    heuristic = await run_judge(*judge_inputs, await CapabilityRegistry.build([]))
    assert heuristic.evidence_used == ["heuristic scoring"]


async def test_unique_bare_slash_model_override_remains_compatible(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]],
) -> None:
    alpha = StubProvider("alpha", "other", strength=0.9)
    gpu = StubProvider("gpu", "meta-llama/Llama-3.3-70B", strength=0.1)
    registry = await CapabilityRegistry.build([alpha, gpu])
    verdict = await run_judge(
        *judge_inputs, registry, judge_model_override="meta-llama/Llama-3.3-70B"
    )
    assert not alpha.requests
    assert len(gpu.requests) == 1
    assert gpu.requests[0].model == "meta-llama/Llama-3.3-70B"
    assert verdict.winner_candidate_id == "candidate-1"


async def test_ambiguous_bare_slash_model_override_is_rejected(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]],
) -> None:
    providers = [
        StubProvider("alpha", "meta-llama/Llama-3.3-70B"),
        StubProvider("beta", "meta-llama/Llama-3.3-70B"),
    ]
    registry = await CapabilityRegistry.build(list(providers))
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        await run_judge(*judge_inputs, registry, judge_model_override="meta-llama/Llama-3.3-70B")
    assert all(not provider.requests for provider in providers)


async def test_registered_provider_prefix_never_falls_back_to_bare_model(
    judge_inputs: tuple[OrchestratedTask, list[CodePatch]],
) -> None:
    providers = [StubProvider("alpha", "other"), StubProvider("beta", "alpha/missing")]
    registry = await CapabilityRegistry.build(list(providers))
    with pytest.raises(ValueError, match="[Mm]odel"):
        await run_judge(*judge_inputs, registry, judge_model_override="alpha/missing")
    assert all(not provider.requests for provider in providers)
