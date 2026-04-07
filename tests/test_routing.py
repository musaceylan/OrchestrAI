"""Unit tests for the routing engine."""
from __future__ import annotations

import pytest

from orchestrai.artifacts.schemas import (
    CostTier,
    LatencyTier,
    PrivacyLevel,
    RoleType,
    TaskType,
)
from orchestrai.config.settings import PolicyConfig
from orchestrai.providers.base import ModelCapability
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.registry.router import RoutingEngine


def _make_cap(
    model_id: str,
    provider: str,
    strengths: dict[RoleType, float],
    cost_tier: CostTier = CostTier.MEDIUM,
    privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL,
) -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        provider=provider,
        context_window=128_000,
        role_strengths=strengths,
        cost_tier=cost_tier,
        latency_tier=LatencyTier.MEDIUM,
        privacy_level=privacy_level,
    )


@pytest.fixture
def caps():
    return [
        _make_cap(
            "claude-opus-4-6", "anthropic",
            {RoleType.PLANNER: 0.98, RoleType.REVIEWER: 0.97, RoleType.JUDGE: 0.96},
            CostTier.EXPENSIVE,
        ),
        _make_cap(
            "claude-sonnet-4-6", "anthropic",
            {RoleType.CODER: 0.95, RoleType.PLANNER: 0.88, RoleType.TESTER: 0.87},
            CostTier.MEDIUM,
        ),
        _make_cap(
            "gpt-4.1", "openai",
            {RoleType.CODER: 0.93, RoleType.TESTER: 0.89, RoleType.REVIEWER: 0.88},
            CostTier.MEDIUM,
        ),
        _make_cap(
            "ollama-codellama", "ollama-local",
            {RoleType.CODER: 0.72},
            CostTier.CHEAP,
            PrivacyLevel.SECRET,
        ),
    ]


@pytest.fixture
def registry(caps):
    r = CapabilityRegistry()
    for cap in caps:
        cap.available = True
        r._capabilities[f"{cap.provider}/{cap.model_id}"] = cap
    r._providers = {}  # no real providers needed for routing tests
    return r


@pytest.fixture
def policy():
    return PolicyConfig()


def test_route_assigns_best_planner(registry, policy):
    engine = RoutingEngine(registry, policy)
    decision = engine.route(
        task_type=TaskType.FEATURE,
        mode="planner_coder_reviewer",
        task_id="t1",
    )
    planner = next((a for a in decision.assignments if a["role"] == "planner"), None)
    assert planner is not None
    assert planner["provider"] == "anthropic"
    assert planner["model"] == "claude-opus-4-6"


def test_route_assigns_best_coder(registry, policy):
    engine = RoutingEngine(registry, policy)
    decision = engine.route(
        task_type=TaskType.FEATURE,
        mode="planner_coder_reviewer",
        task_id="t2",
    )
    coder = next((a for a in decision.assignments if a["role"] == "coder"), None)
    assert coder is not None
    assert coder["model"] in ("claude-sonnet-4-6", "gpt-4.1")


def test_privacy_filter_excludes_cloud_for_secret(registry, policy):
    policy_secret = PolicyConfig(privacy_level="secret")
    engine = RoutingEngine(registry, policy_secret)
    decision = engine.route(
        task_type=TaskType.BUGFIX,
        mode="impl_tester",
        task_id="t3",
    )
    for assignment in decision.assignments:
        assert assignment["provider"] == "ollama-local", (
            f"Expected local provider, got {assignment['provider']}"
        )


def test_recommended_mode_mapping():
    from orchestrai.registry.router import TASK_MODE_MAP
    assert TASK_MODE_MAP[TaskType.BUGFIX] == "bugfix"
    assert TASK_MODE_MAP[TaskType.TEST_GENERATION] == "impl_tester"


def test_registry_capabilities_for_role(registry):
    caps = registry.capabilities_for_role(RoleType.CODER)
    assert len(caps) >= 1
    # Should be sorted by strength descending
    strengths = [c.role_strengths.get(RoleType.CODER, 0) for c in caps]
    assert strengths == sorted(strengths, reverse=True)


def test_registry_to_dict(registry):
    d = registry.to_dict()
    assert "providers" in d
    assert "total_available" in d
    assert d["total_available"] == 4
