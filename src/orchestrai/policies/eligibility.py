"""Immutable admission constraints. Requests can intersect policy, never replace it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from orchestrai.artifacts.schemas import CostTier, PrivacyLevel, ProviderKind
from orchestrai.config.settings import PolicyConfig
from orchestrai.providers.base import ModelCapability

PRIVACY_ORDER = tuple(PrivacyLevel)
COST_ORDER = (CostTier.CHEAP, CostTier.MEDIUM, CostTier.EXPENSIVE)


def _providers(preferences: Mapping[str, Any], key: str) -> frozenset[str] | None:
    value = preferences.get(key)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"Invalid {key}")
    return frozenset(value)


@dataclass(frozen=True)
class Eligibility:
    privacy: PrivacyLevel = PrivacyLevel.PUBLIC
    local_only: bool = False
    allowed: frozenset[str] | None = None
    denied: frozenset[str] = frozenset()
    cost_max: CostTier | None = None
    max_cost_usd: float | None = None

    @classmethod
    def resolve(
        cls, policy: PolicyConfig, preferences: Mapping[str, Any] | None = None,
    ) -> Eligibility:
        return cls(
            privacy=PrivacyLevel(policy.privacy_level),
            local_only=policy.local_only_mode,
            # Config has historically used [] for unrestricted. An explicit
            # request [] (or an empty intersection) instead means deny all.
            allowed=frozenset(policy.allowed_providers) if policy.allowed_providers else None,
            denied=frozenset(policy.denied_providers),
            max_cost_usd=policy.max_cost_usd,
        ).tighten(preferences)

    def tighten(self, preferences: Mapping[str, Any] | None) -> Eligibility:
        prefs = preferences or {}
        local = prefs.get("local_only", False)
        if not isinstance(local, bool):
            raise ValueError("Invalid local_only")
        privacy = PrivacyLevel(prefs.get("privacy_level") or self.privacy)
        allowed = _providers(prefs, "allowed_providers")
        if self.allowed is not None:
            allowed = self.allowed if allowed is None else self.allowed & allowed
        costs = [self.cost_max] if self.cost_max is not None else []
        costs.extend(CostTier(prefs[key]) for key in ("cost_max", "cost_tier") if prefs.get(key))
        budget = prefs.get("max_cost_usd")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not isfinite(budget) or budget < 0
        ):
            raise ValueError("Invalid max_cost_usd")
        budgets = [value for value in (self.max_cost_usd, budget) if value is not None]
        return replace(
            self,
            privacy=max((self.privacy, privacy), key=PRIVACY_ORDER.index),
            local_only=self.local_only or local,
            allowed=allowed,
            denied=self.denied | (_providers(prefs, "denied_providers") or frozenset()),
            cost_max=min(costs, key=COST_ORDER.index) if costs else None,
            max_cost_usd=min(budgets) if budgets else None,
        )

    def intersect(self, other: Eligibility) -> Eligibility:
        """Retain the original task ceiling when a new admin policy is resolved."""
        return self.tighten({
            "privacy_level": other.privacy,
            "local_only": other.local_only,
            "allowed_providers": tuple(other.allowed) if other.allowed is not None else None,
            "denied_providers": tuple(other.denied),
            "cost_max": other.cost_max,
            "max_cost_usd": other.max_cost_usd,
        })

    def allows(self, cap: ModelCapability) -> bool:
        return (
            cap.available
            and (self.allowed is None or cap.provider in self.allowed)
            and cap.provider not in self.denied
            and PRIVACY_ORDER.index(cap.privacy_level) >= PRIVACY_ORDER.index(self.privacy)
            and (self.cost_max is None
                 or COST_ORDER.index(cap.cost_tier) <= COST_ORDER.index(self.cost_max))
            and (not self.local_only or (
                cap.provider_kind == ProviderKind.OPENAI_COMPAT
                and cap.privacy_level == PrivacyLevel.SECRET
            ))
        )


def preferred_providers(preferences: Mapping[str, Any] | None) -> tuple[str, ...]:
    prefs = preferences or {}
    _providers(prefs, "preferred_providers")  # validate, retaining order below
    return tuple(prefs.get("preferred_providers") or ())
