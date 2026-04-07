"""
Capability registry — single source of truth for what models are available
and what they're good at. Populated at startup via discovery, queryable by the router.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from orchestrai.providers.base import BaseProvider, ModelCapability
from orchestrai.artifacts.schemas import ProviderKind, RoleType, PrivacyLevel, CostTier

log = structlog.get_logger()


class CapabilityRegistry:
    """
    Holds all discovered providers and their model capabilities.
    Thread-safe (asyncio) — designed for concurrent reads, infrequent writes.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._capabilities: dict[str, ModelCapability] = {}  # key: "provider/model_id"
        self._lock = asyncio.Lock()

    @classmethod
    async def build(cls, providers: list[BaseProvider]) -> "CapabilityRegistry":
        registry = cls()
        async with registry._lock:
            for provider in providers:
                registry._providers[provider.name] = provider
                try:
                    models = await provider.list_models()
                    for cap in models:
                        cap.available = True
                        registry._capabilities[f"{cap.provider}/{cap.model_id}"] = cap
                    log.info(
                        "registry.loaded",
                        provider=provider.name,
                        models=len(models),
                    )
                except Exception as e:
                    log.error(
                        "registry.load_error", provider=provider.name, error=str(e)
                    )
        return registry

    def get_provider(self, name: str) -> BaseProvider | None:
        return self._providers.get(name)

    def all_providers(self) -> list[BaseProvider]:
        return list(self._providers.values())

    def all_capabilities(self) -> list[ModelCapability]:
        return list(self._capabilities.values())

    def available_capabilities(self) -> list[ModelCapability]:
        return [c for c in self._capabilities.values() if c.available]

    def get_capability(self, provider: str, model_id: str) -> ModelCapability | None:
        return self._capabilities.get(f"{provider}/{model_id}")

    def capabilities_for_role(
        self,
        role: RoleType,
        min_strength: float = 0.0,
        privacy_max: PrivacyLevel | None = None,
        cost_max: CostTier | None = None,
        provider_allowlist: list[str] | None = None,
        provider_denylist: list[str] | None = None,
    ) -> list[ModelCapability]:
        """
        Return models suitable for a role, filtered by policy constraints,
        sorted by role strength descending.
        """
        _privacy_order = {
            PrivacyLevel.PUBLIC: 0,
            PrivacyLevel.INTERNAL: 1,
            PrivacyLevel.CONFIDENTIAL: 2,
            PrivacyLevel.SECRET: 3,
        }
        _cost_order = {
            CostTier.CHEAP: 0,
            CostTier.MEDIUM: 1,
            CostTier.EXPENSIVE: 2,
        }

        results = []
        for cap in self.available_capabilities():
            if cap.strength_for_role(role) < min_strength:
                continue
            if provider_allowlist and cap.provider not in provider_allowlist:
                continue
            if provider_denylist and cap.provider in provider_denylist:
                continue
            if privacy_max is not None:
                # Model must be at least as private as required
                if _privacy_order[cap.privacy_level] < _privacy_order[privacy_max]:
                    continue
            if cost_max is not None:
                if _cost_order[cap.cost_tier] > _cost_order[cost_max]:
                    continue
            results.append(cap)

        results.sort(key=lambda c: c.strength_for_role(role), reverse=True)
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "providers": [p.name for p in self._providers.values()],
            "models": {k: v.to_dict() for k, v in self._capabilities.items()},
            "total_available": len(self.available_capabilities()),
        }
