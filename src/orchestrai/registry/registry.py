"""
Capability registry — single source of truth for what models are available
and what they're good at. Populated at startup via discovery, queryable by the router.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import structlog

from orchestrai.artifacts.schemas import CostTier, PrivacyLevel, RoleType
from orchestrai.policies.eligibility import Eligibility
from orchestrai.providers.base import (
    BaseProvider,
    ModelCapability,
    format_model_reference,
    parse_model_reference,
    validate_provider_names,
)

log = structlog.get_logger()


class CapabilityRegistry:
    """
    Holds all discovered providers and their model capabilities.
    Builds privately; callers publish the complete replacement after validation.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._capabilities: dict[tuple[str, str], ModelCapability] = {}

    @classmethod
    async def build(cls, providers: list[BaseProvider]) -> CapabilityRegistry:
        """Reject invalid inventories without publishing or mutating existing state."""
        names = [provider.name for provider in providers]
        validate_provider_names(names)
        provider_map = dict(zip(names, providers, strict=True))
        capabilities: dict[tuple[str, str], ModelCapability] = {}
        for name, provider in provider_map.items():
            models = await provider.list_models()
            for cap in models:
                if cap.provider != name:
                    raise ValueError("Capability provider does not match its provider")
                if not cap.model_id.strip():
                    raise ValueError("Model ID must not be empty")
                identity = (name, cap.model_id)
                if identity in capabilities:
                    raise ValueError("Duplicate model identity")
                capabilities[identity] = replace(
                    cap,
                    available=True,
                    role_strengths=dict(cap.role_strengths),
                    preferred_roles=list(cap.preferred_roles),
                )
            log.info("registry.loaded", provider=name, models=len(models))

        registry = cls()
        registry._providers = provider_map
        registry._capabilities = capabilities
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
        return self._capabilities.get((provider, model_id))

    def resolve_model_reference(self, reference: str) -> ModelCapability:
        """Registered provider prefixes are authoritative; otherwise require a unique bare ID."""
        provider, model_id = parse_model_reference(reference)
        if provider is not None and provider in self._providers:
            cap = self.get_capability(provider, model_id)
        else:
            matches = [cap for cap in self.all_capabilities() if cap.model_id == reference]
            if len(matches) > 1:
                raise ValueError("Ambiguous model reference; qualify it with a provider")
            cap = matches[0] if matches else None
        if cap is None:
            raise ValueError("Model reference does not exist")
        return cap

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
        eligibility = Eligibility(
            privacy=privacy_max or PrivacyLevel.PUBLIC,
            cost_max=cost_max,
            allowed=frozenset(provider_allowlist) if provider_allowlist else None,
            denied=frozenset(provider_denylist or ()),
        )
        results = [
            cap for cap in self.available_capabilities()
            if cap.strength_for_role(role) >= min_strength and eligibility.allows(cap)
        ]

        results.sort(key=lambda c: c.strength_for_role(role), reverse=True)
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "providers": [p.name for p in self._providers.values()],
            "models": {
                format_model_reference(*identity): cap.to_dict()
                for identity, cap in self._capabilities.items()
            },
            "total_available": len(self.available_capabilities()),
        }
