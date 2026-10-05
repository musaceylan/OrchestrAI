"""Server-owned task admission, ranking and the last check before provider dispatch."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import fsum, isfinite
from threading import Lock
from typing import TYPE_CHECKING, Any

from orchestrai.artifacts.schemas import OrchestratedTask, PrivacyLevel, ProviderKind, RoleType
from orchestrai.config.settings import PolicyConfig, get_settings
from orchestrai.policies.costs import estimate_cost
from orchestrai.policies.eligibility import Eligibility, preferred_providers
from orchestrai.providers.base import CompletionRequest, CompletionResponse, ModelCapability

if TYPE_CHECKING:
    from orchestrai.registry.registry import CapabilityRegistry


class EligibilityError(ValueError):
    """A stale identity or policy denial; never retry or transmit."""


def _maximum_request_cost(cap: ModelCapability, request: CompletionRequest) -> float:
    """Reserve a full input context and the requested output at the existing rates."""
    if any(type(value) is not int or value <= 0 for value in (
        cap.context_window, cap.max_output_tokens, request.max_tokens,
    )) or request.max_tokens > cap.max_output_tokens:
        raise EligibilityError("Cannot bound request token cost")
    if estimate_cost(cap.model_id, 1, 1) == 0 and not (
        cap.provider_kind == ProviderKind.OPENAI_COMPAT
        and cap.privacy_level == PrivacyLevel.SECRET
    ):
        raise EligibilityError("Unknown cloud pricing under task cost ceiling")
    try:
        maximum = estimate_cost(cap.model_id, cap.context_window, request.max_tokens)
    except OverflowError as exc:
        raise EligibilityError("Cannot bound request token cost") from exc
    if not isfinite(maximum) or maximum < 0:
        raise EligibilityError("Cannot bound request token cost")
    return maximum


@dataclass
class _Budget:
    lock: Lock = field(default_factory=Lock)
    reservations: dict[object, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskContext:
    eligibility: Eligibility
    preferred: tuple[str, ...] = ()
    registry_supplier: Callable[[], CapabilityRegistry] | None = field(
        default=None, repr=False, compare=False,
    )
    _budget: _Budget = field(default_factory=_Budget, repr=False, compare=False)

    @classmethod
    def resolve(
        cls, policy: PolicyConfig, preferences: Mapping[str, Any] | None = None,
        original: TaskContext | None = None,
        *, registry_supplier: Callable[[], CapabilityRegistry] | None = None,
    ) -> TaskContext:
        eligibility = Eligibility.resolve(policy, preferences)
        if original is not None:
            eligibility = eligibility.intersect(original.eligibility)
        preferred = preferred_providers(preferences)
        if original is not None and "preferred_providers" not in (preferences or {}):
            preferred = original.preferred
        return cls(eligibility, preferred, registry_supplier)

    def current_registry(self, fallback: CapabilityRegistry) -> CapabilityRegistry:
        return self.registry_supplier() if self.registry_supplier is not None else fallback

    def candidates(self, registry: CapabilityRegistry, role: RoleType) -> list[ModelCapability]:
        registry = self.current_registry(registry)
        caps = [cap for cap in registry.available_capabilities() if self.eligibility.allows(cap)]
        return sorted(caps, key=lambda cap: (
            self.preferred.index(cap.provider)
            if cap.provider in self.preferred else len(self.preferred),
            -cap.strength_for_role(role),
        ))

    async def complete(
        self, task: OrchestratedTask, registry: CapabilityRegistry,
        provider_name: str, request: CompletionRequest,
    ) -> CompletionResponse:
        """Re-resolve both identities on every attempt, with no intervening await."""
        registry = self.current_registry(registry)
        cap = registry.get_capability(provider_name, request.model or "")
        provider = registry.get_provider(provider_name)
        if (
            cap is None or provider is None or provider.name != provider_name
            or cap.provider != provider_name or cap.model_id != request.model
            or cap.provider_kind != provider.kind or not self.eligibility.allows(cap)
        ):
            raise EligibilityError("Provider/model identity is not eligible under task policy")
        ceiling = self.eligibility.max_cost_usd
        reservation = object()
        maximum = _maximum_request_cost(cap, request) if ceiling is not None else 0.0
        with self._budget.lock:
            if ceiling is not None:
                if not isfinite(task.cost_usd) or task.cost_usd < 0:
                    raise EligibilityError("Task cost accounting must be finite and non-negative")
                committed = fsum((task.cost_usd, *self._budget.reservations.values(), maximum))
                if committed > ceiling:
                    raise EligibilityError("Task policy cost ceiling reached")
                self._budget.reservations[reservation] = maximum

        # Never hold the budget lock across provider I/O. Failed/cancelled calls
        # retain their reservation: without valid usage, their charge is unknown.
        response = await provider.complete(request)
        if response.provider != provider_name or response.model != request.model:
            raise EligibilityError("Response provider/model identity does not match request")
        if any(type(value) is not int or value < 0 for value in (
            response.input_tokens, response.output_tokens,
        )):
            raise EligibilityError("Response token accounting must be non-negative integers")
        try:
            cost = estimate_cost(response.model, response.input_tokens, response.output_tokens)
        except OverflowError as exc:
            raise EligibilityError("Response cost accounting is not finite") from exc
        if not isfinite(cost) or cost < 0:
            raise EligibilityError("Response cost accounting must be finite and non-negative")
        if ceiling is not None and cost > maximum:
            raise EligibilityError("Response exceeds reserved request cost")
        key = f"{provider_name}/{response.model}"
        with self._budget.lock:
            total_cost = task.cost_usd + cost
            if not isfinite(total_cost) or total_cost < 0:
                raise EligibilityError("Task cost accounting must be finite and non-negative")
            self._budget.reservations.pop(reservation, None)
            task.cost_usd = total_cost
            task.tokens_used[key] = task.tokens_used.get(key, 0) + response.total_tokens
        return response


def task_context(task: OrchestratedTask) -> TaskContext:
    """Bind direct internal callers once; request dictionaries cannot supply this state."""
    if task._context is None:
        task._context = TaskContext.resolve(get_settings().policy)
    return task._context
