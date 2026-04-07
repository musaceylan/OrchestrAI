"""
Abstract provider interface — every backend implements this contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from orchestrai.artifacts.schemas import (
    CostTier,
    LatencyTier,
    PrivacyLevel,
    ProviderKind,
    RoleType,
)


@dataclass
class ModelCapability:
    """Declarative capability descriptor for a model."""
    provider: str
    provider_kind: ProviderKind
    model_id: str
    display_name: str

    # Role suitability (0.0–1.0)
    planning_strength: float = 0.5
    coding_strength: float = 0.5
    debugging_strength: float = 0.5
    review_strength: float = 0.5
    test_gen_strength: float = 0.5
    docs_strength: float = 0.5
    long_context_strength: float = 0.5

    # Operational properties
    context_window: int = 8192
    max_output_tokens: int = 4096
    latency_tier: LatencyTier = LatencyTier.MEDIUM
    cost_tier: CostTier = CostTier.MEDIUM
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC

    # Feature flags
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = True

    # Runtime state
    available: bool = True
    error_message: str | None = None

    # Role routing: which roles is this model best for
    preferred_roles: list[RoleType] = field(default_factory=list)

    def strength_for_role(self, role: RoleType) -> float:
        mapping = {
            RoleType.PLANNER: self.planning_strength,
            RoleType.CODER: self.coding_strength,
            RoleType.DEBUGGER: self.debugging_strength,
            RoleType.REVIEWER: self.review_strength,
            RoleType.TESTER: self.test_gen_strength,
            RoleType.DOCUMENTER: self.docs_strength,
            RoleType.ANALYZER: (self.debugging_strength + self.long_context_strength) / 2,
            RoleType.JUDGE: (self.review_strength + self.planning_strength) / 2,
            RoleType.REFACTOR: (self.coding_strength + self.review_strength) / 2,
            RoleType.RESEARCHER: (self.long_context_strength + self.planning_strength) / 2,
        }
        return mapping.get(role, 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_kind": self.provider_kind.value,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "latency_tier": self.latency_tier.value,
            "cost_tier": self.cost_tier.value,
            "privacy_level": self.privacy_level.value,
            "supports_tool_calling": self.supports_tool_calling,
            "supports_structured_output": self.supports_structured_output,
            "supports_streaming": self.supports_streaming,
            "available": self.available,
            "error_message": self.error_message,
            "strengths": {
                "planning": self.planning_strength,
                "coding": self.coding_strength,
                "debugging": self.debugging_strength,
                "review": self.review_strength,
                "test_gen": self.test_gen_strength,
                "docs": self.docs_strength,
                "long_context": self.long_context_strength,
            },
            "preferred_roles": [r.value for r in self.preferred_roles],
        }


@dataclass
class CompletionRequest:
    """Unified request structure for any provider."""
    messages: list[dict[str, Any]]
    system: str = ""
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    role: RoleType = RoleType.CODER
    task_id: str = ""
    subtask_id: str = ""
    agent_run_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Unified response structure from any provider."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseProvider(ABC):
    """Abstract base class for all AI providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name, e.g. 'anthropic', 'openai', 'ollama'."""
        ...

    @property
    @abstractmethod
    def kind(self) -> ProviderKind:
        """Provider family classification."""
        ...

    @abstractmethod
    async def probe(self) -> bool:
        """
        Test connectivity and credentials.
        Returns True if provider is available, False otherwise.
        Must not raise — handle exceptions internally.
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelCapability]:
        """
        Return all available models with capability metadata.
        Called once at startup and on refresh.
        """
        ...

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Execute a completion request.
        Must raise ProviderError on failure.
        """
        ...

    async def stream(
        self, request: CompletionRequest
    ) -> AsyncIterator[str]:
        """
        Optional streaming completion. Default falls back to complete().
        Yields text deltas.
        """
        response = await self.complete(request)
        yield response.content


class ProviderError(Exception):
    """Raised when a provider fails to complete a request."""
    def __init__(
        self,
        message: str,
        provider: str,
        model: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retryable = retryable
