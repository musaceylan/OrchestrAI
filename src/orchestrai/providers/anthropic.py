"""Anthropic / Claude provider adapter."""
from __future__ import annotations

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from orchestrai.artifacts.schemas import (
    CostTier, LatencyTier, PrivacyLevel, ProviderKind, RoleType,
)
from orchestrai.config.settings import get_settings
from orchestrai.providers.base import (
    BaseProvider, CompletionRequest, CompletionResponse,
    ModelCapability, ProviderError,
)

log = structlog.get_logger()

# Declarative model registry — update as new models ship
ANTHROPIC_MODELS: list[ModelCapability] = [
    ModelCapability(
        provider="anthropic",
        provider_kind=ProviderKind.ANTHROPIC,
        model_id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        planning_strength=0.97,
        coding_strength=0.95,
        debugging_strength=0.95,
        review_strength=0.97,
        test_gen_strength=0.93,
        docs_strength=0.95,
        long_context_strength=0.90,
        context_window=200_000,
        max_output_tokens=32_000,
        latency_tier=LatencyTier.SLOW,
        cost_tier=CostTier.EXPENSIVE,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.PLANNER, RoleType.REVIEWER, RoleType.JUDGE],
    ),
    ModelCapability(
        provider="anthropic",
        provider_kind=ProviderKind.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        planning_strength=0.90,
        coding_strength=0.93,
        debugging_strength=0.91,
        review_strength=0.90,
        test_gen_strength=0.90,
        docs_strength=0.90,
        long_context_strength=0.88,
        context_window=200_000,
        max_output_tokens=16_000,
        latency_tier=LatencyTier.MEDIUM,
        cost_tier=CostTier.MEDIUM,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.CODER, RoleType.DEBUGGER, RoleType.REFACTOR],
    ),
    ModelCapability(
        provider="anthropic",
        provider_kind=ProviderKind.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        planning_strength=0.75,
        coding_strength=0.80,
        debugging_strength=0.75,
        review_strength=0.75,
        test_gen_strength=0.78,
        docs_strength=0.80,
        long_context_strength=0.72,
        context_window=200_000,
        max_output_tokens=8_000,
        latency_tier=LatencyTier.FAST,
        cost_tier=CostTier.CHEAP,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.TESTER, RoleType.DOCUMENTER],
    ),
]


class AnthropicProvider(BaseProvider):
    def __init__(self) -> None:
        self._cfg = get_settings().anthropic
        self._client: "anthropic.AsyncAnthropic | None" = None  # lazy init

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.ANTHROPIC

    def _get_client(self) -> "anthropic.AsyncAnthropic":
        if self._client is None:
            import anthropic
            import httpx
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=100,
                    keepalive_expiry=30.0,
                ),
                timeout=httpx.Timeout(self._cfg.timeout),
            )
            kwargs: dict = {"api_key": self._cfg.api_key, "http_client": http_client}
            if self._cfg.base_url:
                kwargs["base_url"] = self._cfg.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def probe(self) -> bool:
        if not self._cfg.api_key:
            log.info("anthropic.probe", available=False, reason="no api key")
            return False
        try:
            import anthropic
            client = self._get_client()
            await client.models.list()
            log.info("anthropic.probe", available=True)
            return True
        except Exception as e:
            log.warning("anthropic.probe", available=False, error=str(e))
            return False

    async def list_models(self) -> list[ModelCapability]:
        return [m for m in ANTHROPIC_MODELS]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        import anthropic

        model = request.model or self._cfg.default_model
        client = self._get_client()

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=min(request.max_tokens, self._cfg.max_tokens),
                system=request.system or "You are a world-class software engineer.",
                messages=request.messages,
                temperature=request.temperature,
            )
            content = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )
            return CompletionResponse(
                content=content,
                model=model,
                provider=self.name,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason or "stop",
                raw_response=response,
            )
        except anthropic.AuthenticationError as e:
            raise ProviderError(str(e), self.name, model, retryable=False) from e
        except anthropic.RateLimitError as e:
            raise ProviderError(str(e), self.name, model, retryable=True) from e
        except anthropic.APIError as e:
            raise ProviderError(str(e), self.name, model, retryable=True) from e
