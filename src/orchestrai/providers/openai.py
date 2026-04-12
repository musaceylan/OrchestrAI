"""OpenAI / Codex provider adapter."""
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

OPENAI_MODELS: list[ModelCapability] = [
    ModelCapability(
        provider="openai",
        provider_kind=ProviderKind.OPENAI,
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        planning_strength=0.90,
        coding_strength=0.92,
        debugging_strength=0.88,
        review_strength=0.90,
        test_gen_strength=0.88,
        docs_strength=0.87,
        long_context_strength=0.85,
        context_window=1_000_000,
        max_output_tokens=32_768,
        latency_tier=LatencyTier.MEDIUM,
        cost_tier=CostTier.MEDIUM,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.CODER, RoleType.REFACTOR],
    ),
    ModelCapability(
        provider="openai",
        provider_kind=ProviderKind.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        planning_strength=0.87,
        coding_strength=0.88,
        debugging_strength=0.85,
        review_strength=0.87,
        test_gen_strength=0.85,
        docs_strength=0.85,
        long_context_strength=0.82,
        context_window=128_000,
        max_output_tokens=16_384,
        latency_tier=LatencyTier.FAST,
        cost_tier=CostTier.MEDIUM,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.CODER, RoleType.TESTER],
    ),
    ModelCapability(
        provider="openai",
        provider_kind=ProviderKind.OPENAI,
        model_id="o4-mini",
        display_name="o4-mini (Reasoning)",
        planning_strength=0.95,
        coding_strength=0.88,
        debugging_strength=0.93,
        review_strength=0.90,
        test_gen_strength=0.85,
        docs_strength=0.82,
        long_context_strength=0.88,
        context_window=200_000,
        max_output_tokens=100_000,
        latency_tier=LatencyTier.SLOW,
        cost_tier=CostTier.MEDIUM,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.PLANNER, RoleType.DEBUGGER, RoleType.ANALYZER],
    ),
]


class OpenAIProvider(BaseProvider):
    def __init__(self) -> None:
        self._cfg = get_settings().openai
        self._client: "openai.AsyncOpenAI | None" = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.OPENAI

    def _get_client(self) -> "openai.AsyncOpenAI":
        if self._client is None:
            import openai
            import httpx
            self._client = openai.AsyncOpenAI(
                api_key=self._cfg.api_key,
                base_url=self._cfg.base_url,
                timeout=self._cfg.timeout,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=100,
                        keepalive_expiry=30.0,
                    ),
                    timeout=httpx.Timeout(self._cfg.timeout),
                ),
            )
        return self._client

    async def probe(self) -> bool:
        if not self._cfg.api_key:
            log.info("openai.probe", available=False, reason="no api key")
            return False
        try:
            client = self._get_client()
            await client.models.list()
            log.info("openai.probe", available=True)
            return True
        except Exception as e:
            log.warning("openai.probe", available=False, error=str(e))
            return False

    async def list_models(self) -> list[ModelCapability]:
        return list(OPENAI_MODELS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        import openai

        model = request.model or self._cfg.default_model
        client = self._get_client()
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        try:
            # Reasoning models (o-series) do not accept temperature parameter
            _is_reasoning = model.startswith("o") and model[1:2].isdigit()
            create_kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
            }
            if not _is_reasoning:
                create_kwargs["temperature"] = request.temperature
            resp = await client.chat.completions.create(**create_kwargs)
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            return CompletionResponse(
                content=content,
                model=model,
                provider=self.name,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                finish_reason=resp.choices[0].finish_reason or "stop",
                raw_response=resp,
            )
        except openai.AuthenticationError as e:
            raise ProviderError(str(e), self.name, model, retryable=False) from e
        except openai.RateLimitError as e:
            raise ProviderError(str(e), self.name, model, retryable=True) from e
        except openai.APIError as e:
            raise ProviderError(str(e), self.name, model, retryable=True) from e
