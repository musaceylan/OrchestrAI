"""Google Gemini provider adapter."""
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

GEMINI_MODELS: list[ModelCapability] = [
    ModelCapability(
        provider="gemini",
        provider_kind=ProviderKind.GEMINI,
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        planning_strength=0.92,
        coding_strength=0.90,
        debugging_strength=0.88,
        review_strength=0.88,
        test_gen_strength=0.87,
        docs_strength=0.88,
        long_context_strength=0.98,   # 1M+ context window
        context_window=1_000_000,
        max_output_tokens=65_536,
        latency_tier=LatencyTier.SLOW,
        cost_tier=CostTier.EXPENSIVE,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.ANALYZER, RoleType.RESEARCHER],
    ),
    ModelCapability(
        provider="gemini",
        provider_kind=ProviderKind.GEMINI,
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        planning_strength=0.82,
        coding_strength=0.85,
        debugging_strength=0.80,
        review_strength=0.80,
        test_gen_strength=0.82,
        docs_strength=0.82,
        long_context_strength=0.85,
        context_window=1_000_000,
        max_output_tokens=8_192,
        latency_tier=LatencyTier.FAST,
        cost_tier=CostTier.CHEAP,
        privacy_level=PrivacyLevel.INTERNAL,
        supports_tool_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        preferred_roles=[RoleType.TESTER, RoleType.DOCUMENTER],
    ),
]


class GeminiProvider(BaseProvider):
    def __init__(self) -> None:
        self._cfg = get_settings().gemini

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.GEMINI

    async def probe(self) -> bool:
        if not self._cfg.api_key:
            log.info("gemini.probe", available=False, reason="no api key")
            return False
        try:
            import asyncio
            import google.generativeai as genai
            genai.configure(api_key=self._cfg.api_key)
            # list_models() is synchronous — run in thread to avoid blocking the event loop
            await asyncio.to_thread(list, genai.list_models())
            log.info("gemini.probe", available=True)
            return True
        except Exception as e:
            log.warning("gemini.probe", available=False, error=str(e))
            return False

    async def list_models(self) -> list[ModelCapability]:
        return list(GEMINI_MODELS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._cfg.api_key)

            model_id = request.model or self._cfg.default_model
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=request.system or None,
            )

            # Build prompt from messages
            prompt_parts = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt = "\n\n".join(prompt_parts)

            resp = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=request.max_tokens,
                    temperature=request.temperature,
                ),
            )
            content = resp.text or ""
            usage = resp.usage_metadata
            return CompletionResponse(
                content=content,
                model=model_id,
                provider=self.name,
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
                finish_reason="stop",
                raw_response=resp,
            )
        except Exception as e:
            raise ProviderError(str(e), self.name, request.model, retryable=True) from e
