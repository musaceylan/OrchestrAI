"""
OpenAI-compatible adapter — Ollama, vLLM, LM Studio, llama.cpp server, etc.
Any endpoint that speaks the OpenAI chat completions API.
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from orchestrai.artifacts.schemas import (
    CostTier, LatencyTier, PrivacyLevel, ProviderKind, RoleType,
)
from orchestrai.config.settings import LocalProviderEndpoint
from orchestrai.providers.base import (
    BaseProvider, CompletionRequest, CompletionResponse,
    ModelCapability, ProviderError,
)

log = structlog.get_logger()


class OpenAICompatProvider(BaseProvider):
    """
    Wraps any OpenAI-compatible local endpoint.
    Auto-discovers available models via /v1/models.
    """

    def __init__(self, cfg: LocalProviderEndpoint) -> None:
        self._cfg = cfg
        self._discovered_models: list[ModelCapability] = []

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.OPENAI_COMPAT

    async def probe(self) -> bool:
        if not self._cfg.enabled:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._cfg.probe_timeout) as client:
                r = await client.get(f"{self._cfg.base_url}/v1/models")
                if r.status_code == 200:
                    models_data = r.json().get("data", [])
                    self._discovered_models = self._build_capabilities(models_data)
                    log.info(
                        "local.probe",
                        provider=self.name,
                        available=True,
                        models=[m.model_id for m in self._discovered_models],
                    )
                    return True
        except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            log.info(
                "local.probe",
                provider=self.name,
                available=False,
                error=str(e),
            )
        except Exception as e:
            log.warning("local.probe", provider=self.name, available=False, error=str(e))
        return False

    def _build_capabilities(self, models_data: list[dict]) -> list[ModelCapability]:
        caps = []
        for m in models_data:
            model_id = m.get("id", "unknown")
            # Heuristic capability estimation based on model name patterns
            coding = self._estimate_coding(model_id)
            caps.append(
                ModelCapability(
                    provider=self.name,
                    provider_kind=ProviderKind.OPENAI_COMPAT,
                    model_id=model_id,
                    display_name=f"{self.name}/{model_id}",
                    planning_strength=coding * 0.85,
                    coding_strength=coding,
                    debugging_strength=coding * 0.85,
                    review_strength=coding * 0.80,
                    test_gen_strength=coding * 0.80,
                    docs_strength=coding * 0.82,
                    long_context_strength=0.70,
                    context_window=self._estimate_context(model_id),
                    max_output_tokens=4096,
                    latency_tier=LatencyTier.FAST,  # local = fast
                    cost_tier=CostTier.CHEAP,        # local = free
                    privacy_level=PrivacyLevel.SECRET,  # local = most private
                    supports_tool_calling=False,
                    supports_structured_output=False,
                    supports_streaming=True,
                    preferred_roles=[RoleType.CODER, RoleType.TESTER],
                )
            )
        return caps

    def _estimate_coding(self, model_id: str) -> float:
        """Estimate coding strength from model name heuristics."""
        mid = model_id.lower()
        if any(k in mid for k in ["deepseek-coder", "codestral", "qwen2.5-coder", "codellama"]):
            return 0.88
        if any(k in mid for k in ["qwen3", "llama3.3", "llama3.2", "mistral-nemo"]):
            return 0.78
        if any(k in mid for k in ["70b", "72b"]):
            return 0.82
        if any(k in mid for k in ["7b", "8b"]):
            return 0.65
        return 0.72

    def _estimate_context(self, model_id: str) -> int:
        mid = model_id.lower()
        if "128k" in mid:
            return 131_072
        if "32k" in mid:
            return 32_768
        if "16k" in mid:
            return 16_384
        return 8_192

    async def list_models(self) -> list[ModelCapability]:
        return list(self._discovered_models)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        import openai

        model = request.model or (
            self._discovered_models[0].model_id if self._discovered_models else "default"
        )
        client = openai.AsyncOpenAI(
            api_key="local",
            base_url=f"{self._cfg.base_url}/v1",
            timeout=120.0,
        )
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
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
        except Exception as e:
            raise ProviderError(str(e), self.name, model, retryable=True) from e
