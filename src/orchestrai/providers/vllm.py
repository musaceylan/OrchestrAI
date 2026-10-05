"""
vLLM native provider — uses vLLM's OpenAI-compatible REST API.

vLLM exposes:
  GET  /v1/models               — list models with max_model_len in each entry
  POST /v1/chat/completions     — standard OpenAI chat completions

Advantages over the generic OpenAICompatProvider:
  - Context window discovered per-model from max_model_len field
  - Always tagged privacy=SECRET (runs locally)
  - Named "vllm" in the registry for policy filtering
  - Liveness check via /health (vLLM >= 0.4) with /v1/models fallback
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from orchestrai.artifacts.schemas import (
    CostTier,
    LatencyTier,
    PrivacyLevel,
    ProviderKind,
    RoleType,
)
from orchestrai.config.settings import LocalProviderEndpoint
from orchestrai.providers.base import (
    BaseProvider,
    CompletionRequest,
    CompletionResponse,
    ModelCapability,
    ProviderError,
)

log = structlog.get_logger()

_DEFAULT_BASE_URL = "http://localhost:8000"
_PROBE_TIMEOUT = 5.0
_COMPLETION_TIMEOUT = 120.0


def _coding_strength(model_name: str) -> float:
    """Estimate coding strength from the model name / path."""
    mn = model_name.lower().split("/")[-1]  # strip HF org prefix
    if any(k in mn for k in ("deepseek-coder", "codestral", "qwen2.5-coder", "codellama")):
        return 0.90
    if any(k in mn for k in ("70b", "72b")):
        return 0.83
    if any(k in mn for k in ("qwen3", "llama-3.3", "llama-3.2", "mistral-nemo")):
        return 0.78
    if any(k in mn for k in ("30b", "32b", "34b")):
        return 0.78
    if any(k in mn for k in ("13b", "14b")):
        return 0.72
    if any(k in mn for k in ("7b", "8b")):
        return 0.66
    if any(k in mn for k in ("3b",)):
        return 0.58
    return 0.70


class VLLMProvider(BaseProvider):
    """
    vLLM provider using the OpenAI-compatible REST API.

    Completions use POST /v1/chat/completions.
    Model discovery uses GET /v1/models (includes max_model_len per model).
    Liveness check uses GET /health with /v1/models fallback.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, name: str = "vllm") -> None:
        self._base_url = base_url.rstrip("/")
        self._name = name
        self._models: list[ModelCapability] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> ProviderKind:
        return ProviderKind.OPENAI_COMPAT

    async def probe(self) -> bool:
        """Check liveness: try GET /health first, fall back to GET /v1/models."""
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                # vLLM >= 0.4 exposes /health; older versions don't
                for path in ("/health", "/v1/models"):
                    try:
                        r = await client.get(f"{self._base_url}{path}")
                        if r.status_code == 200:
                            self._models = await self._discover_models()
                            log.info(
                                "vllm.probe",
                                available=True,
                                endpoint=self._base_url,
                                models=[m.model_id for m in self._models],
                            )
                            return bool(self._models)
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue
        except (httpx.ConnectError, asyncio.TimeoutError) as e:
            log.info("vllm.probe", available=False, error=str(e))
        except Exception as e:
            log.warning("vllm.probe", available=False, error=str(e))
        return False

    async def list_models(self) -> list[ModelCapability]:
        return list(self._models)

    async def _discover_models(self) -> list[ModelCapability]:
        """Fetch model list from GET /v1/models."""
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                r = await client.get(f"{self._base_url}/v1/models")
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("vllm.list_models.error", error=str(e))
            return []

        caps: list[ModelCapability] = []
        is_loopback = LocalProviderEndpoint(base_url=self._base_url).is_loopback
        for entry in data.get("data", []):
            model_id: str = entry.get("id", "")
            if not model_id:
                continue
            # vLLM includes max_model_len in the model object
            ctx = int(entry.get("max_model_len") or entry.get("context_window") or 4096)
            coding = _coding_strength(model_id)
            caps.append(
                ModelCapability(
                    provider=self._name,
                    provider_kind=ProviderKind.OPENAI_COMPAT,
                    model_id=model_id,
                    display_name=f"vLLM/{model_id.split('/')[-1]}",
                    planning_strength=coding * 0.85,
                    coding_strength=coding,
                    debugging_strength=coding * 0.85,
                    review_strength=coding * 0.80,
                    test_gen_strength=coding * 0.80,
                    docs_strength=coding * 0.82,
                    long_context_strength=min(ctx / 131_072, 1.0),
                    context_window=ctx,
                    max_output_tokens=min(ctx // 4, 8192),
                    latency_tier=LatencyTier.FAST,
                    cost_tier=CostTier.CHEAP,
                    privacy_level=(
                        PrivacyLevel.SECRET if is_loopback else PrivacyLevel.PUBLIC
                    ),
                    supports_tool_calling=False,
                    supports_streaming=True,
                    preferred_roles=[RoleType.CODER, RoleType.TESTER],
                )
            )
        return caps

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """POST /v1/chat/completions — OpenAI-compatible completion."""
        model = request.model or (self._models[0].model_id if self._models else "")
        if not model:
            raise ProviderError(
                "No model available on vLLM endpoint",
                provider=self._name,
                model="",
                retryable=False,
            )

        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=_COMPLETION_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()

            choice = data.get("choices", [{}])[0]
            content: str = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return CompletionResponse(
                content=content,
                model=model,
                provider=self._name,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=data,
            )
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"vLLM HTTP {e.response.status_code}: {e.response.text[:200]}",
                provider=self._name,
                model=model,
                retryable=e.response.status_code >= 500,
            ) from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderError(str(e), provider=self._name, model=model, retryable=True) from e
        except Exception as e:
            raise ProviderError(str(e), provider=self._name, model=model, retryable=False) from e
