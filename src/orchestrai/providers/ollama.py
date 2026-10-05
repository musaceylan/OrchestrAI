"""
Ollama native provider — uses Ollama's /api/* endpoints directly.

Advantages over the generic OpenAI-compat adapter:
- Accurate context_length from /api/show (each model varies)
- Richer model metadata (family, parameter size, quantization)
- Health check via /api/version (more reliable than /v1/models)
- No OpenAI SDK dependency for completions
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

_DEFAULT_BASE_URL = "http://localhost:11434"
_PROBE_TIMEOUT = 5.0
_COMPLETION_TIMEOUT = 120.0
_SHOW_TIMEOUT = 8.0


def _coding_strength(model_name: str) -> float:
    """Estimate coding strength from the Ollama model name."""
    mn = model_name.lower()
    if any(k in mn for k in ("deepseek-coder", "codestral", "qwen2.5-coder", "codellama")):
        return 0.90
    if any(k in mn for k in ("70b", "72b")):
        return 0.83
    if any(k in mn for k in ("qwen3", "llama3.3", "llama3.2", "mistral-nemo")):
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


class OllamaProvider(BaseProvider):
    """
    Ollama provider using the native REST API.

    Completions use POST /api/chat.
    Model discovery uses GET /api/tags + GET /api/show for context lengths.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, name: str = "ollama") -> None:
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
        """Check liveness via GET /api/version."""
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                r = await client.get(f"{self._base_url}/api/version")
                if r.status_code == 200:
                    version = r.json().get("version", "unknown")
                    self._models = await self._discover_models()
                    log.info(
                        "ollama.probe",
                        available=True,
                        version=version,
                        models=[m.model_id for m in self._models],
                    )
                    return True
        except (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError) as e:
            log.info("ollama.probe", available=False, error=str(e))
        except Exception as e:
            log.warning("ollama.probe", available=False, error=str(e))
        return False

    async def list_models(self) -> list[ModelCapability]:
        return list(self._models)

    async def _discover_models(self) -> list[ModelCapability]:
        """Fetch model list from /api/tags, enrich each with /api/show."""
        try:
            async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                r.raise_for_status()
                tag_list: list[dict] = r.json().get("models", [])
        except Exception as e:
            log.warning("ollama.list_models.error", error=str(e))
            return []

        caps: list[ModelCapability] = []
        is_loopback = LocalProviderEndpoint(base_url=self._base_url).is_loopback
        # Fetch context lengths in parallel (best-effort; ignore failures)
        ctx_map = await self._batch_context_lengths([m.get("name", "") for m in tag_list])

        for entry in tag_list:
            model_name: str = entry.get("name", "")
            if not model_name:
                continue
            coding = _coding_strength(model_name)
            ctx = ctx_map.get(model_name, 8192)
            details = entry.get("details", {})
            param_size = details.get("parameter_size", "")

            caps.append(
                ModelCapability(
                    provider=self._name,
                    provider_kind=ProviderKind.OPENAI_COMPAT,
                    model_id=model_name,
                    display_name=f"Ollama/{model_name}" + (f" ({param_size})" if param_size else ""),
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

    async def _batch_context_lengths(self, model_names: list[str]) -> dict[str, int]:
        """
        Fetch context_length for each model via GET /api/show.
        Returns a dict model_name → context_length (defaults to 8192 on error).
        """
        async def _show(name: str) -> tuple[str, int]:
            try:
                async with httpx.AsyncClient(timeout=_SHOW_TIMEOUT) as client:
                    r = await client.post(
                        f"{self._base_url}/api/show",
                        json={"name": name},
                    )
                    if r.status_code == 200:
                        info = r.json()
                        ctx = (
                            info.get("model_info", {}).get("llama.context_length")
                            or info.get("details", {}).get("context_length")
                            or 8192
                        )
                        return name, int(ctx)
            except Exception:
                pass
            return name, 8192

        results = await asyncio.gather(*[_show(n) for n in model_names], return_exceptions=True)
        out: dict[str, int] = {}
        for item in results:
            if isinstance(item, tuple):
                out[item[0]] = item[1]
        return out

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """POST /api/chat — Ollama native completion API."""
        model = request.model or (self._models[0].model_id if self._models else "llama3.2")
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages)

        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=_COMPLETION_TIMEOUT) as client:
                r = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()

            content = data.get("message", {}).get("content", "")
            input_tokens = data.get("prompt_eval_count", 0)
            output_tokens = data.get("eval_count", 0)
            return CompletionResponse(
                content=content,
                model=model,
                provider=self._name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason="stop",
                raw_response=data,
            )
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"Ollama HTTP {e.response.status_code}: {e.response.text[:200]}",
                provider=self._name,
                model=model,
                retryable=e.response.status_code >= 500,
            ) from e
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ProviderError(str(e), provider=self._name, model=model, retryable=True) from e
        except Exception as e:
            raise ProviderError(str(e), provider=self._name, model=model, retryable=False) from e
