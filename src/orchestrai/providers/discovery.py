"""
Provider auto-discovery: probe all configured providers in parallel,
build the available provider list, fail gracefully on missing credentials
or unreachable local endpoints.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from orchestrai.config.settings import get_settings
from orchestrai.providers.anthropic import AnthropicProvider
from orchestrai.providers.base import BaseProvider
from orchestrai.providers.gemini import GeminiProvider
from orchestrai.providers.ollama import OllamaProvider
from orchestrai.providers.openai import OpenAIProvider
from orchestrai.providers.openai_compat import OpenAICompatProvider
from orchestrai.providers.vllm import VLLMProvider

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


async def discover_providers() -> list[BaseProvider]:
    """
    Probe all configured providers in parallel.
    Returns only the ones that are available.
    Never raises — bad providers are logged and skipped.
    """
    settings = get_settings()
    candidates: list[BaseProvider] = [
        AnthropicProvider(),
        OpenAIProvider(),
        GeminiProvider(),
        # Ollama — native provider (probes /api/version; skipped silently if not running)
        OllamaProvider(),
        # vLLM — OpenAI-compat local inference (probes /health or /v1/models)
        VLLMProvider(),
    ]

    # Add additional local providers from config (generic OpenAI-compat endpoints)
    for lp in settings.local_providers:
        # Skip defaults handled by native providers above
        if lp.name == "ollama" and lp.base_url == "http://localhost:11434":
            continue
        if lp.name == "vllm" and lp.base_url == "http://localhost:8000":
            continue
        candidates.append(OpenAICompatProvider(lp))

    # Probe all in parallel
    results = await asyncio.gather(
        *[_probe_one(p) for p in candidates],
        return_exceptions=True,
    )

    available: list[BaseProvider] = []
    for provider, result in zip(candidates, results):
        if isinstance(result, Exception):
            log.error(
                "provider.discovery.error",
                provider=provider.name,
                error=str(result),
            )
        elif result is True:
            available.append(provider)
        else:
            log.info("provider.discovery.unavailable", provider=provider.name)

    log.info(
        "provider.discovery.complete",
        available=[p.name for p in available],
        total=len(candidates),
    )
    return available


async def _probe_one(provider: BaseProvider) -> bool:
    try:
        return await asyncio.wait_for(provider.probe(), timeout=10.0)
    except asyncio.TimeoutError:
        log.warning("provider.probe.timeout", provider=provider.name)
        return False
    except Exception as e:
        log.warning("provider.probe.error", provider=provider.name, error=str(e))
        return False
