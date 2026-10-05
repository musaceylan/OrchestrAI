"""Privacy boundaries for generic OpenAI-compatible endpoints."""

import respx

from orchestrai.artifacts.schemas import PrivacyLevel
from orchestrai.config.settings import LocalProviderEndpoint
from orchestrai.providers.openai_compat import OpenAICompatProvider


async def test_remote_endpoint_capabilities_are_public() -> None:
    provider = OpenAICompatProvider(
        LocalProviderEndpoint(name="remote", base_url="https://models.example.com")
    )
    with respx.mock as mock:
        mock.get("https://models.example.com/v1/models").respond(
            200, json={"data": [{"id": "qwen2.5-coder:7b"}]}
        )
        assert await provider.probe()

    capabilities = await provider.list_models()
    assert len(capabilities) == 1
    assert capabilities[0].privacy_level is PrivacyLevel.PUBLIC
