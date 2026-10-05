"""Machine-loopback trust boundaries for compatible provider endpoints."""

from typing import Any

import pytest
import respx

from orchestrai.artifacts.schemas import PrivacyLevel, TaskType
from orchestrai.config.settings import LocalProviderEndpoint, PolicyConfig
from orchestrai.providers.openai_compat import OpenAICompatProvider
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.registry.router import RoutingEngine


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434",
        "https://LOCALHOST:443/api",
        "http://models.localhost:8000",
        "http://a.b.localhost",
        "http://localhost.",
        "http://MODELS.LOCALHOST.:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.0",
        "http://127.42.7.9",
        "http://127.255.255.255",
        "http://[::1]:8000",
        "https://[0:0:0:0:0:0:0:1]",
    ],
)
def test_machine_loopback_endpoints(base_url: str) -> None:
    assert LocalProviderEndpoint(base_url=base_url).is_loopback is True


@pytest.mark.parametrize(
    "base_url",
    [
        "https://models.example.com",
        "https://8.8.8.8",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://192.168.1.10",
        "http://169.254.1.1",
        "http://0.0.0.0",
        "http://126.255.255.255",
        "http://128.0.0.0",
        "http://[2001:4860:4860::8888]",
        "http://[fd00::1]",
        "http://[fe80::1]",
        "http://[::]",
        "http://localhost.example.com",
        "http://notlocalhost",
        "http://localhost@models.example.com",
        "http://127.0.0.1@models.example.com",
        "http://models.example.com/path/localhost",
        "",
        "not a URL",
        "localhost:8000",
        "//localhost:8000",
        "http:///localhost",
        "http://:8000",
        "ftp://localhost",
        "http://[::1",
        "http://[not-an-ip]",
        "http://localhost:bad",
        "http://127.0.0.1:65536",
        "http://[::1]:-1",
        " http://localhost",
        "http://local\nhost",
        "http://local\rhost",
        "http://local\thost",
        "http://\x00.localhost",
        "http://bad name.localhost",
        "http://bad\\name.localhost",
        "http://.localhost",
        "http://a..localhost",
        "http://-model.localhost",
        "http://model-.localhost",
        "http://localhost..",
        "http://[::1]suffix",
        "http://[127.0.0.1]",
        "http://127.1",
        "http://2130706433",
        "http://127.0.0.999",
    ],
)
def test_non_loopback_or_malformed_endpoints(base_url: str) -> None:
    assert LocalProviderEndpoint(base_url=base_url).is_loopback is False


def test_loopback_classification_is_read_only() -> None:
    endpoint = LocalProviderEndpoint(base_url="https://models.example.com")
    with pytest.raises(AttributeError):
        object.__setattr__(endpoint, "is_loopback", True)
    assert endpoint.is_loopback is False


@pytest.mark.parametrize(
    ("base_url", "privacy_level"),
    [
        ("http://localhost:8000", PrivacyLevel.SECRET),
        ("http://models.localhost:8000", PrivacyLevel.SECRET),
        ("http://127.42.7.9:8000", PrivacyLevel.SECRET),
        ("http://[::1]:8000", PrivacyLevel.SECRET),
        ("http://192.168.1.10:8000", PrivacyLevel.PUBLIC),
        ("https://8.8.8.8", PrivacyLevel.PUBLIC),
    ],
)
async def test_discovered_capabilities_follow_endpoint_trust(
    base_url: str, privacy_level: PrivacyLevel
) -> None:
    provider = OpenAICompatProvider(LocalProviderEndpoint(name="endpoint", base_url=base_url))
    with respx.mock as mock:
        mock.get(f"{base_url}/v1/models").respond(
            200, json={"data": [{"id": "qwen2.5-coder:7b"}, {"id": "llama3.2:8b"}]}
        )
        assert await provider.probe()

    capabilities = await provider.list_models()
    assert len(capabilities) == 2
    assert all(cap.privacy_level is privacy_level for cap in capabilities)


@pytest.mark.parametrize(
    ("remote_url", "loopback_url"),
    [
        ("https://models.example.com", "http://localhost:8000"),
        ("http://192.168.1.10:8000", "http://127.42.7.9:8000"),
        ("http://[fd00::1]:8000", "http://[::1]:8000"),
    ],
)
@pytest.mark.parametrize(
    ("policy_options", "preferences"),
    [
        pytest.param({"local_only_mode": True}, {}, id="policy-local-only"),
        pytest.param({}, {"local_only": True}, id="preference-local-only"),
        pytest.param({"privacy_level": "secret"}, {}, id="secret"),
    ],
)
async def test_private_routing_keeps_only_loopback_compatible_capabilities(
    remote_url: str,
    loopback_url: str,
    policy_options: dict[str, Any],
    preferences: dict[str, Any],
) -> None:
    remote = OpenAICompatProvider(LocalProviderEndpoint(name="remote", base_url=remote_url))
    loopback = OpenAICompatProvider(LocalProviderEndpoint(name="loopback", base_url=loopback_url))
    with respx.mock as mock:
        mock.get(f"{remote_url}/v1/models").respond(
            200, json={"data": [{"id": "qwen2.5-coder:7b"}]}
        )
        mock.get(f"{loopback_url}/v1/models").respond(
            200, json={"data": [{"id": "tinyllama:7b"}]}
        )
        assert await remote.probe()
        assert await loopback.probe()

    registry = await CapabilityRegistry.build([remote, loopback])
    public_decision = RoutingEngine(registry, PolicyConfig(privacy_level="public")).route(
        task_type=TaskType.FEATURE, mode="impl_tester", task_id="public"
    )
    # The stronger remote model is available when public routing permits it.
    assert public_decision.assignments[0]["provider"] == "remote"

    policy = PolicyConfig(**policy_options)
    private_decision = RoutingEngine(registry, policy).route(
        task_type=TaskType.FEATURE,
        mode="impl_tester",
        task_id="private",
        user_preferences=preferences,
    )
    assert len(private_decision.assignments) == 3
    assert {a["provider"] for a in private_decision.assignments} == {"loopback"}
    assert {a["privacy_level"] for a in private_decision.assignments} == {"secret"}

    remote_registry = await CapabilityRegistry.build([remote])
    remote_only_decision = RoutingEngine(remote_registry, policy).route(
        task_type=TaskType.FEATURE,
        mode="impl_tester",
        task_id="remote-only",
        user_preferences=preferences,
    )
    assert remote_only_decision.assignments == []
