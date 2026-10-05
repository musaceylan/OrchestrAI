"""Fail-closed configuration and transactional reload regressions."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from orchestrai.config import settings as config
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.tools import _probe_providers, _reload_config

SHIPPED = Path(__file__).resolve().parents[1] / "config/default.yaml"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    # Do not consume host environment credentials or dotenv files in these tests.
    with patch.dict("os.environ", {}, clear=True):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(config.Settings.model_config, "env_file", None)
        monkeypatch.setattr(config, "_settings", None)
        yield


def write_config(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_shipped_loads(monkeypatch):
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(SHIPPED))
    loaded = config.Settings.load()
    assert loaded.openai.default_model == "gpt-4.1"
    assert loaded.policy.privacy_level == "public"


def test_missing_default_uses_typed_defaults():
    loaded = config.Settings.load()
    assert loaded.policy.privacy_level == "public"
    assert loaded.server.transport == "stdio"


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda settings: setattr(settings.server, "port", 9000), ValidationError),
        (lambda settings: settings.policy.allowed_providers.append("openai"), AttributeError),
        (lambda settings: settings.local_providers.append("not-a-provider"), AttributeError),
        (lambda settings: setattr(settings, "_config_source", "tampered"), AttributeError),
    ],
    ids=[
        "nested-scalar",
        "policy-collection",
        "provider-collection",
        "config-source",
    ],
)
def test_published_settings_are_deeply_immutable(mutate, error):
    settings = config.Settings.load()
    config.publish_settings(settings)

    with pytest.raises(error):
        mutate(config.get_settings())


@pytest.mark.parametrize("source_kind", ["argument", "environment", "implicit", "defaults"])
async def test_reload_reports_resolved_yaml_source(tmp_path, monkeypatch, source_kind):
    explicit_settings = None
    if source_kind == "implicit":
        path = tmp_path / "config" / "default.yaml"
        path.parent.mkdir()
        path.write_text("{}")
    elif source_kind == "defaults":
        path = None
    else:
        path = write_config(tmp_path, "{}")
        if source_kind == "environment":
            monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
        else:
            explicit_settings = config.Settings.load(path)

    registry = CapabilityRegistry()
    orchestrator = Orchestrator(registry)
    candidate_registry = CapabilityRegistry()
    reload_patch = (
        patch("orchestrai.config.settings.reload_settings", return_value=explicit_settings)
        if explicit_settings is not None
        else patch("orchestrai.config.settings.reload_settings", wraps=config.reload_settings)
    )
    with (
        reload_patch,
        patch("orchestrai.providers.discovery.discover_providers", AsyncMock(return_value=[])),
        patch.object(CapabilityRegistry, "build", AsyncMock(return_value=candidate_registry)),
    ):
        result = await _reload_config({}, orchestrator, registry)

    assert result["config_path"] == (str(path.resolve()) if path is not None else "defaults")


@pytest.mark.parametrize("explicit", ["argument", "environment"])
@pytest.mark.parametrize(
    "content",
    [
        None,
        "[broken",
        "[]",
        "null",
        "",
        "42",
        "unexpected: true",
        "policy: {unexpected: true}",
        "server: {port: invalid}",
        "policy: {privacy_level: invalid}",
        "server: {port: 0}",
        "orchestrator: {max_parallel_agents: -1}",
        "policy: {local_only_mode: true, local_only_mode: false}",
        "local_providers: [{name: test, unexpected: true}]",
        "policy: &policy {allowed_providers: [*policy]}",
        "policy: {max_cost_usd: .nan}",
        "policy: {max_cost_usd: -1}",
        "server: {transport: invalid}",
        "orchestrator: {default_mode: invalid}",
    ],
)
def test_explicit_errors_fail_closed(tmp_path, monkeypatch, explicit, content):
    path = tmp_path / "config.yaml"
    if content is not None:
        path.write_text(content)
    if explicit == "environment":
        monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    with pytest.raises(ValueError, match="Invalid configuration") as error:
        if explicit == "argument":
            config.Settings.load(path)
        else:
            config.Settings.load()
    assert "invalid" not in str(error.value)
    assert error.value.__suppress_context__


def test_unreadable_is_safe(tmp_path, monkeypatch):
    path = write_config(tmp_path, "{}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    with (
        patch("pathlib.Path.open", side_effect=PermissionError("untrusted detail")),
        pytest.raises(ValueError, match="Invalid configuration") as error,
    ):
        config.Settings.load()
    assert "untrusted detail" not in str(error.value)


@pytest.mark.parametrize(
    "content",
    ["#" * (1024 * 1024 + 1), "[" * 100 + "]" * 100, "policy: &p {allowed_providers: *p}"],
    ids=["bytes", "depth", "alias"],
)
def test_yaml_bounds(tmp_path, monkeypatch, content):
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(write_config(tmp_path, content)))
    with pytest.raises(ValueError, match="Invalid configuration"):
        config.Settings.load()


def test_environment_precedence(tmp_path, monkeypatch):
    path = write_config(tmp_path, "openai: {default_model: yaml}\nserver: {port: 8001}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "provider-environment")
    monkeypatch.setenv("ORCHESTRAI__OPENAI__DEFAULT_MODEL", "nested-environment")
    monkeypatch.setenv("ORCHESTRAI__SERVER__PORT", "8002")
    assert config.Settings.load().openai.default_model == "nested-environment"
    assert config.Settings.load().server.port == 8002
    monkeypatch.delenv("ORCHESTRAI__OPENAI__DEFAULT_MODEL")
    assert config.Settings.load().openai.default_model == "provider-environment"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_credential_alias_precedence(tmp_path, monkeypatch, provider):
    # Empty strings exercise alias handling without creating credential values.
    path = write_config(tmp_path, f"{provider}: {{api_key: null}}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "")
    assert getattr(config.Settings.load(), provider).api_key == ""
    monkeypatch.setenv(f"ORCHESTRAI__{provider.upper()}", '{"api_key": null}')
    assert getattr(config.Settings.load(), provider).api_key is None


@pytest.mark.parametrize("stage", ["validation", "discovery", "registry", "cancel"])
async def test_failed_reload_retains_runtime(tmp_path, monkeypatch, stage):
    import asyncio

    previous = config.get_settings()
    registry = CapabilityRegistry()
    orch = Orchestrator(registry)
    old_runtime = (orch._settings, orch._registry, orch._router._registry, orch._router._policy)
    path = write_config(tmp_path, "policy: {privacy_level: secret}")
    if stage == "validation":
        path.write_text("policy: {privacy_level: invalid}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))

    async def discover():
        assert orch._settings is previous
        assert config._settings is previous
        assert config.get_settings().policy.privacy_level == "secret"
        if stage == "discovery":
            raise RuntimeError("untrusted detail")
        if stage == "cancel":
            raise asyncio.CancelledError
        return []

    build = AsyncMock(side_effect=RuntimeError("untrusted detail"))
    with (
        patch("orchestrai.providers.discovery.discover_providers", discover),
        patch.object(CapabilityRegistry, "build", build),
    ):
        if stage == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await _reload_config({}, orch, registry)
        else:
            result = await _reload_config({}, orch, registry)
            assert "error" in result
            assert "untrusted detail" not in result["error"]
    assert config.get_settings() is previous
    assert (
        orch._settings,
        orch._registry,
        orch._router._registry,
        orch._router._policy,
    ) == old_runtime


async def test_successful_reload_publishes_candidate(tmp_path, monkeypatch):
    previous = config.get_settings()
    registry = CapabilityRegistry()
    orch = Orchestrator(registry)
    monkeypatch.setenv(
        "ORCHESTRAI_CONFIG", str(write_config(tmp_path, "policy: {privacy_level: secret}"))
    )
    candidate_registry = CapabilityRegistry()
    with (
        patch("orchestrai.providers.discovery.discover_providers", AsyncMock(return_value=[])),
        patch.object(CapabilityRegistry, "build", AsyncMock(return_value=candidate_registry)),
    ):
        result = await _reload_config({}, orch, registry)
    assert result["reloaded"]
    assert config.get_settings() is orch._settings
    assert orch._settings is not previous
    assert orch._settings.policy.privacy_level == "secret"
    assert orch._registry is orch._router._registry is candidate_registry
    assert orch._router._policy is orch._settings.policy


async def test_stale_probe_cannot_overwrite_reloaded_runtime(tmp_path, monkeypatch):
    old_registry = CapabilityRegistry()
    new_registry = CapabilityRegistry()
    stale_registry = CapabilityRegistry()
    orchestrator = Orchestrator(old_registry)
    monkeypatch.setenv(
        "ORCHESTRAI_CONFIG", str(write_config(tmp_path, "policy: {privacy_level: secret}"))
    )
    probe_entered = asyncio.Event()
    resume_probe = asyncio.Event()
    discovery_count = 0

    async def discover():
        nonlocal discovery_count
        discovery_count += 1
        if discovery_count == 1:
            assert config.get_settings().policy.privacy_level == "public"
            probe_entered.set()
            await resume_probe.wait()
            return [SimpleNamespace(name="old-provider")]
        assert config.get_settings().policy.privacy_level == "secret"
        return [SimpleNamespace(name="new-provider")]

    async def build(providers):
        if providers[0].name == "old-provider":
            return stale_registry
        return new_registry

    with (
        patch("orchestrai.providers.discovery.discover_providers", discover),
        patch.object(CapabilityRegistry, "build", AsyncMock(side_effect=build)),
    ):
        probe = asyncio.create_task(_probe_providers({}, orchestrator, old_registry))
        await asyncio.wait_for(probe_entered.wait(), timeout=2)
        reload_result = await _reload_config({}, orchestrator, old_registry)
        resume_probe.set()
        probe_result = await probe

    assert reload_result["reloaded"] is True
    assert orchestrator._settings.policy.privacy_level == "secret"
    assert orchestrator._registry is new_registry
    assert probe_result["published"] is False


async def test_probe_rejects_independently_published_settings(tmp_path, monkeypatch):
    old_registry = CapabilityRegistry()
    candidate_registry = CapabilityRegistry()
    orchestrator = Orchestrator(old_registry)
    old_settings = orchestrator._settings
    monkeypatch.setenv(
        "ORCHESTRAI_CONFIG", str(write_config(tmp_path, "policy: {privacy_level: secret}"))
    )
    published_settings = config.reload_settings()
    discovery_settings = None

    async def discover():
        nonlocal discovery_settings
        discovery_settings = config.get_settings()
        return [SimpleNamespace(name="captured-provider")]

    with (
        patch("orchestrai.providers.discovery.discover_providers", discover),
        patch.object(
            CapabilityRegistry, "build", AsyncMock(return_value=candidate_registry)
        ),
    ):
        result = await _probe_providers({}, orchestrator, old_registry)

    assert discovery_settings is old_settings
    assert result["published"] is False
    assert config.get_settings() is published_settings
    assert orchestrator._settings is old_settings
    assert orchestrator._registry is old_registry


async def test_stale_reload_cannot_overwrite_newer_reload():
    old_registry = CapabilityRegistry()
    internal_registry = CapabilityRegistry()
    secret_registry = CapabilityRegistry()
    orchestrator = Orchestrator(old_registry)
    internal_settings = config.Settings(
        policy=config.PolicyConfig(privacy_level="internal")
    )
    secret_settings = config.Settings(policy=config.PolicyConfig(privacy_level="secret"))
    first_entered = asyncio.Event()
    resume_first = asyncio.Event()

    async def discover():
        privacy = config.get_settings().policy.privacy_level
        if privacy == "internal":
            first_entered.set()
            await resume_first.wait()
        return [SimpleNamespace(name=privacy)]

    async def build(providers):
        if providers[0].name == "internal":
            return internal_registry
        return secret_registry

    with (
        patch(
            "orchestrai.config.settings.reload_settings",
            side_effect=[internal_settings, secret_settings],
        ),
        patch("orchestrai.providers.discovery.discover_providers", discover),
        patch.object(CapabilityRegistry, "build", AsyncMock(side_effect=build)),
    ):
        first = asyncio.create_task(_reload_config({}, orchestrator, old_registry))
        await asyncio.wait_for(first_entered.wait(), timeout=2)
        second_result = await _reload_config({}, orchestrator, old_registry)
        resume_first.set()
        first_result = await first

    assert second_result["reloaded"] is True
    assert first_result["reloaded"] is False
    assert orchestrator._settings is secret_settings
    assert orchestrator._registry is secret_registry
    assert config.get_settings() is secret_settings


async def test_reload_publishes_registry_to_server_main(tmp_path, monkeypatch):
    from orchestrai.server import main

    old_registry = CapabilityRegistry()
    orchestrator = Orchestrator(old_registry)
    candidate_registry = CapabilityRegistry()
    monkeypatch.setattr(main, "_orchestrator", orchestrator)
    monkeypatch.setattr(main, "_registry", old_registry)
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(write_config(tmp_path, "{}")))

    with (
        patch("orchestrai.providers.discovery.discover_providers", AsyncMock(return_value=[])),
        patch.object(CapabilityRegistry, "build", AsyncMock(return_value=candidate_registry)),
    ):
        result = await _reload_config({}, await main.get_orchestrator(), main._registry)

    assert result["reloaded"]
    assert main._registry is candidate_registry
    assert orchestrator._registry is candidate_registry
    assert orchestrator._router._registry is candidate_registry


def test_explicit_argument_wins_over_environment_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(tmp_path / "missing.yaml"))
    path = write_config(tmp_path, "server: {port: 8003}")
    assert config.Settings.load(path).server.port == 8003


def test_invalid_yaml_cannot_be_masked_by_environment(tmp_path, monkeypatch):
    path = write_config(tmp_path, "server: {port: invalid}")
    monkeypatch.setenv("ORCHESTRAI__SERVER__PORT", "8004")
    with pytest.raises(config.ConfigurationError):
        config.Settings.load(path)


@pytest.mark.parametrize("value", ["invalid", "{broken", "0"])
def test_invalid_environment_fails_without_default_file(monkeypatch, value):
    monkeypatch.setenv("ORCHESTRAI__SERVER__PORT", value)
    with pytest.raises(config.ConfigurationError) as error:
        config.Settings.load()
    assert str(error.value) == "Invalid configuration: settings"


@pytest.mark.parametrize(
    "content",
    [
        b"server: {port: 8001}\n---\nserver: {port: 8002}",
        b"\xff\xfe\xff",
        b"policy: {<<: {privacy_level: secret}}",
        b"{1: value}",
        b"server: {}\nserver: {}",
        b"local_providers: [{name: a, name: b}]",
        b"policy: {allowed_providers: [" + b"a," * 10001 + b"]}",
    ],
    ids=["documents", "encoding", "merge", "key-type", "duplicate-root", "duplicate-list", "nodes"],
)
def test_additional_yaml_boundaries(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_bytes(content)
    with pytest.raises(config.ConfigurationError):
        config.Settings.load(path)


def test_present_implicit_default_must_validate(tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "default.yaml").write_text("server: {port: invalid}")
    with pytest.raises(config.ConfigurationError):
        config.Settings.load()


async def test_reload_candidate_is_private_to_discovery(tmp_path, monkeypatch):
    import asyncio

    previous = config.get_settings()
    registry = CapabilityRegistry()
    orch = Orchestrator(registry)
    monkeypatch.setenv(
        "ORCHESTRAI_CONFIG", str(write_config(tmp_path, "policy: {privacy_level: secret}"))
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def discover():
        entered.set()
        await release.wait()
        assert config.get_settings().policy.privacy_level == "secret"

        # asyncio child tasks used by discovery inherit the candidate.
        async def child():
            return config.get_settings()

        assert await asyncio.create_task(child()) is config.get_settings()
        return []

    with patch("orchestrai.providers.discovery.discover_providers", discover):
        task = asyncio.create_task(_reload_config({}, orch, registry))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            assert config.get_settings() is previous
            assert orch._settings is previous
            assert orch._registry is registry
        finally:
            release.set()
            result = await task
    assert result["reloaded"]
    assert config.get_settings() is orch._settings


@pytest.mark.parametrize(
    "entry",
    ["UNRELATED=value\n", "OPENAI_API_KEY=[REDACTED]\n"],
    ids=["unrelated", "provider-credential"],
)
def test_dotenv_ignores_non_settings_keys(tmp_path, monkeypatch, entry):
    path = write_config(tmp_path, "server: {port: 8001}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    monkeypatch.setitem(config.Settings.model_config, "env_file", ".env")
    (tmp_path / ".env").write_text(entry)

    assert config.Settings.load().server.port == 8001


def test_dotenv_compatibility_without_reading_host_files(tmp_path, monkeypatch):
    path = write_config(tmp_path, "server: {port: 8001}")
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(path))
    monkeypatch.setitem(config.Settings.model_config, "env_file", ".env")
    (tmp_path / ".env").write_text("ORCHESTRAI__SERVER__PORT=8002\n")

    assert config.Settings.load().server.port == 8002
    monkeypatch.setenv("ORCHESTRAI__SERVER__PORT", "8003")
    assert config.Settings.load().server.port == 8003
