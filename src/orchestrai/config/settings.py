"""
Configuration — environment variables + YAML config, all typed via Pydantic.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ConfigSection(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        frozen=True,
    )


class AnthropicConfig(ConfigSection):
    api_key: str | None = Field(
        default=None, repr=False, validation_alias=AliasChoices("api_key", "ANTHROPIC_API_KEY")
    )
    default_model: str = "claude-opus-4-6"
    base_url: str | None = None
    max_tokens: int = Field(default=8192, gt=0)
    timeout: float = Field(default=120.0, gt=0)


class OpenAIConfig(ConfigSection):
    api_key: str | None = Field(
        default=None, repr=False, validation_alias=AliasChoices("api_key", "OPENAI_API_KEY")
    )
    default_model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = Field(default=8192, gt=0)
    timeout: float = Field(default=120.0, gt=0)


class GeminiConfig(ConfigSection):
    api_key: str | None = Field(
        default=None, repr=False, validation_alias=AliasChoices("api_key", "GEMINI_API_KEY")
    )
    default_model: str = "gemini-2.0-flash"
    timeout: float = Field(default=120.0, gt=0)


class LocalProviderEndpoint(ConfigSection):
    name: str = "ollama"
    base_url: str = "http://localhost:11434"
    kind: Literal["openai_compat", "anthropic_compat"] = "openai_compat"
    probe_timeout: float = Field(default=5.0, gt=0)
    enabled: bool = True

    @property
    def is_loopback(self) -> bool:
        """Whether this HTTP endpoint is on the current machine's loopback interface."""
        # Reject characters that URL parsers may silently strip or interpret differently.
        if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in self.base_url):
            return False
        if "\\" in self.base_url:
            return False
        try:
            url = urlsplit(self.base_url)
            hostname = url.hostname
            # Parsing alone does not validate malformed or out-of-range ports.
            _ = url.port
        except ValueError:
            return False
        if url.scheme not in {"http", "https"} or not hostname:
            return False
        hostname = hostname.removesuffix(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            return len(hostname) <= 253 and all(
                0 < len(label) <= 63
                and label.isascii()
                and label.replace("-", "").isalnum()
                and not label.startswith("-")
                and not label.endswith("-")
                for label in hostname.split(".")
            )
        try:
            return ip_address(hostname).is_loopback
        except ValueError:
            return False


class OrchestratorConfig(ConfigSection):
    max_parallel_agents: int = Field(default=6, gt=0)
    default_mode: Literal[
        "planner_coder_reviewer", "parallel_draft", "impl_tester", "bugfix", "refactor", "docs"
    ] = "planner_coder_reviewer"
    timeout_budget_sec: float = Field(default=300.0, gt=0)
    require_human_approval_for_writes: bool = False
    safe_mode: bool = False
    judge_enabled: bool = True
    judge_model: str | None = None  # override judge model
    min_candidates_for_judge: int = Field(default=2, ge=2)


class PolicyConfig(ConfigSection):
    local_only_mode: bool = False
    allowed_providers: tuple[str, ...] = ()
    denied_providers: tuple[str, ...] = ()
    max_cost_usd: float | None = Field(default=None, ge=0)
    privacy_level: Literal["public", "internal", "confidential", "secret"] = "public"
    sensitive_path_patterns: tuple[str, ...] = (".env", "secrets/", "*.pem", "*.key")


class ObservabilityConfig(ConfigSection):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"  # json | console
    trace_dir: str = ".orchestrai/traces"
    artifact_dir: str = ".orchestrai/artifacts"
    metrics_enabled: bool = True


class ServerConfig(ConfigSection):
    transport: Literal["stdio", "sse"] = "stdio"  # stdio | sse
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    name: str = "orchestrai"
    version: str = "0.1.0"


class SettingsValues(ConfigSection):
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    local_providers: tuple[LocalProviderEndpoint, ...] = Field(
        default_factory=lambda: (LocalProviderEndpoint(),)
    )
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


class ConfigurationError(ValueError):
    """A configuration failure with no user-controlled values in its message."""


# Bound both input allocation and parser work. Aliases/merge keys are intentionally
# unsupported: they complicate duplicate detection and permit expansion attacks.
MAX_CONFIG_BYTES = 1024 * 1024
MAX_YAML_DEPTH = 32
MAX_YAML_NODES = 10000


class ConfigLoader(yaml.SafeLoader):  # type: ignore[misc]  # PyYAML loader base is untyped.
    def __init__(self, stream: bytes) -> None:
        super().__init__(stream)
        self.depth = 0
        self.nodes = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        self.depth += 1
        self.nodes += 1
        try:
            if (
                self.depth > MAX_YAML_DEPTH
                or self.nodes > MAX_YAML_NODES
                or self.check_event(yaml.AliasEvent)
            ):
                raise ConfigurationError("Invalid configuration: YAML limits")
            return super().compose_node(parent, index)
        finally:
            self.depth -= 1

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ConfigurationError("Invalid configuration: YAML mapping")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        content = stream.read(MAX_CONFIG_BYTES + 1)
    if len(content) > MAX_CONFIG_BYTES:
        raise ConfigurationError("Invalid configuration: YAML size")
    data = yaml.load(content, Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise ConfigurationError("Invalid configuration: YAML mapping")
    # Validate the file independently so an environment override cannot hide an
    # invalid file. Normalize aliases before merging environment sources.
    return SettingsValues.model_validate(data).model_dump(exclude_unset=True)


def _provider_environment() -> dict[str, Any]:
    result = {}
    for name, model in (
        ("anthropic", AnthropicConfig),
        ("openai", OpenAIConfig),
        ("gemini", GeminiConfig),
    ):
        values = {}
        for field in model.model_fields:
            key = f"{name}_{field}".upper()
            if key in os.environ:
                values[field] = os.environ[key]
        if values:
            result[name] = values
    return result


class Settings(BaseSettings, SettingsValues):
    _config_source: str = PrivateAttr(default="defaults")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ORCHESTRAI__",
        env_nested_delimiter="__",
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
        frozen=True,
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_config_source":
            raise AttributeError("config source is immutable")
        super().__setattr__(name, value)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[Any, ...]:
        # Highest priority first; retain dotenv support without accepting unrelated keys.
        dotenv_source = cast(DotEnvSettingsSource, dotenv_settings)
        dotenv_source.dotenv_filtering = "only_existing"
        return env_settings, _provider_environment, dotenv_source, init_settings

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        return cls.load(path)

    @property
    def config_source(self) -> str:
        """Resolved YAML source, or ``defaults`` when no YAML file was loaded."""
        return self._config_source

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Load defaults < YAML < dotenv < provider environment < ORCHESTRAI__ environment.

        Only an absent implicit config/default.yaml is optional. Every explicit
        path and every present file must validate, even if environment overrides
        would otherwise mask its invalid fields. Errors never include values.
        """
        explicit = path is not None or "ORCHESTRAI_CONFIG" in os.environ
        try:
            yaml_path = Path(
                path
                if path is not None
                else os.environ.get("ORCHESTRAI_CONFIG", "config/default.yaml")
            )
            loaded_yaml = True
            try:
                data = _read_yaml(yaml_path)
            except FileNotFoundError:
                if explicit:
                    raise
                data = {}
                loaded_yaml = False
            settings = cls(**data)
            object.__setattr__(
                settings,
                "_config_source",
                str(yaml_path.resolve()) if loaded_yaml else "defaults",
            )
            return settings
        except Exception:
            # Do not expose parser excerpts, validation inputs, unknown keys,
            # environment values, or even a user-controlled filename.
            raise ConfigurationError("Invalid configuration: settings") from None


_settings: Settings | None = None
_candidate: ContextVar[Settings | None] = ContextVar("candidate_settings", default=None)


@contextmanager
def candidate_settings(settings: Settings) -> Iterator[None]:
    """Make a candidate visible only to discovery and its child asyncio tasks."""
    token = _candidate.set(settings)
    try:
        yield
    finally:
        _candidate.reset(token)


def publish_settings(settings: Settings) -> None:
    global _settings
    _settings = settings


def get_settings() -> Settings:
    global _settings
    candidate = _candidate.get()
    if candidate is not None:
        return candidate
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings(*, publish: bool = True) -> Settings:
    """Validate disk/env settings; optionally defer publication for runtime reload."""
    settings = Settings.load()
    if publish:
        publish_settings(settings)
    return settings
