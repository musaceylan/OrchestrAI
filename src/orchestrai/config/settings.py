"""
Configuration — environment variables + YAML config, all typed via Pydantic.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnthropicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")
    api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    default_model: str = "claude-opus-4-6"
    base_url: str | None = None
    max_tokens: int = 8192
    timeout: float = 120.0


class OpenAIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    default_model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 8192
    timeout: float = 120.0


class GeminiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI_")
    api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    default_model: str = "gemini-2.0-flash"
    timeout: float = 120.0


class LocalProviderEndpoint(BaseSettings):
    name: str = "ollama"
    base_url: str = "http://localhost:11434"
    kind: str = "openai_compat"  # openai_compat | anthropic_compat
    probe_timeout: float = 5.0
    enabled: bool = True


class OrchestratorConfig(BaseSettings):
    max_parallel_agents: int = 6
    default_mode: str = "planner_coder_reviewer"
    timeout_budget_sec: float = 300.0
    require_human_approval_for_writes: bool = False
    safe_mode: bool = False
    judge_enabled: bool = True
    judge_model: str | None = None   # override judge model
    min_candidates_for_judge: int = 2


class PolicyConfig(BaseSettings):
    local_only_mode: bool = False
    allowed_providers: list[str] = Field(default_factory=list)
    denied_providers: list[str] = Field(default_factory=list)
    max_cost_usd: float | None = None
    privacy_level: str = "public"    # public | internal | confidential | secret
    sensitive_path_patterns: list[str] = Field(
        default_factory=lambda: [".env", "secrets/", "*.pem", "*.key"]
    )


class ObservabilityConfig(BaseSettings):
    log_level: str = "INFO"
    log_format: str = "json"   # json | console
    trace_dir: str = ".orchestrai/traces"
    artifact_dir: str = ".orchestrai/artifacts"
    metrics_enabled: bool = True


class ServerConfig(BaseSettings):
    transport: str = "stdio"   # stdio | sse
    host: str = "127.0.0.1"
    port: int = 8765
    name: str = "orchestrai"
    version: str = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    local_providers: list[LocalProviderEndpoint] = Field(
        default_factory=lambda: [LocalProviderEndpoint()]
    )
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        # env vars override YAML
        return cls(**data)

    @classmethod
    def load(cls) -> "Settings":
        """Load settings: YAML file if present, then env overrides."""
        import logging
        yaml_path = Path(os.environ.get("ORCHESTRAI_CONFIG", "config/default.yaml"))
        if yaml_path.exists():
            try:
                return cls.from_yaml(yaml_path)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Failed to load config from %s: %s. Falling back to env/defaults.",
                    yaml_path,
                    exc,
                )
        return cls()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
