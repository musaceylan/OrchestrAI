"""
OrchestrAI MCP Server entrypoint.
Registers all tools, resources, and prompts. Boots provider discovery.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import click
import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

from orchestrai.config.settings import get_settings
from orchestrai.observability.trace import configure_logging
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.providers.discovery import discover_providers
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.prompts import BUILTIN_PROMPTS, render_prompt
from orchestrai.server.tools import build_tools, handle_tool

log = structlog.get_logger()

_orchestrator: Orchestrator | None = None
_registry: CapabilityRegistry | None = None


async def get_orchestrator() -> Orchestrator:
    global _orchestrator, _registry
    if _orchestrator is None:
        providers = await discover_providers()
        _registry = await CapabilityRegistry.build(providers)
        _orchestrator = Orchestrator(_registry)
    return _orchestrator


async def serve() -> None:
    settings = get_settings()
    configure_logging(
        level=settings.observability.log_level,
        fmt=settings.observability.log_format,
    )

    server: Server = Server(settings.server.name)
    tools = build_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        orch = await get_orchestrator()
        result = await handle_tool(name, arguments, orch, _registry)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="orchestrai://registry",
                name="Capability Registry",
                description="All available providers and models with their capabilities",
                mimeType="application/json",
            ),
            Resource(
                uri="orchestrai://status",
                name="Server Status",
                description="Active tasks and server health",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        orch = await get_orchestrator()
        if uri == "orchestrai://registry":
            return json.dumps(_registry.to_dict() if _registry else {}, indent=2)
        if uri == "orchestrai://status":
            return json.dumps(
                {
                    "status": "running",
                    "active_tasks": orch.get_active_tasks(),
                    "providers": [p.name for p in (_registry.all_providers() if _registry else [])],
                },
                indent=2,
            )
        return json.dumps({"error": f"Unknown resource: {uri}"})

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name=p["name"],
                description=p["description"],
                arguments=[
                    PromptArgument(name=a["name"], description=a["description"], required=a.get("required", False))
                    for a in p.get("arguments", [])
                ],
            )
            for p in BUILTIN_PROMPTS
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
        rendered = render_prompt(name, arguments or {})
        return GetPromptResult(
            description=rendered["description"],
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=rendered["content"]),
                )
            ],
        )

    log.info("orchestrai.starting", transport=settings.server.transport)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


@click.command()
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--log-format", default="json", type=click.Choice(["json", "console"]))
def cli(log_level: str, log_format: str) -> None:
    """OrchestrAI MCP Server"""
    configure_logging(level=log_level, fmt=log_format)
    asyncio.run(serve())


if __name__ == "__main__":
    cli()
