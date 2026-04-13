"""
OrchestrAI MCP Server entrypoint.
Registers all tools, resources, and prompts. Boots provider discovery.
Supports two transports:
  stdio  — default; works with Claude Desktop, Cursor, etc.
  sse    — HTTP/SSE; run with --transport sse [--host HOST] [--port PORT]
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


def _build_mcp_server() -> Server:
    """Create and configure the MCP Server with all tools, resources, and prompts."""
    settings = get_settings()
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
            Resource(
                uri="orchestrai://costs/summary",
                name="Cost Summary",
                description="Per-task and session-total token usage and cost in USD",
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
        if uri == "orchestrai://costs/summary":
            return json.dumps(orch.get_cost_summary(), indent=2, default=str)
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

    return server


async def serve_stdio() -> None:
    """Run the MCP server over stdio (default — for Claude Desktop / Cursor)."""
    settings = get_settings()
    configure_logging(
        level=settings.observability.log_level,
        fmt=settings.observability.log_format,
    )
    server = _build_mcp_server()
    log.info("orchestrai.starting", transport="stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def serve_sse(host: str, port: int) -> None:
    """
    Run the MCP server over HTTP/SSE.

    Clients connect to:
      GET  http://host:port/sse          — establishes the SSE stream
      POST http://host:port/messages/    — sends tool calls / requests
    """
    try:
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.routing import Mount, Route
    except ImportError as e:
        raise SystemExit(
            f"HTTP transport requires extra dependencies: {e}\n"
            "Install with: pip install orchestrai[http]"
        ) from e

    settings = get_settings()
    configure_logging(
        level=settings.observability.log_level,
        fmt=settings.observability.log_format,
    )

    server = _build_mcp_server()
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Any:
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send  # type: ignore[attr-defined]
        ) as streams:
            await server.run(*streams, server.create_initialization_options())

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]
    )

    log.info("orchestrai.starting", transport="sse", host=host, port=port)
    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    await uv_server.serve()


@click.command()
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--log-format", default="json", type=click.Choice(["json", "console"]))
@click.option(
    "--transport",
    default="stdio",
    type=click.Choice(["stdio", "sse"]),
    help="Transport: stdio (default) or sse (HTTP)",
)
@click.option("--host", default="127.0.0.1", help="Host for SSE transport (default: 127.0.0.1)")
@click.option("--port", default=8765, type=int, help="Port for SSE transport (default: 8765)")
def cli(log_level: str, log_format: str, transport: str, host: str, port: int) -> None:
    """OrchestrAI MCP Server — multi-model orchestration via MCP."""
    configure_logging(level=log_level, fmt=log_format)
    if transport == "sse":
        asyncio.run(serve_sse(host=host, port=port))
    else:
        asyncio.run(serve_stdio())


if __name__ == "__main__":
    cli()
