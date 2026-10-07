"""Keep the existing handler contracts on MCP >=2.1.0.

MCP 2.x replaced low-level decorators with typed request registration. The
stdio transport and wire schemas remain owned by the SDK and existing handlers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from jsonschema import ValidationError, validate
from mcp import types

from orchestrai.server.runtime import current_principal
from orchestrai.server.tools import build_tools

Handler = Callable[..., Awaitable[Any]]
F = TypeVar("F", bound=Handler)


def register(server: Any, name: str) -> Callable[[F], F]:
    legacy = getattr(server, name, None)

    registrations = {
        "list_tools": ("tools/list", types.PaginatedRequestParams),
        "call_tool": ("tools/call", types.CallToolRequestParams),
        "list_resources": ("resources/list", types.PaginatedRequestParams),
        "read_resource": ("resources/read", types.ReadResourceRequestParams),
        "list_prompts": ("prompts/list", types.PaginatedRequestParams),
        "get_prompt": ("prompts/get", types.GetPromptRequestParams),
    }
    schemas = {tool.name: tool.model_dump(by_alias=True)["inputSchema"] for tool in build_tools()}

    def decorate(handler: F) -> F:
        async def guarded(*args: Any, **kwargs: Any) -> Any:
            current_principal()
            result = await handler(*args, **kwargs)
            current_principal()
            return result

        if legacy is not None:
            legacy()(guarded)
            return handler

        async def invoke(context: Any, params: Any) -> Any:
            if name == "list_tools":
                return types.ListToolsResult(tools=await guarded())
            if name == "list_resources":
                return types.ListResourcesResult(resources=await guarded())
            if name == "list_prompts":
                return types.ListPromptsResult(prompts=await guarded())
            if name == "read_resource":
                return types.ReadResourceResult(
                    contents=[
                        types.TextResourceContents(
                            uri=params.uri,
                            mime_type="application/json",
                            text=await guarded(str(params.uri)),
                        )
                    ]
                )
            if name == "get_prompt":
                return await guarded(params.name, params.arguments)
            if params.name in schemas:
                try:
                    validate(params.arguments or {}, schemas[params.name])
                except ValidationError:
                    return types.CallToolResult(
                        is_error=True,
                        content=[types.TextContent(type="text", text="Invalid tool arguments")],
                    )
            return types.CallToolResult(content=await guarded(params.name, params.arguments or {}))

        method, params_type = registrations[name]
        server.add_request_handler(method, params_type, invoke)
        return handler

    return decorate
