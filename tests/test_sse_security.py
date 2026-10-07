"""HTTP trust boundaries, exercised without opening a listening socket."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from orchestrai.server.main import serve_sse


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.1", "example.invalid"])
async def test_non_loopback_startup_refused(host: str) -> None:
    with (
        patch("orchestrai.server.main._build_mcp_server"),
        patch("orchestrai.server.main.configure_logging"),
        patch("uvicorn.Server.serve", new_callable=AsyncMock) as serve,
        pytest.raises(ValueError, match="SSE bind is not permitted"),
    ):
        await serve_sse(host, 8765)
    serve.assert_not_called()


async def captured_app(real: bool = False) -> ASGIApp:
    """Capture the production ASGI app; never bind or probe providers."""
    from orchestrai.server.main import _build_mcp_server

    async def drain(read: AsyncIterable[Any], write: Any, options: Any) -> None:
        async for _message in read:
            pass

    with (
        patch(
            "orchestrai.server.main._build_mcp_server", wraps=_build_mcp_server if real else None
        ) as build,
        patch("orchestrai.server.main.configure_logging"),
        patch("uvicorn.Config") as config,
        patch("uvicorn.Server.serve", new_callable=AsyncMock),
    ):
        if not real:
            build.return_value.run = AsyncMock(side_effect=drain)
        await serve_sse("127.0.0.1", 8765)
    return cast(ASGIApp, config.call_args.args[0])


async def first_status(
    app: ASGIApp,
    path: str,
    method: str = "GET",
    headers: Sequence[tuple[bytes, bytes]] = (),
    peer: str = "127.0.0.1",
    host: bytes = b"127.0.0.1:8765",
    bodies: tuple[bytes, ...] = (b"",),
) -> int:
    import asyncio
    from contextlib import suppress

    incoming: asyncio.Queue[Message] = asyncio.Queue()
    for index, body in enumerate(bodies):
        await incoming.put(
            {"type": "http.request", "body": body, "more_body": index < len(bodies) - 1}
        )
    started: asyncio.Future[int] = asyncio.Future()
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": method,
        "root_path": "",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "server": ("127.0.0.1", 8765),
        "client": (peer, 1234),
        "headers": [(b"host", host), *headers],
    }

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start" and not started.done():
            started.set_result(message["status"])

    task = asyncio.ensure_future(app(scope, incoming.get, send))
    try:
        return await asyncio.wait_for(started, timeout=2)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task


@pytest.mark.parametrize(
    "path,method",
    [
        ("/sse", "GET"),
        ("/messages/", "POST"),
        ("/messages", "POST"),
        ("/missing", "GET"),
        ("/sse", "OPTIONS"),
        ("/messages/", "GET"),
    ],
)
@pytest.mark.parametrize("credentials", ["missing", "wrong", "basic", "duplicate"])
async def test_bearer_required_on_every_http_route(
    monkeypatch: pytest.MonkeyPatch, path: str, method: str, credentials: str
) -> None:
    from orchestrai.config import settings as config

    # Obviously dummy fixtures, never usable credentials.
    dummy = "DUMMY_TEST_TOKEN_ALICE_NOT_A_SECRET"
    config.publish_settings(
        config.Settings.model_validate(
            {
                "server": {
                    "sse": {
                        "tokens": [
                            {
                                "token": dummy,
                                "principal": "alice",
                                "scopes": ["tasks:read"],
                                "allowed_roots": [],
                            }
                        ]
                    }
                }
            }
        )
    )
    headers = []
    if credentials != "missing":
        headers.append(
            (
                b"authorization",
                b"Basic invalid"
                if credentials == "basic"
                else b"Bearer DUMMY_WRONG_TOKEN_NOT_A_SECRET",
            )
        )
    if credentials == "duplicate":
        headers.append((b"authorization", b"Bearer " + dummy.encode()))
    assert await first_status(await captured_app(), path, method, headers) == 401


@pytest.mark.parametrize(
    "path,method", [("/sse", "GET"), ("/messages/", "POST"), ("/missing", "GET")]
)
@pytest.mark.parametrize(
    "case",
    [
        "host",
        "origin",
        "null-origin",
        "duplicate-host",
        "duplicate-origin",
        "remote",
        "forwarded",
        "no-peer",
    ],
)
async def test_direct_loopback_trust_checks(path: str, method: str, case: str) -> None:
    headers = []
    host = b"127.0.0.1:8765"
    peer = "127.0.0.1"
    if case == "host":
        host = b"attacker.invalid:8765"
    elif case == "origin":
        headers = [(b"origin", b"https://attacker.invalid")]
    elif case == "null-origin":
        headers = [(b"origin", b"null")]
    elif case == "duplicate-host":
        headers = [(b"host", b"127.0.0.1:8765")]
    elif case == "duplicate-origin":
        headers = [(b"origin", b"http://127.0.0.1:8765")] * 2
    elif case in {"remote", "forwarded"}:
        peer = "192.0.2.1"
        if case == "forwarded":
            headers = [(b"x-forwarded-for", b"127.0.0.1"), (b"x-forwarded-proto", b"https")]
    elif case == "no-peer":
        peer = ""
    assert await first_status(await captured_app(), path, method, headers, peer, host) == 403


@pytest.mark.parametrize("peer,host", [("127.42.1.1", b"127.0.0.1:8765"), ("::1", b"[::1]:8765")])
async def test_tokenless_direct_loopback_compatibility(peer: str, host: bytes) -> None:
    assert await first_status(await captured_app(), "/missing", peer=peer, host=host) == 404


@asynccontextmanager
async def sse_session(
    app: ASGIApp,
    headers: Sequence[tuple[bytes, bytes]],
    messages: bool = False,
    graceful: bool = False,
) -> AsyncIterator[Any]:
    import asyncio
    from contextlib import suppress

    incoming: asyncio.Queue[Message] = asyncio.Queue()
    outgoing: asyncio.Queue[Message] = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": b"", "more_body": False})
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "GET",
        "root_path": "",
        "path": "/sse",
        "raw_path": b"/sse",
        "query_string": b"",
        "server": ("127.0.0.1", 8765),
        "client": ("127.0.0.1", 1234),
        "headers": [(b"host", b"127.0.0.1:8765"), *headers],
    }
    task = asyncio.ensure_future(app(scope, incoming.get, outgoing.put))
    try:
        start = await asyncio.wait_for(outgoing.get(), 2)
        assert start["status"] == 200
        body = await asyncio.wait_for(outgoing.get(), 2)
        endpoint = body["body"].decode().split("data: ")[1].split("\r\n")[0]
        yield (endpoint, outgoing) if messages else endpoint
    finally:
        if graceful:
            await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 2)
        else:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def auth_header(who: str) -> list[tuple[bytes, bytes]]:
    return [(b"authorization", f"Bearer DUMMY_TEST_TOKEN_{who}_NOT_A_SECRET".encode())]


def publish_tokens(*identities: tuple[str, str], **options: Any) -> None:
    from orchestrai.config import settings as config

    config.publish_settings(
        config.Settings.model_validate(
            {
                "server": {
                    "sse": {
                        "tokens": [
                            {
                                "token": f"DUMMY_TEST_TOKEN_{who}_NOT_A_SECRET",
                                "principal": owner,
                                "scopes": ["tasks:read", "tasks:write"],
                                "allowed_roots": [],
                            }
                            for who, owner in identities
                        ],
                        **options,
                    }
                }
            }
        )
    )


@pytest.mark.parametrize("other", ["BOB", "ALICE_SECOND"])
async def test_message_post_bound_to_opening_credential(other: str) -> None:
    import httpx

    publish_tokens(("ALICE", "alice"), ("BOB", "bob"), ("ALICE_SECOND", "alice"))
    app = await captured_app()
    async with (
        sse_session(app, auth_header("ALICE")) as endpoint,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client,
    ):
        response = await client.post(
            endpoint,
            headers=dict(auth_header(other)),
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
        )
        assert response.status_code == 404
        response = await client.post(
            endpoint,
            headers=dict(auth_header("ALICE")),
            json={"jsonrpc": "2.0", "method": "ping", "id": 2},
        )
        assert response.status_code == 202
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
    ) as client:
        response = await client.post(
            endpoint,
            headers=dict(auth_header("ALICE")),
            json={"jsonrpc": "2.0", "method": "ping", "id": 3},
        )
        assert response.status_code == 404


async def test_http_assigns_immutable_principal_without_forwarding_bearer() -> None:
    from dataclasses import FrozenInstanceError

    from starlette.responses import Response

    from orchestrai.server.runtime import current_principal
    from orchestrai.server.security import SecurityMiddleware

    publish_tokens(("ALICE", "alice"))
    observed = []

    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        principal = current_principal()
        assert principal is not None
        observed.append(principal)
        assert all(key.lower() != b"authorization" for key, _ in scope["headers"])
        assert scope["user"].access_token.token == ""
        await Response(status_code=204)(scope, receive, send)

    app = SecurityMiddleware(endpoint, 8765)
    assert await first_status(app, "/anything", headers=auth_header("ALICE")) == 204
    assert observed[0].name == "alice"
    assert observed[0].scopes == frozenset({"tasks:read", "tasks:write"})
    with pytest.raises(FrozenInstanceError):
        observed[0].name = "bob"  # type: ignore[misc]  # Exercise the runtime frozen-field guard.
    assert current_principal() is None


async def test_revocation_rejects_both_routes_without_tokenless_fallback() -> None:
    publish_tokens(("ALICE", "alice"))
    app = await captured_app()
    assert await first_status(app, "/missing", headers=auth_header("ALICE")) == 404
    publish_tokens()
    for path, method in [("/sse", "GET"), ("/messages/", "POST"), ("/missing", "GET")]:
        assert await first_status(app, path, method, headers=auth_header("ALICE")) == 401
        assert await first_status(app, path, method) == 401


@pytest.mark.parametrize(
    "path,method",
    [("/sse", "GET"), ("/messages/", "POST"), ("/messages", "POST"), ("/missing", "PUT")],
)
@pytest.mark.parametrize("framing", ["length", "stream", "lying-length"])
async def test_request_bytes_bounded_before_routing(path: str, method: str, framing: str) -> None:
    from orchestrai.config import settings as config

    settings = config.Settings()
    config.publish_settings(
        settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={
                        "sse": settings.server.sse.model_copy(update={"max_request_bytes": 64})
                    },
                )
            }
        )
    )
    headers = [(b"content-type", b"application/json")]
    if framing == "length":
        headers.append((b"content-length", b"80"))
    elif framing == "lying-length":
        headers.append((b"content-length", b"1"))
    assert (
        await first_status(
            await captured_app(), path, method, headers, bodies=(b"x" * 40, b"x" * 40)
        )
        == 413
    )


def update_sse(**values: Any) -> None:
    from orchestrai.config import settings as config

    settings = config.get_settings()
    config.publish_settings(
        settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={"sse": settings.server.sse.model_copy(update=values)},
                )
            }
        )
    )


async def test_rate_limit_is_atomic_and_shared_by_principal() -> None:
    import asyncio

    publish_tokens(("ALICE", "alice"), ("ALICE_SECOND", "alice"), ("BOB", "bob"))
    update_sse(requests_per_minute=3)
    app = await captured_app()
    statuses = await asyncio.gather(
        *(
            first_status(app, "/missing", headers=auth_header("ALICE" if i % 2 else "ALICE_SECOND"))
            for i in range(20)
        )
    )
    assert statuses.count(404) == 3
    assert statuses.count(429) == 17
    assert await first_status(app, "/missing", headers=auth_header("BOB")) == 404
    # Publishing new settings must not reset a principal's consumption.
    update_sse(requests_per_minute=3)
    assert await first_status(app, "/missing", headers=auth_header("ALICE")) == 429


@pytest.mark.parametrize(
    "missing", ["enable", "tokens", "tls", "proxies", "hosts", "origins", None]
)
def test_remote_bind_requires_explicit_complete_policy(missing: str | None) -> None:
    from orchestrai.server.security import validate_bind

    publish_tokens(("ALICE", "alice"))
    policy = dict(
        allow_remote=True,
        tls_terminated=True,
        trusted_proxies=("192.0.2.10",),
        allowed_hosts=("gateway.example:443",),
        allowed_origins=("https://gateway.example",),
    )
    key = {
        "enable": "allow_remote",
        "tls": "tls_terminated",
        "proxies": "trusted_proxies",
        "hosts": "allowed_hosts",
        "origins": "allowed_origins",
    }.get(missing or "")
    if key:
        policy[key] = False if missing in {"enable", "tls"} else ()
    if missing == "tokens":
        policy["tokens"] = ()
    update_sse(**policy)
    if missing:
        with pytest.raises(ValueError, match="SSE bind is not permitted"):
            validate_bind("0.0.0.0")
    else:
        validate_bind("0.0.0.0")


@pytest.mark.parametrize(
    "case",
    ["trusted-tls", "untrusted-peer", "plain", "spoofed-host", "spoofed-origin", "duplicate-proto"],
)
async def test_remote_requests_require_trusted_tls_proxy(case: str) -> None:
    publish_tokens(("ALICE", "alice"))
    update_sse(
        allow_remote=True,
        tls_terminated=True,
        trusted_proxies=("192.0.2.10",),
        allowed_hosts=("gateway.example:443",),
        allowed_origins=("https://gateway.example",),
    )
    headers = [
        *auth_header("ALICE"),
        (b"x-forwarded-proto", b"https"),
        (b"origin", b"https://gateway.example"),
    ]
    peer, host = "192.0.2.10", b"gateway.example:443"
    if case == "untrusted-peer":
        peer = "192.0.2.11"
    elif case == "plain":
        headers[1] = (b"x-forwarded-proto", b"http")
    elif case == "spoofed-host":
        host = b"attacker.example:443"
    elif case == "spoofed-origin":
        headers[2] = (b"origin", b"https://attacker.example")
    elif case == "duplicate-proto":
        headers.append((b"x-forwarded-proto", b"https"))
    assert await first_status(
        await captured_app(), "/missing", headers=headers, peer=peer, host=host
    ) == (404 if case == "trusted-tls" else 403)


async def test_tokenless_loopback_rejects_forwarded_requests() -> None:
    assert (
        await first_status(
            await captured_app(),
            "/missing",
            headers=[
                (b"x-forwarded-for", b"192.0.2.11"),
            ],
        )
        == 403
    )


async def test_real_mcp_sse_workflow_preserves_contract_and_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import json

    import httpx

    from orchestrai.server import main
    from orchestrai.server.tools import build_tools
    from tests.test_tool_authorization import setup_owners

    orch = setup_owners(tmp_path)
    monkeypatch.setattr(main, "_orchestrator", orch)
    monkeypatch.setattr(main, "_registry", orch._registry)
    app = await captured_app(real=True)

    async with (
        sse_session(app, auth_header("ALICE"), messages=True) as (endpoint, outgoing),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8765",
            headers=dict(auth_header("ALICE")),
        ) as client,
    ):

        async def rpc(method: str, params: dict[str, Any], number: int) -> dict[str, Any]:
            response = await client.post(
                endpoint,
                json={
                    "jsonrpc": "2.0",
                    "id": number,
                    "method": method,
                    "params": params,
                },
            )
            assert response.status_code == 202
            event = await asyncio.wait_for(outgoing.get(), 2)
            return cast(
                dict[str, Any],
                json.loads(event["body"].decode().split("data: ")[1].split("\r\n")[0]),
            )

        initialized = await rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "dummy-test-client", "version": "1"},
            },
            1,
        )
        assert "result" in initialized
        assert (
            await client.post(
                endpoint, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
            )
        ).status_code == 202
        listing = await rpc("tools/list", {}, 2)
        actual = [
            {"name": tool["name"], "inputSchema": tool["inputSchema"]}
            for tool in listing["result"]["tools"]
        ]
        expected = [
            {"name": tool.name, "inputSchema": tool.model_dump(by_alias=True)["inputSchema"]}
            for tool in build_tools()
        ]
        assert actual == expected
        with patch.object(orch, "_run", new_callable=AsyncMock):
            submitted = await rpc(
                "tools/call",
                {
                    "name": "submit_task",
                    "arguments": {
                        "request": "Explain dummy code",
                        "user_preferences": {"principal": "bob"},
                    },
                },
                3,
            )
        result = json.loads(submitted["result"]["content"][0]["text"])
        owner = orch._owners[result["task_id"]]
        assert owner is not None and owner.name == "alice"
        status = await rpc("resources/read", {"uri": "orchestrai://status"}, 4)
        assert result["task_id"] in status["result"]["contents"][0]["text"]


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-token",
        "empty-root",
        "wildcard-host",
        "host-userinfo",
        "host-port",
        "origin-path",
        "proxy-name",
    ],
)
def test_security_config_rejects_ambiguous_or_unsafe_values(tmp_path: Path, case: str) -> None:
    from pydantic import ValidationError

    from orchestrai.config.settings import SSEConfig

    grant = {"token": "DUMMY_TEST_TOKEN_ALICE_NOT_A_SECRET", "principal": "alice"}
    options: dict[str, Any] = {"tokens": [grant]}
    if case == "duplicate-token":
        options["tokens"] = [grant, {**grant, "principal": "bob"}]
    elif case == "empty-root":
        options["tokens"] = [{**grant, "allowed_roots": [""]}]
    elif case == "wildcard-host":
        options["allowed_hosts"] = ["*"]
    elif case == "host-userinfo":
        options["allowed_hosts"] = ["user@gateway.example"]
    elif case == "host-port":
        options["allowed_hosts"] = ["gateway.example:65536"]
    elif case == "origin-path":
        options["allowed_origins"] = ["https://gateway.example/unexpected"]
    else:
        options["trusted_proxies"] = ["proxy.example"]
    with pytest.raises(ValidationError) as caught:
        SSEConfig.model_validate(options)
    assert grant["token"] not in str(caught.value)


async def test_stdio_real_framing_ignores_http_auth_and_logs_to_stderr(tmp_path: Path) -> None:
    import asyncio
    import json
    import os
    import sys

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("server:\n  sse:\n    auth_required: true\n")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "orchestrai.server.main",
        cwd=tmp_path,
        env={
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "ORCHESTRAI_CONFIG": str(settings_path),
        },
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "dummy-test-client", "version": "1"},
                    },
                }
            ).encode()
            + b"\n"
        )
        await process.stdin.drain()
        first = await asyncio.wait_for(process.stdout.readline(), 10)
        assert first, (await process.stderr.read()).decode()
        assert json.loads(first)["id"] == 1
        process.stdin.write(
            b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
        )
        await process.stdin.drain()
        second = await asyncio.wait_for(process.stdout.readline(), 5)
        assert len(json.loads(second)["result"]["tools"]) == 16
        process.stdin.close()
        await asyncio.wait_for(process.wait(), 5)
        stderr = (await process.stderr.read()).decode()
        assert "orchestrai.starting" in stderr
        assert process.returncode == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_sse_disconnect_finishes_cleanly() -> None:
    app = await captured_app()
    async with sse_session(app, [], graceful=True):
        pass


async def test_concurrent_first_requests_share_one_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from orchestrai.server import main

    monkeypatch.setattr(main, "_orchestrator", None)
    monkeypatch.setattr(main, "_registry", None)

    async def discover() -> list[Any]:
        await asyncio.sleep(0)
        return []

    with patch.object(main, "discover_providers", side_effect=discover) as discovery:
        runtimes = await asyncio.gather(*(main.get_orchestrator() for _ in range(10)))
    assert len({id(runtime) for runtime in runtimes}) == 1
    assert discovery.call_count == 1


@pytest.mark.parametrize("change", ["principal", "scopes", "roots"])
async def test_session_authority_change_requires_reconnect(tmp_path: Path, change: str) -> None:
    import httpx

    from orchestrai.config import settings as config

    publish_tokens(("ALICE", "alice"))
    app = await captured_app()
    async with sse_session(app, auth_header("ALICE")) as endpoint:
        grant = config.get_settings().server.sse.tokens[0]
        updates = (
            {"principal": "bob"}
            if change == "principal"
            else (
                {"scopes": (*grant.scopes, "admin")}
                if change == "scopes"
                else {"allowed_roots": (tmp_path,)}
            )
        )
        update_sse(tokens=(grant.model_copy(update=updates),))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client:
            response = await client.post(
                endpoint,
                headers=dict(auth_header("ALICE")),
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )
        assert response.status_code == 404


async def test_sdk_adapter_validates_tool_schema_before_discovery() -> None:
    from mcp import types

    from orchestrai.server import main

    server = main._build_mcp_server()
    entry = server.get_request_handler("tools/call")
    assert entry is not None
    with patch.object(main, "get_orchestrator", side_effect=AssertionError("unexpected discovery")):
        result = await entry.handler(
            cast(Any, None),
            types.CallToolRequestParams(
                name="submit_task",
                arguments={"request": 123},
            ),
        )
    assert isinstance(result, types.CallToolResult)
    assert result.is_error is True
    assert isinstance(result.content[0], types.TextContent)
    assert result.content[0].text == "Invalid tool arguments"


@pytest.mark.parametrize("name,minimum,unsupported", [
    ("mcp", "2.1.0", "2.0.0"),
    ("jsonschema", "4.20.0", "4.19.2"),
])
def test_production_dependency_minimums(name: str, minimum: str, unsupported: str) -> None:
    import tomllib

    from packaging.requirements import Requirement

    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    requirements = {
        item.name: item for item in map(Requirement, metadata["project"]["dependencies"])
    }
    assert name in requirements, f"Production import {name} must be a direct dependency"
    assert minimum in requirements[name].specifier
    assert unsupported not in requirements[name].specifier


def test_mcp_adapter_documents_supported_sdk() -> None:
    from orchestrai.server import mcp_compat

    assert "MCP >=2.1.0" in (mcp_compat.__doc__ or "")
    assert "MCP 1.x/2.x" not in (mcp_compat.__doc__ or "")


async def test_required_mcp_apis_build_and_dispatch() -> None:
    from mcp import types
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken
    from mcp.server.sse import SseServerTransport
    from mcp.server.transport_security import RequestBodyLimitMiddleware

    from orchestrai.server import main

    server = main._build_mcp_server()
    assert server.create_initialization_options().server_name == "orchestrai"
    for method in (
        "tools/list", "tools/call", "resources/list", "resources/read",
        "prompts/list", "prompts/get",
    ):
        assert server.get_request_handler(method) is not None
    entry = server.get_request_handler("tools/list")
    assert entry is not None
    result = await entry.handler(cast(Any, None), types.PaginatedRequestParams())
    assert isinstance(result, types.ListToolsResult)
    assert len(result.tools) == 16
    user = AuthenticatedUser(
        AccessToken(token="", client_id="dummy-client", subject="scope", scopes=[])
    )
    assert user.is_authenticated
    transport = SseServerTransport("/messages/")
    app = RequestBodyLimitMiddleware(transport.handle_post_message, 1024)
    assert await first_status(app, "/messages/", "POST", [(b"content-length", b"1025")]) == 413


@pytest.mark.parametrize("field", ["max_sse_connections", "max_sse_connections_per_principal"])
@pytest.mark.parametrize("value", [0, -1, 257, True, 1.5])
def test_sse_connection_settings_reject_invalid_limits(field: str, value: int | float) -> None:
    from pydantic import ValidationError

    from orchestrai.config.settings import SSEConfig

    with pytest.raises(ValidationError):
        SSEConfig.model_validate({field: value})


def test_sse_connection_settings_have_bounded_defaults_and_accept_limits() -> None:
    from orchestrai.config.settings import SSEConfig

    defaults = SSEConfig().model_dump()
    assert defaults["max_sse_connections"] == 32
    assert defaults["max_sse_connections_per_principal"] == 4
    for limit in (1, 256):
        settings = SSEConfig.model_validate({
            "max_sse_connections": limit, "max_sse_connections_per_principal": limit,
        })
        assert settings.max_sse_connections == limit
        assert settings.max_sse_connections_per_principal == limit


@pytest.mark.parametrize("graceful", [False, True])
async def test_real_mcp_sse_connection_caps_and_slot_reuse(graceful: bool) -> None:
    import asyncio

    publish_tokens(
        ("ALICE", "alice"), ("ALICE_SECOND", "alice"), ("BOB", "bob"), ("CAROL", "carol")
    )
    # model_copy lets the RED reach behavior before the new settings fields exist.
    update_sse(max_sse_connections=2, max_sse_connections_per_principal=1, max_active_tasks=1)
    app = await captured_app(real=True)
    async with sse_session(app, auth_header("ALICE"), graceful=graceful):
        statuses = await asyncio.gather(*(
            first_status(app, "/sse", headers=auth_header("ALICE_SECOND")) for _ in range(8)
        ))
        assert statuses == [429] * 8
        async with sse_session(app, auth_header("BOB"), graceful=graceful):
            assert await first_status(app, "/sse", headers=auth_header("CAROL")) == 429
        # Bob's release frees the global slot while Alice's reservation survives.
        async with sse_session(app, auth_header("CAROL"), graceful=graceful):
            assert await first_status(app, "/sse", headers=auth_header("ALICE_SECOND")) == 429
    async with sse_session(app, auth_header("ALICE_SECOND"), graceful=True):
        pass


@pytest.mark.parametrize("exit_kind", ["cancel", "disconnect", "handshake", "server", "normal"])
async def test_real_mcp_sse_cleanup_releases_exactly_once(
    monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    import asyncio

    import httpx
    from mcp.server import Server
    from starlette.responses import Response

    publish_tokens(("ALICE", "alice"), ("BOB", "bob"), ("CAROL", "carol"))
    update_sse(max_sse_connections=2, max_sse_connections_per_principal=1)
    app = await captured_app(real=True)
    async with sse_session(app, auth_header("ALICE"), graceful=True):
        # First prove real admission behavior; the original server returns 200.
        assert await first_status(app, "/sse", headers=auth_header("ALICE")) == 429

        from orchestrai.server.runtime import SSEConnectionLimits

        acquire = SSEConnectionLimits.acquire
        releases: list[Mock] = []

        def tracked_acquire(limits: SSEConnectionLimits) -> Callable[[], None]:
            release = Mock(wraps=acquire(limits))
            releases.append(release)
            return release

        monkeypatch.setattr(SSEConnectionLimits, "acquire", tracked_acquire)
        if exit_kind in {"cancel", "disconnect"}:
            async with sse_session(app, auth_header("BOB"), graceful=exit_kind == "disconnect"):
                assert len(releases) == 1
                releases[0].assert_not_called()
        elif exit_kind == "handshake":
            # Exercise rejection inside the real SDK context manager, before yield.
            with patch(
                "mcp.server.transport_security.TransportSecurityMiddleware.validate_request",
                new=AsyncMock(return_value=Response("Dummy handshake rejection", 403)),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
                ) as client:
                    with pytest.raises(ValueError, match="Request validation failed"):
                        await asyncio.wait_for(
                            client.get("/sse", headers=dict(auth_header("BOB"))), 2
                        )
        elif exit_kind == "server":
            with patch.object(
                Server, "run", new=AsyncMock(side_effect=RuntimeError("Dummy server failure"))
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
                ) as client:
                    with pytest.raises(ExceptionGroup) as caught:
                        await asyncio.wait_for(
                            client.get("/sse", headers=dict(auth_header("BOB"))), 2
                        )
            assert caught.value.subgroup(RuntimeError) is not None
        else:
            # Normal Server.run return, then let the SDK finish its real stream
            # via the ASGI disconnect rather than cancelling its task group.
            with patch.object(Server, "run", new=AsyncMock()) as run:
                async with sse_session(app, auth_header("BOB"), graceful=True):
                    run.assert_awaited_once()

        assert len(releases) == 1
        releases[0].assert_called_once_with()
        # A repeated release cannot erase Alice's live reservation or underflow.
        releases[0]()
        async with sse_session(app, auth_header("BOB"), graceful=True):
            assert await first_status(app, "/sse", headers=auth_header("ALICE")) == 429
            assert await first_status(app, "/sse", headers=auth_header("CAROL")) == 429
        assert len(releases) == 2  # rejected streams acquired no reservations
        releases[1].assert_called_once_with()


@pytest.mark.parametrize("fmt", ["json", "console"])
async def test_rejected_bearers_never_reach_production_stderr(
    capfd: pytest.CaptureFixture[str], fmt: str
) -> None:
    import structlog

    from orchestrai.observability.trace import configure_logging

    publish_tokens(("ALICE", "alice"))
    previous = structlog.get_config().copy()
    try:
        configure_logging(level="DEBUG", fmt=fmt)
        assert isinstance(structlog.get_config()["logger_factory"], structlog.PrintLoggerFactory)
        structlog.get_logger().info("dummy.production.logging.sentinel")
        # A fresh lazy proxy avoids retaining a prior test's closed capture FD.
        with patch("orchestrai.server.main.log", structlog.get_logger()):
            app = await captured_app(real=True)
        rejected = b"DUMMY_REJECTED_BEARER_NOT_A_SECRET_12345678"
        for path, method in (("/sse", "GET"), ("/messages/", "POST"), ("/missing", "GET")):
            for headers in (
                [(b"authorization", b"Bearer " + rejected)],
                [(b"authorization", b"Bearer " + rejected + b"!")],
                [*auth_header("ALICE"), (b"authorization", b"Bearer " + rejected)],
            ):
                assert await first_status(app, path, method, headers) == 401
        captured = capfd.readouterr()
        assert "dummy.production.logging.sentinel" in captured.err
        assert "dummy.production.logging.sentinel" not in captured.out
        assert rejected.decode() not in captured.err + captured.out
        assert "DUMMY_TEST_TOKEN_ALICE_NOT_A_SECRET" not in captured.err + captured.out
    finally:
        structlog.configure(**previous)
