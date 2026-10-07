"""HTTP trust boundary for the bounded opaque-token SSE resource server.

This transport does not advertise OAuth discovery or an authorization server.
"""

import re
from hashlib import sha256
from ipaddress import ip_address

from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import RequestBodyLimitMiddleware
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from orchestrai.config.settings import SSEConfig, get_settings
from orchestrai.server.runtime import (
    Access,
    AccessDenied,
    Principal,
    RateLimits,
    access_for_token,
    bind_access,
)


def is_loopback(address: str) -> bool:
    """Use the actual IP, never DNS or forwarded client headers, for peer trust."""
    try:
        return ip_address(address).is_loopback
    except ValueError:
        return False


def validate_bind(host: str) -> None:
    config = get_settings().server.sse
    if (not is_loopback(host) or config.allow_remote) and not remote_policy_ready(config):
        raise ValueError("SSE bind is not permitted")


def remote_policy_ready(config: SSEConfig) -> bool:
    return bool(
        config.allow_remote
        and config.tokens
        and config.tls_terminated
        and config.trusted_proxies
        and config.allowed_hosts
        and config.allowed_origins
        and all(origin.startswith("https://") for origin in config.allowed_origins)
    )


class SecurityMiddleware:
    """Authenticate before routing, including redirects, errors and method handling."""

    def __init__(self, app: ASGIApp, port: int) -> None:
        self.app = app
        self.rate_limits = RateLimits()
        self.hosts = frozenset(f"{host}:{port}" for host in ("127.0.0.1", "localhost", "[::1]"))
        self.origins = frozenset(f"http://{host}" for host in self.hosts)
        config = get_settings().server.sse
        self.auth_required = config.auth_required or bool(config.tokens)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        config = get_settings().server.sse
        self.auth_required = self.auth_required or config.auth_required or bool(config.tokens)
        identity = "direct-loopback"
        access = Access(
            Principal(
                "@loopback",
                frozenset({"tasks:read", "tasks:write", "admin"}),
                get_settings().policy.allowed_roots,
            )
        )
        if self.auth_required:
            authorization = Headers(scope=scope).getlist("authorization")
            match = (
                re.fullmatch(r"(?i:Bearer) ([A-Za-z0-9_-]{32,256})", authorization[0])
                if len(authorization) == 1
                else None
            )
            authenticated = False
            if match:
                try:
                    access = access_for_token(match.group(1))
                    identity = access.credential or identity
                    authenticated = True
                except AccessDenied:
                    pass
            if not authenticated:
                await Response("Unauthorized", 401, headers={"WWW-Authenticate": "Bearer"})(
                    scope,
                    receive,
                    send,
                )
                return
        headers = Headers(scope=scope)
        hosts = headers.getlist("host")
        origins = headers.getlist("origin")
        peer = scope.get("client")
        allowed_hosts = config.allowed_hosts or self.hosts
        allowed_origins = config.allowed_origins or self.origins
        if config.allow_remote:
            trusted_peer = bool(
                remote_policy_ready(config)
                and peer
                and peer[0] in config.trusted_proxies
                and headers.getlist("x-forwarded-proto") == ["https"]
            )
        else:
            trusted_peer = bool(peer and is_loopback(peer[0])) and not any(
                key == "forwarded" or key.startswith("x-forwarded-") for key in headers
            )
        if (
            len(hosts) != 1
            or hosts[0] not in allowed_hosts
            or len(origins) > 1
            or (origins and origins[0] not in allowed_origins)
            or not trusted_peer
        ):
            await Response("Forbidden", 403)(scope, receive, send)
            return
        if not self.rate_limits.admit(access.principal, config.requests_per_minute):
            await Response("Too many requests", 429, headers={"Retry-After": "60"})(
                scope,
                receive,
                send,
            )
            return
        # The SDK binds sessions to client_id. Use a credential fingerprint so
        # two credentials for one principal cannot post to each other's sessions.
        # Neither the SDK nor request metadata receives the bearer value.
        authority = sha256(
            repr(
                (
                    access.principal.name,
                    sorted(access.principal.scopes),
                    sorted(str(root) for root in access.principal.roots),
                )
            ).encode("utf-8")
        ).hexdigest()
        scope = {
            **scope,
            "headers": [
                (key, value) for key, value in scope["headers"] if key.lower() != b"authorization"
            ],
            "user": AuthenticatedUser(
                AccessToken(
                    token="",
                    client_id=identity,
                    subject=authority,
                    scopes=[],
                )
            ),
        }
        with bind_access(access):
            await RequestBodyLimitMiddleware(self.app, config.max_request_bytes)(
                scope,
                receive,
                send,
            )
