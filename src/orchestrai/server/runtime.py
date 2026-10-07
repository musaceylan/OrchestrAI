"""Server-owned request authority; absence means the unchanged local stdio path."""

from __future__ import annotations

import hmac
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import monotonic

from orchestrai.config.settings import get_settings


class AccessDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Access denied")


@dataclass(frozen=True)
class Principal:
    name: str
    scopes: frozenset[str]
    roots: tuple[Path, ...]


@dataclass(frozen=True)
class Access:
    principal: Principal
    credential: str | None = field(default=None, repr=False)


_access: ContextVar[Access | None] = ContextVar("sse_access", default=None)


def access_for_token(value: str) -> Access:
    if re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value) is None:
        raise AccessDenied
    matched = None
    for grant in get_settings().server.sse.tokens:
        if hmac.compare_digest(value, grant.token.get_secret_value()):
            matched = grant
    if matched is None:
        raise AccessDenied
    return Access(
        Principal(matched.principal, frozenset(matched.scopes), matched.allowed_roots),
        sha256(value.encode("ascii")).hexdigest(),
    )


@contextmanager
def bind_access(access: Access) -> Iterator[None]:
    token = _access.set(access)
    try:
        yield
    finally:
        _access.reset(token)


def current_principal() -> Principal | None:
    access = _access.get()
    if access is None:
        return None
    config = get_settings().server.sse
    original = access.principal
    if access.credential is None:
        if config.auth_required or config.tokens:
            raise AccessDenied
        return original
    matched = None
    for grant in config.tokens:
        fingerprint = sha256(grant.token.get_secret_value().encode("ascii")).hexdigest()
        if hmac.compare_digest(access.credential, fingerprint):
            matched = grant
    if matched is None or matched.principal != original.name:
        raise AccessDenied
    roots = tuple(
        {
            root if root.is_relative_to(previous) else previous
            for root in matched.allowed_roots
            for previous in original.roots
            if root.is_relative_to(previous) or previous.is_relative_to(root)
        }
    )
    return Principal(original.name, original.scopes.intersection(matched.scopes), roots)


def require_scope(scope: str) -> None:
    principal = current_principal()
    if principal is not None and scope not in principal.scopes:
        raise AccessDenied


class RateLimits:
    """One process-local window per principal, shared across their credentials."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def admit(self, principal: Principal, limit: int) -> bool:
        now = monotonic()
        with self._lock:
            self._windows = {
                name: window for name, window in self._windows.items() if now - window[0] < 60
            }
            start, count = self._windows.get(principal.name, (now, 0))
            if count >= limit or (count == 0 and len(self._windows) >= 1024):
                return False
            self._windows[principal.name] = (start, count + 1)
            return True


class SSEConnectionLimit(ValueError):
    def __init__(self) -> None:
        super().__init__("Active SSE connection limit reached")


class SSEConnectionLimits:
    """Bound live HTTP streams per process and principal, across credentials."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: dict[str, int] = {}

    def acquire(self) -> Callable[[], None]:
        principal = current_principal()
        if principal is None:
            raise AccessDenied
        config = get_settings().server.sse
        name = principal.name
        with self._lock:
            count = self._active.get(name, 0)
            if (
                count >= config.max_sse_connections_per_principal
                or sum(self._active.values()) >= config.max_sse_connections
            ):
                raise SSEConnectionLimit
            self._active[name] = count + 1
        released = False

        def release() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                remaining = self._active[name] - 1
                if remaining:
                    self._active[name] = remaining
                else:
                    del self._active[name]

        return release


class ActiveTaskLimit(ValueError):
    def __init__(self) -> None:
        super().__init__("Active task limit reached")


class TaskLimits:
    """Reserve before preparation; release exactly once even on early cancellation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: dict[str, int] = {}

    def acquire(self) -> Callable[[], None]:
        principal = current_principal()
        if principal is None:
            return lambda: None
        config = get_settings().server.sse
        name = principal.name
        with self._lock:
            count = self._active.get(name, 0)
            if (
                count >= config.max_active_tasks_per_principal
                or sum(self._active.values()) >= config.max_active_tasks
            ):
                raise ActiveTaskLimit
            self._active[name] = count + 1
        released = False

        def release() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                remaining = self._active[name] - 1
                if remaining:
                    self._active[name] = remaining
                else:
                    del self._active[name]

        return release
