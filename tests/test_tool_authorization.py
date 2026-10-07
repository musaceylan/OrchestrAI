"""Server-owned principals must authorize before any tool side effect."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrai.config import settings as config
from orchestrai.config.settings import SSETokenConfig
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.tools import handle_tool


def setup_owners(tmp_path: Path) -> Orchestrator:
    config.publish_settings(
        config.Settings.model_validate(
            {
                "policy": {"allowed_roots": [tmp_path]},
                "observability": {
                    "artifact_dir": str(tmp_path / "artifacts"),
                    "trace_dir": str(tmp_path / "traces"),
                },
                "server": {
                    "sse": {
                        "tokens": [
                            {
                                "token": f"DUMMY_TEST_TOKEN_{name.upper()}_NOT_A_SECRET",
                                "principal": name,
                                "scopes": ["tasks:read", "tasks:write"],
                                "allowed_roots": [tmp_path],
                            }
                            for name in ("alice", "bob")
                        ]
                    }
                },
            }
        )
    )
    return Orchestrator(CapabilityRegistry())


def access(name: str) -> AbstractContextManager[None]:
    from orchestrai.server.runtime import access_for_token, bind_access

    return bind_access(access_for_token(f"DUMMY_TEST_TOKEN_{name.upper()}_NOT_A_SECRET"))


@pytest.mark.parametrize(
    "tool",
    [
        "inspect_plan",
        "inspect_agents",
        "inspect_artifacts",
        "inspect_trace",
        "get_task_status",
        "get_task_result",
        "get_task_events",
        "cancel_task",
        "rerun_with_policy",
        "compare_candidates",
    ],
)
async def test_cross_owner_tools_denied_before_side_effects(tmp_path: Path, tool: str) -> None:
    orch = setup_owners(tmp_path)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Explain dummy code")
    with (
        access("bob"),
        patch("orchestrai.artifacts.store.ArtifactStore") as store,
        patch("orchestrai.server.tools.run_judge", new_callable=AsyncMock) as judge,
        patch.object(orch, "submit", new_callable=AsyncMock) as submit,
        patch.object(orch, "cancel_task", new_callable=AsyncMock) as cancel,
    ):
        result = await handle_tool(
            tool, {"task_id": task.id, "policy_overrides": {}}, orch, orch._registry
        )
    assert result == {"error": "Access denied"}
    for effect in (store, judge, submit, cancel):
        effect.assert_not_called()


async def test_task_and_cost_lists_are_owner_filtered(tmp_path: Path) -> None:
    orch = setup_owners(tmp_path)
    with patch.object(orch, "_run", new_callable=AsyncMock):
        with access("alice"):
            alice = await orch.submit("Alice dummy task")
        with access("bob"):
            bob = await orch.submit("Bob dummy task")
    with access("alice"):
        listing = await handle_tool("list_tasks", {}, orch, orch._registry)
        assert [task["id"] for task in listing["active"]] == [alice.id]
        summary = orch.get_cost_summary()
        assert [task["task_id"] for task in summary["active_tasks"]] == [alice.id]
        orch._finished.update(orch._active)
        orch._active.clear()
        assert [task["id"] for task in orch.get_recent_tasks()] == [alice.id]
    with access("bob"):
        assert [task["id"] for task in orch.get_recent_tasks()] == [bob.id]


@pytest.mark.parametrize("tool", ["reload_config", "probe_providers"])
async def test_admin_only_tools(tmp_path: Path, tool: str) -> None:
    orch = setup_owners(tmp_path)
    with (
        access("alice"),
        patch(
            "orchestrai.providers.discovery.discover_providers", new_callable=AsyncMock
        ) as discover,
    ):
        result = await handle_tool(
            tool, {"scopes": ["admin"], "principal": "admin"}, orch, orch._registry
        )
    assert result == {"error": "Access denied"}
    discover.assert_not_called()


@pytest.mark.parametrize(
    "tool",
    [
        "submit_task",
        "cancel_task",
        "rerun_with_policy",
        "compare_candidates",
        "get_task_status",
        "list_tasks",
    ],
)
async def test_scopes_do_not_come_from_arguments(tmp_path: Path, tool: str) -> None:
    orch = setup_owners(tmp_path)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Explain dummy code")
    settings = config.get_settings()
    grant = settings.server.sse.tokens[0].model_copy(update={"scopes": ()})
    config.publish_settings(
        settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={"sse": settings.server.sse.model_copy(update={"tokens": (grant,)})},
                )
            }
        )
    )
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        result = await handle_tool(
            tool,
            {
                "task_id": task.id,
                "request": "Dummy task",
                "scopes": ["tasks:read", "tasks:write", "admin"],
                "policy_overrides": {},
            },
            orch,
            orch._registry,
        )
    assert result == {"error": "Access denied"}


async def test_direct_task_operations_also_enforce_ownership(tmp_path: Path) -> None:
    orch = setup_owners(tmp_path)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Explain dummy code")
    with access("bob"):
        with pytest.raises(PermissionError, match="Access denied"):
            await orch.cancel_task(task.id)
        with pytest.raises(PermissionError, match="Access denied"):
            await orch.rerun(task.id, {})
        with pytest.raises(PermissionError, match="Access denied"):
            orch.get_events(task.id)


@pytest.mark.parametrize("case", ["principal-root", "global-root", "deny-all", "spoof"])
async def test_authorized_roots_can_only_narrow(tmp_path: Path, case: str) -> None:
    orch = setup_owners(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    settings = config.get_settings()
    grant = settings.server.sse.tokens[0].model_copy(
        update={
            "allowed_roots": () if case == "deny-all" else (allowed,),
        }
    )
    if case == "global-root":
        grant = grant.model_copy(update={"allowed_roots": (tmp_path,)})
        settings = settings.model_copy(
            update={
                "policy": settings.policy.model_copy(
                    update={"allowed_roots": (allowed,)},
                )
            }
        )
    config.publish_settings(
        settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={"sse": settings.server.sse.model_copy(update={"tokens": (grant,)})},
                )
            }
        )
    )
    with (
        access("alice"),
        patch("orchestrai.orchestrator.orchestrator.ArtifactStore") as store,
        pytest.raises(ValueError, match="Repository path is not permitted"),
    ):
        await orch.submit(
            "Dummy task",
            repo_root=str(outside),
            user_preferences={
                "allowed_roots": [str(tmp_path)],
                "principal": "bob",
                "scopes": ["admin"],
            },
        )
    store.assert_not_called()


async def test_own_task_and_rerun_keep_server_assigned_owner(tmp_path: Path) -> None:
    orch = setup_owners(tmp_path)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Explain dummy code", user_preferences={"principal": "bob"})
        status = await handle_tool("get_task_status", {"task_id": task.id}, orch, orch._registry)
        assert status["task_id"] == task.id
        rerun = await orch.rerun(task.id, {"principal": "bob"})
        assert orch._owners[rerun.id] == orch._owners[task.id]
        owner = orch._owners[task.id]
        assert owner is not None and owner.name == "alice"


def replace_grants(grants: Sequence[SSETokenConfig]) -> None:
    settings = config.get_settings()
    config.publish_settings(
        settings.model_copy(
            update={
                "server": settings.server.model_copy(
                    update={
                        "sse": settings.server.sse.model_copy(update={"tokens": tuple(grants)})
                    },
                )
            }
        )
    )


@pytest.mark.parametrize("change", ["removed", "principal", "scopes", "roots"])
async def test_live_revocation_applies_to_already_captured_context(
    tmp_path: Path, change: str
) -> None:
    orch = setup_owners(tmp_path)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        grant = config.get_settings().server.sse.tokens[0]
        if change == "removed":
            replace_grants([])
        elif change == "principal":
            replace_grants([grant.model_copy(update={"principal": "bob"})])
        elif change == "scopes":
            replace_grants([grant.model_copy(update={"scopes": ()})])
        else:
            replace_grants([grant.model_copy(update={"allowed_roots": ()})])
        result = await handle_tool(
            "submit_task",
            {"request": "Dummy task", "repo_root": str(tmp_path)},
            orch,
            orch._registry,
        )
    assert "error" in result
    assert not orch._active
    assert not (tmp_path / "artifacts").exists()


async def test_expanding_grant_cannot_expand_an_existing_context(tmp_path: Path) -> None:
    from orchestrai.server.runtime import current_principal

    setup_owners(tmp_path)
    grant = config.get_settings().server.sse.tokens[0]
    replace_grants([grant.model_copy(update={"scopes": ("tasks:read",), "allowed_roots": ()})])
    with access("alice"):
        replace_grants(
            [grant.model_copy(update={"scopes": ("tasks:read", "tasks:write", "admin")})]
        )
        principal = current_principal()
        assert principal is not None
        assert principal.scopes == frozenset({"tasks:read"})
        assert principal.roots == ()


@pytest.mark.parametrize("tool", ["reload_config", "probe_providers"])
async def test_revoked_admin_cannot_publish_inflight_discovery(tmp_path: Path, tool: str) -> None:
    orch = setup_owners(tmp_path)
    grant = config.get_settings().server.sse.tokens[0]
    replace_grants([grant.model_copy(update={"scopes": ("admin",)})])
    orch._settings = config.get_settings()
    before = orch._registry

    async def discover() -> list[Any]:
        replace_grants([])
        return []

    with access("alice"), patch("orchestrai.providers.discovery.discover_providers", discover):
        result = await handle_tool(tool, {}, orch, orch._registry)
    assert result == {"error": "Access denied"}
    assert orch._registry is before


async def test_active_task_limits_are_atomic_and_release_on_cancel(tmp_path: Path) -> None:
    import asyncio

    from tests.test_sse_security import update_sse

    orch = setup_owners(tmp_path)
    update_sse(max_active_tasks=3, max_active_tasks_per_principal=2)
    release = asyncio.Event()

    async def slow(*args: Any) -> None:
        await release.wait()

    async def submit(name: str) -> dict[str, Any]:
        with access(name):
            return cast(
                dict[str, Any],
                await handle_tool(
                    "submit_task", {"request": "Dummy task", "wait": False}, orch, orch._registry
                ),
            )

    try:
        with patch.object(orch, "_run", side_effect=slow):
            alice = await asyncio.gather(*(submit("alice") for _ in range(10)))
            assert sum("task_id" in result for result in alice) == 2
            bob = await asyncio.gather(*(submit("bob") for _ in range(10)))
            assert sum("task_id" in result for result in bob) == 1
            with access("alice"):
                task_id = next(result["task_id"] for result in alice if "task_id" in result)
                await orch.cancel_task(task_id)
            assert "task_id" in await submit("alice")
    finally:
        release.set()
        await asyncio.gather(*orch._bg_tasks.values(), return_exceptions=True)


async def test_admission_releases_on_prepare_failure_and_wait_completion(tmp_path: Path) -> None:
    from tests.test_sse_security import update_sse

    orch = setup_owners(tmp_path)
    update_sse(max_active_tasks=1, max_active_tasks_per_principal=1)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        with pytest.raises(ValueError):
            await orch.submit("Dummy task", repo_root="")
        first = await orch.submit("Dummy task")
        second = await orch.submit("Dummy task")
        assert first.id != second.id


async def test_compare_cannot_bypass_active_task_limit(tmp_path: Path) -> None:
    import asyncio

    from orchestrai.artifacts.schemas import CodePatch, JudgeVerdict, Provenance
    from tests.test_sse_security import update_sse

    orch = setup_owners(tmp_path)
    update_sse(max_active_tasks=1, max_active_tasks_per_principal=1)
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Explain dummy code")
    store = orch._stores[task.id]
    store.put(CodePatch(id="dummy-patch", provenance=Provenance(task_id=task.id)))
    release, entered = asyncio.Event(), asyncio.Event()

    async def judge(**kwargs: Any) -> JudgeVerdict:
        entered.set()
        await release.wait()
        return JudgeVerdict(
            id="dummy-verdict",
            provenance=Provenance(task_id=task.id),
            winner_candidate_id="dummy-patch",
        )

    async def compare() -> dict[str, Any]:
        with access("alice"):
            return cast(
                dict[str, Any],
                await handle_tool("compare_candidates", {"task_id": task.id}, orch, orch._registry),
            )

    with patch("orchestrai.server.tools.run_judge", side_effect=judge) as run_judge:
        pending = [asyncio.create_task(compare()) for _ in range(6)]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert run_judge.await_count == 1
        finally:
            release.set()
            results = await asyncio.gather(*pending)
    assert sum(result.get("error") == "Active task limit reached" for result in results) == 5


async def test_revocation_is_rechecked_after_resource_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mcp import types

    from orchestrai.server import main

    orch = setup_owners(tmp_path)

    async def get_orchestrator() -> Orchestrator:
        replace_grants([])
        return orch

    monkeypatch.setattr(main, "get_orchestrator", get_orchestrator)
    monkeypatch.setattr(main, "_registry", orch._registry)
    server = main._build_mcp_server()
    entry = server.get_request_handler("resources/read")
    assert entry is not None
    with access("alice"), pytest.raises(PermissionError, match="Access denied"):
        await entry.handler(
            cast(Any, None), types.ReadResourceRequestParams(uri="orchestrai://registry")
        )


@pytest.mark.parametrize("tool", ["reload_config", "probe_providers"])
async def test_admin_tools_allow_server_granted_scope(tmp_path: Path, tool: str) -> None:
    orch = setup_owners(tmp_path)
    grant = config.get_settings().server.sse.tokens[0]
    replace_grants([grant.model_copy(update={"scopes": ("admin",)})])
    orch._settings = config.get_settings()
    with (
        access("alice"),
        patch("orchestrai.config.settings.reload_settings", return_value=orch._settings),
        patch("orchestrai.providers.discovery.discover_providers", AsyncMock(return_value=[])),
    ):
        result = await handle_tool(tool, {}, orch, orch._registry)
    assert "error" not in result
    assert result.get("published", result.get("reloaded")) is True


async def test_tokens_stay_out_of_logs_and_artifacts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    import httpx

    from tests.test_sse_security import auth_header, captured_app, sse_session

    orch = setup_owners(tmp_path)
    caplog.set_level(logging.DEBUG)
    app = await captured_app()
    async with (
        sse_session(app, auth_header("ALICE")) as endpoint,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765"
        ) as client,
    ):
        response = await client.post(
            endpoint,
            headers=dict(auth_header("ALICE")),
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        assert response.status_code == 202
    with access("alice"), patch.object(orch, "_run", new_callable=AsyncMock):
        await orch.submit("Explain dummy code")
    evidence = caplog.text + repr(config.get_settings()) + config.get_settings().model_dump_json()
    evidence += "".join(path.read_text() for path in tmp_path.rglob("*.json"))
    for grant in config.get_settings().server.sse.tokens:
        assert grant.token.get_secret_value() not in evidence


@pytest.mark.parametrize("uri", [
    "orchestrai://registry", "orchestrai://status", "orchestrai://costs/summary",
])
@pytest.mark.parametrize("change", [
    "zero-scope", "revoked-before", "narrowed-before", "revoked-during", "narrowed-during",
])
async def test_resource_authorization_requires_live_read_scope(
    tmp_path: Path, uri: str, change: str
) -> None:
    import asyncio

    from mcp import types

    from orchestrai.server import main

    orch = setup_owners(tmp_path)
    grant = config.get_settings().server.sse.tokens[0]
    if change == "zero-scope":
        replace_grants([grant.model_copy(update={"scopes": ()})])

    def restrict() -> None:
        replace_grants(
            [] if change.startswith("revoked") else [grant.model_copy(update={"scopes": ()})]
        )

    async def discover() -> Orchestrator:
        await asyncio.sleep(0)
        if change.endswith("during"):
            restrict()
        return orch

    server = main._build_mcp_server()
    entry = server.get_request_handler("resources/read")
    assert entry is not None
    with (
        access("alice"),
        patch.object(main, "get_orchestrator", side_effect=discover) as get_orchestrator,
        patch.object(main, "_registry", orch._registry),
        patch.object(orch._registry, "to_dict", return_value={}) as registry_read,
        patch.object(orch, "get_active_tasks", return_value=[]) as status_read,
        patch.object(orch, "get_cost_summary", return_value={}) as costs_read,
    ):
        if change.endswith("before"):
            restrict()
        with pytest.raises(PermissionError, match="Access denied"):
            await entry.handler(cast(Any, None), types.ReadResourceRequestParams(uri=uri))
        if not change.endswith("during"):
            get_orchestrator.assert_not_called()
        for read in (registry_read, status_read, costs_read):
            read.assert_not_called()


@pytest.mark.parametrize("wait", [True, False])
async def test_late_prepare_failure_leaves_no_runtime_or_admission_residue(
    tmp_path: Path, wait: bool
) -> None:
    from orchestrai.orchestrator.orchestrator import MODE_MAP
    from tests.test_sse_security import update_sse

    orch = setup_owners(tmp_path)
    update_sse(max_active_tasks=1, max_active_tasks_per_principal=1)
    with access("alice"):
        constructor = Mock(side_effect=RuntimeError("dummy executor construction failure"))
        with (
            patch.dict(MODE_MAP, {"docs": constructor}),
            pytest.raises(RuntimeError, match="dummy executor construction failure"),
        ):
            await orch.submit("Explain dummy code", mode="docs", wait=wait)
        constructor.assert_called_once()
        assert constructor.call_args.kwargs["store"]._task_id
        assert constructor.call_args.kwargs["tracer"].task_id
        for state in (
            orch._owners, orch._active, orch._stores, orch._events, orch._bg_tasks,
            orch.task_limits._active,
        ):
            assert state == {}
        with patch.object(orch, "_run", new_callable=AsyncMock):
            task = await orch.submit("Explain dummy code", mode="docs")
        assert orch._owners[task.id] is not None
        assert not orch.task_limits._active
