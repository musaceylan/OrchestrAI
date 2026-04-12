"""
Tests for the Ollama native provider and task event streaming.

Ollama tests mock httpx so no real Ollama installation is required.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrai.providers.base import CompletionRequest, ProviderError, RoleType
from orchestrai.providers.ollama import OllamaProvider, _coding_strength


# ─── _coding_strength heuristic ──────────────────────────────────────────────

class TestCodingStrength:
    def test_code_specialist_high(self):
        assert _coding_strength("qwen2.5-coder:7b") == 0.90

    def test_large_model_high(self):
        assert _coding_strength("llama3.3:70b") >= 0.82

    def test_small_model_lower(self):
        assert _coding_strength("tinyllama:1b") < 0.80

    def test_unknown_default(self):
        assert _coding_strength("mystery-model") == 0.70


# ─── OllamaProvider.probe ────────────────────────────────────────────────────

class TestOllamaProbe:
    async def test_probe_success(self):
        provider = OllamaProvider()
        version_resp = MagicMock()
        version_resp.status_code = 200
        version_resp.json.return_value = {"version": "0.3.0"}

        tags_resp = MagicMock()
        tags_resp.status_code = 200
        tags_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b", "details": {"parameter_size": "7B"}},
            ]
        }

        show_resp = MagicMock()
        show_resp.status_code = 200
        show_resp.json.return_value = {"model_info": {"llama.context_length": 32768}}

        async def _fake_get(url: str, **kwargs):
            if url.endswith("/api/version"):
                return version_resp
            if url.endswith("/api/tags"):
                return tags_resp
            return MagicMock(status_code=404, json=lambda: {})

        async def _fake_post(url: str, **kwargs):
            return show_resp

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.get.side_effect = _fake_get
            client.post.side_effect = _fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.probe()

        assert result is True
        models = await provider.list_models()
        assert len(models) == 1
        assert models[0].model_id == "qwen2.5-coder:7b"
        assert models[0].cost_tier.value == "cheap"

    async def test_probe_connection_refused(self):
        provider = OllamaProvider()
        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.probe()

        assert result is False
        models = await provider.list_models()
        assert models == []

    async def test_probe_timeout(self):
        provider = OllamaProvider()
        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.get.side_effect = httpx.TimeoutException("Timeout")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await provider.probe()

        assert result is False


# ─── OllamaProvider context length ───────────────────────────────────────────

class TestContextLengthDiscovery:
    async def test_context_length_from_model_info(self):
        provider = OllamaProvider()

        show_resp = MagicMock()
        show_resp.status_code = 200
        show_resp.json.return_value = {"model_info": {"llama.context_length": 131072}}

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.return_value = show_resp
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            ctx_map = await provider._batch_context_lengths(["qwen2.5-coder:7b"])

        assert ctx_map["qwen2.5-coder:7b"] == 131072

    async def test_context_length_defaults_on_error(self):
        provider = OllamaProvider()
        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.side_effect = Exception("network error")
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            ctx_map = await provider._batch_context_lengths(["some-model"])

        assert ctx_map.get("some-model", 8192) == 8192

    async def test_context_length_from_details_fallback(self):
        provider = OllamaProvider()
        show_resp = MagicMock()
        show_resp.status_code = 200
        # No model_info key — falls back to details.context_length
        show_resp.json.return_value = {"details": {"context_length": 16384}}

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.return_value = show_resp
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            ctx_map = await provider._batch_context_lengths(["model-x"])

        assert ctx_map["model-x"] == 16384


# ─── OllamaProvider.complete ─────────────────────────────────────────────────

class TestOllamaComplete:
    def _make_request(self, model: str = "llama3.2") -> CompletionRequest:
        return CompletionRequest(
            messages=[{"role": "user", "content": "Write hello world"}],
            system="You are a coder.",
            model=model,
            max_tokens=256,
            temperature=0.2,
            role=RoleType.CODER,
            task_id="task-1",
            subtask_id="sub-1",
            agent_run_id="run-1",
        )

    async def test_complete_success(self):
        provider = OllamaProvider()
        api_resp = MagicMock()
        api_resp.json.return_value = {
            "message": {"role": "assistant", "content": "print('hello world')"},
            "prompt_eval_count": 20,
            "eval_count": 10,
            "done": True,
        }
        api_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.return_value = api_resp
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = await provider.complete(self._make_request())

        assert resp.content == "print('hello world')"
        assert resp.input_tokens == 20
        assert resp.output_tokens == 10
        assert resp.provider == "ollama"

    async def test_complete_http_error_retryable_on_500(self):
        provider = OllamaProvider()
        http_resp = MagicMock()
        http_resp.status_code = 503
        http_resp.text = "Service Unavailable"

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=http_resp
            )
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(self._make_request())

        assert exc_info.value.retryable is True

    async def test_complete_http_error_not_retryable_on_400(self):
        provider = OllamaProvider()
        http_resp = MagicMock()
        http_resp.status_code = 400
        http_resp.text = "Bad Request"

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.side_effect = httpx.HTTPStatusError(
                "400", request=MagicMock(), response=http_resp
            )
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(self._make_request())

        assert exc_info.value.retryable is False

    async def test_complete_uses_system_message(self):
        provider = OllamaProvider()
        captured_payload: dict = {}

        async def _capture_post(url: str, json: dict, **kwargs) -> MagicMock:
            captured_payload.update(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "message": {"content": "ok"},
                "prompt_eval_count": 5,
                "eval_count": 5,
            }
            return resp

        with patch("httpx.AsyncClient") as mock_cls:
            client = AsyncMock()
            client.post.side_effect = _capture_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await provider.complete(self._make_request())

        assert captured_payload["messages"][0]["role"] == "system"
        assert captured_payload["messages"][0]["content"] == "You are a coder."
        assert captured_payload["stream"] is False


# ─── Task event streaming ─────────────────────────────────────────────────────

class TestTaskEventStreaming:
    """Tests for Orchestrator.push_event / get_events and get_task_events tool."""

    def _make_orchestrator(self):
        from unittest.mock import MagicMock
        from orchestrai.orchestrator.orchestrator import Orchestrator
        registry = MagicMock()
        registry.all_capabilities.return_value = []
        with patch("orchestrai.orchestrator.orchestrator.RoutingEngine"):
            orch = Orchestrator.__new__(Orchestrator)
            orch._registry = registry
            orch._settings = MagicMock()
            orch._active = {}
            orch._finished = {}
            orch._bg_tasks = {}
            orch._events = {}
        return orch

    def test_push_event_appends(self):
        from orchestrai.orchestrator.orchestrator import Orchestrator
        orch = self._make_orchestrator()
        orch._events["task-1"] = []
        orch.push_event("task-1", {"event": "subtask_started", "role": "coder"})
        assert len(orch._events["task-1"]) == 1
        assert orch._events["task-1"][0]["event"] == "subtask_started"

    def test_push_event_ignored_for_unknown_task(self):
        orch = self._make_orchestrator()
        # Should not raise
        orch.push_event("nonexistent", {"event": "whatever"})

    def test_get_events_with_offset(self):
        orch = self._make_orchestrator()
        orch._events["task-1"] = [
            {"event": "task_started", "ts": 1.0},
            {"event": "subtask_started", "ts": 2.0},
            {"event": "subtask_finished", "ts": 3.0},
            {"event": "task_finished", "ts": 4.0},
        ]
        result = orch.get_events("task-1", offset=2)
        assert len(result) == 2
        assert result[0]["event"] == "subtask_finished"

    def test_get_events_empty_for_unknown_task(self):
        orch = self._make_orchestrator()
        assert orch.get_events("no-such-task") == []

    async def test_get_task_events_tool_returns_events(self):
        from orchestrai.server.tools import handle_tool
        from unittest.mock import AsyncMock, MagicMock
        orch = self._make_orchestrator()
        task_id = "test-task-01"

        # Fake task in _finished
        from orchestrai.artifacts.schemas import (
            ArtifactKind, OrchestratedTask, Provenance, RoutingDecision,
            TaskBrief, TaskType,
        )
        from orchestrai.observability.trace import make_artifact_id, make_trace_id

        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
            description="test",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="done",
        )
        orch._finished[task_id] = task
        orch._events[task_id] = [
            {"event": "task_started", "ts": 100.0, "mode": "impl_tester"},
            {"event": "task_finished", "ts": 110.0, "status": "done"},
        ]

        result = await handle_tool("get_task_events", {"task_id": task_id}, orch, None)

        assert result["task_id"] == task_id
        assert len(result["events"]) == 2
        assert result["next_offset"] == 2
        assert result["done"] is True

    async def test_get_task_events_tool_with_offset(self):
        from orchestrai.server.tools import handle_tool
        orch = self._make_orchestrator()
        task_id = "test-task-02"

        from orchestrai.artifacts.schemas import (
            ArtifactKind, OrchestratedTask, Provenance, RoutingDecision,
            TaskBrief, TaskType,
        )
        from orchestrai.observability.trace import make_artifact_id, make_trace_id

        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
            description="test",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=TaskType.GENERAL,
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="impl_tester",
            routing=routing,
            status="running",
        )
        orch._active[task_id] = task
        orch._events[task_id] = [
            {"event": "task_started", "ts": 1.0},
            {"event": "subtask_started", "ts": 2.0, "role": "coder"},
        ]

        # First poll: offset=0 → gets 2 events
        r1 = await handle_tool("get_task_events", {"task_id": task_id, "offset": 0}, orch, None)
        assert len(r1["events"]) == 2
        assert r1["next_offset"] == 2
        assert r1["done"] is False

        # Add another event
        orch._events[task_id].append({"event": "subtask_finished", "ts": 3.0})

        # Second poll: offset=2 → gets only the new event
        r2 = await handle_tool("get_task_events", {"task_id": task_id, "offset": 2}, orch, None)
        assert len(r2["events"]) == 1
        assert r2["events"][0]["event"] == "subtask_finished"
        assert r2["next_offset"] == 3
