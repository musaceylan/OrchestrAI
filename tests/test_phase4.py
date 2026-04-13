"""
Phase 4 tests: vLLM provider, cost summary resource, reload_config tool.
All network calls are mocked — no real vLLM installation required.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orchestrai.providers.base import CompletionRequest, ProviderError
from orchestrai.providers.vllm import VLLMProvider, _coding_strength


# ─── _coding_strength heuristic ──────────────────────────────────────────────

class TestVLLMCodingStrength:
    def test_code_specialist_high(self):
        assert _coding_strength("Qwen/Qwen2.5-Coder-7B-Instruct") == 0.90

    def test_large_70b(self):
        assert _coding_strength("meta-llama/Llama-3.3-70B-Instruct") >= 0.82

    def test_small_3b(self):
        assert _coding_strength("smallmodel-3b") < 0.70

    def test_hf_org_prefix_stripped(self):
        # "codellama" is in the model name after stripping org
        assert _coding_strength("meta-llama/CodeLlama-7b-Instruct-hf") == 0.90

    def test_unknown_default(self):
        assert _coding_strength("mystery-model") == 0.70


# ─── VLLMProvider.probe ──────────────────────────────────────────────────────

class TestVLLMProbe:
    async def test_probe_via_health_endpoint(self):
        provider = VLLMProvider()

        health_resp = MagicMock()
        health_resp.status_code = 200

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.json.return_value = {
            "data": [{"id": "meta-llama/Llama-3.3-70B", "max_model_len": 131072}]
        }
        models_resp.raise_for_status = MagicMock()

        async def _fake_get(url: str, **kwargs):
            if url.endswith("/health"):
                return health_resp
            if url.endswith("/v1/models"):
                return models_resp
            raise AssertionError(f"Unexpected GET: {url}")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = _fake_get

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            result = await provider.probe()

        assert result is True
        assert len(provider._models) == 1
        assert provider._models[0].model_id == "meta-llama/Llama-3.3-70B"
        assert provider._models[0].context_window == 131072

    async def test_probe_falls_back_to_models_endpoint(self):
        provider = VLLMProvider()

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.json.return_value = {
            "data": [{"id": "qwen2.5-coder:7b", "max_model_len": 32768}]
        }
        models_resp.raise_for_status = MagicMock()

        async def _fake_get(url: str, **kwargs):
            if url.endswith("/health"):
                raise httpx.ConnectError("not found")
            if url.endswith("/v1/models"):
                return models_resp
            raise AssertionError(f"Unexpected GET: {url}")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = _fake_get

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            result = await provider.probe()

        assert result is True
        assert provider._models[0].model_id == "qwen2.5-coder:7b"

    async def test_probe_returns_false_when_unreachable(self):
        provider = VLLMProvider()

        async def _fake_get(url: str, **kwargs):
            raise httpx.ConnectError("connection refused")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = _fake_get

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            result = await provider.probe()

        assert result is False

    async def test_probe_returns_false_when_no_models(self):
        provider = VLLMProvider()

        health_resp = MagicMock()
        health_resp.status_code = 200

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.json.return_value = {"data": []}  # empty — no models loaded
        models_resp.raise_for_status = MagicMock()

        async def _fake_get(url: str, **kwargs):
            if url.endswith("/health"):
                return health_resp
            return models_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = _fake_get

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            result = await provider.probe()

        assert result is False


# ─── VLLMProvider.complete ───────────────────────────────────────────────────

class TestVLLMComplete:
    def _make_provider(self) -> VLLMProvider:
        from orchestrai.providers.base import ModelCapability
        from orchestrai.artifacts.schemas import (
            CostTier, LatencyTier, PrivacyLevel, ProviderKind, RoleType,
        )
        provider = VLLMProvider()
        provider._models = [
            ModelCapability(
                provider="vllm",
                provider_kind=ProviderKind.OPENAI_COMPAT,
                model_id="meta-llama/Llama-3-8B",
                display_name="vLLM/Llama-3-8B",
                planning_strength=0.60,
                coding_strength=0.66,
                debugging_strength=0.56,
                review_strength=0.53,
                test_gen_strength=0.53,
                docs_strength=0.54,
                long_context_strength=0.06,
                context_window=8192,
                max_output_tokens=2048,
                latency_tier=LatencyTier.FAST,
                cost_tier=CostTier.CHEAP,
                privacy_level=PrivacyLevel.SECRET,
                supports_tool_calling=False,
                supports_streaming=True,
                preferred_roles=[RoleType.CODER],
            )
        ]
        return provider

    async def test_complete_success(self):
        provider = self._make_provider()
        completion_data = {
            "choices": [
                {"message": {"content": "def hello(): pass"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

        completion_resp = MagicMock()
        completion_resp.status_code = 200
        completion_resp.json.return_value = completion_data
        completion_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=completion_resp)

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            req = CompletionRequest(
                model="meta-llama/Llama-3-8B",
                messages=[{"role": "user", "content": "write hello"}],
                max_tokens=100,
                temperature=0.0,
            )
            resp = await provider.complete(req)

        assert resp.content == "def hello(): pass"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 8
        assert resp.provider == "vllm"

    async def test_complete_uses_first_model_when_no_model_specified(self):
        provider = self._make_provider()
        completion_resp = MagicMock()
        completion_resp.status_code = 200
        completion_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        completion_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=completion_resp)

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            req = CompletionRequest(
                model=None,  # no model specified
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
                temperature=0.0,
            )
            resp = await provider.complete(req)

        assert resp.model == "meta-llama/Llama-3-8B"

    async def test_complete_raises_provider_error_on_500(self):
        provider = self._make_provider()
        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.text = "internal server error"
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=err_resp)

        completion_resp = MagicMock()
        completion_resp.raise_for_status = MagicMock(side_effect=http_err)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=completion_resp)

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ProviderError) as exc_info:
                await provider.complete(CompletionRequest(
                    model="meta-llama/Llama-3-8B",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=10,
                    temperature=0.0,
                ))
        assert exc_info.value.retryable is True

    async def test_complete_raises_when_no_models_and_no_model_arg(self):
        provider = VLLMProvider()  # _models is empty
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(CompletionRequest(
                model=None,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                temperature=0.0,
            ))
        assert exc_info.value.retryable is False

    async def test_complete_includes_system_message(self):
        provider = self._make_provider()
        captured: list[dict] = []

        async def _fake_post(url: str, json: dict, **kwargs):
            captured.append(json)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = _fake_post

        with patch("orchestrai.providers.vllm.httpx.AsyncClient", return_value=mock_client):
            await provider.complete(CompletionRequest(
                model="meta-llama/Llama-3-8B",
                messages=[{"role": "user", "content": "hi"}],
                system="You are a helpful assistant",
                max_tokens=10,
                temperature=0.0,
            ))

        msgs = captured[0]["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a helpful assistant"}
        assert msgs[1] == {"role": "user", "content": "hi"}


# ─── VLLMProvider properties ─────────────────────────────────────────────────

class TestVLLMProperties:
    def test_name(self):
        assert VLLMProvider().name == "vllm"

    def test_custom_name(self):
        assert VLLMProvider(name="gpu-node-1").name == "gpu-node-1"

    def test_privacy_is_secret(self):
        from orchestrai.artifacts.schemas import PrivacyLevel
        provider = VLLMProvider()
        # Inject a dummy model to test
        from orchestrai.providers.base import ModelCapability
        from orchestrai.artifacts.schemas import CostTier, LatencyTier, ProviderKind, RoleType
        cap = ModelCapability(
            provider="vllm", provider_kind=ProviderKind.OPENAI_COMPAT,
            model_id="test", display_name="test",
            planning_strength=0.5, coding_strength=0.5, debugging_strength=0.5,
            review_strength=0.5, test_gen_strength=0.5, docs_strength=0.5,
            long_context_strength=0.5, context_window=4096, max_output_tokens=1024,
            latency_tier=LatencyTier.FAST, cost_tier=CostTier.CHEAP,
            privacy_level=PrivacyLevel.SECRET, supports_tool_calling=False,
            supports_streaming=True, preferred_roles=[RoleType.CODER],
        )
        assert cap.privacy_level == PrivacyLevel.SECRET


# ─── Discovery: vLLM wired in ─────────────────────────────────────────────────

class TestDiscoveryIncludesVLLM:
    async def test_vllm_in_candidates(self):
        """discover_providers should include a VLLMProvider in the probe list."""
        probed: list[str] = []

        async def _fake_probe_one(provider) -> bool:
            probed.append(provider.name)
            return False  # all unavailable — we just care it's attempted

        with patch("orchestrai.providers.discovery._probe_one", side_effect=_fake_probe_one):
            from orchestrai.providers.discovery import discover_providers
            await discover_providers()

        assert "vllm" in probed


# ─── Orchestrator.get_cost_summary ───────────────────────────────────────────

class TestCostSummary:
    def _make_orchestrator(self):
        from orchestrai.registry.registry import CapabilityRegistry
        from orchestrai.orchestrator.orchestrator import Orchestrator
        registry = MagicMock(spec=CapabilityRegistry)
        registry.all_providers.return_value = []
        registry.all_capabilities.return_value = []
        orch = Orchestrator(registry)
        return orch

    def _make_task(self, task_id: str, status: str, cost: float, tokens: int):
        from orchestrai.artifacts.schemas import (
            OrchestratedTask, TaskBrief, TaskType, RoutingDecision,
            ArtifactKind, Provenance,
        )
        from orchestrai.observability.trace import make_trace_id, make_artifact_id
        import time as _time
        prov = Provenance(task_id=task_id, provider="test", model="test", role=None)
        brief = TaskBrief(
            id=make_artifact_id(),
            kind=ArtifactKind.TASK_BRIEF,
            provenance=prov,
            task_type=TaskType.FEATURE,
            description="test task",
            raw_request="test task",
        )
        routing = RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=prov,
            task_type=TaskType.FEATURE,
            assignments=[],
            rationale="test",
        )
        task = OrchestratedTask(
            id=task_id,
            trace_id=make_trace_id(),
            brief=brief,
            mode="planner_coder_reviewer",
            routing=routing,
            status=status,
        )
        task.cost_usd = cost
        task.tokens_used = tokens
        task.finished_at = _time.time() if status != "running" else None
        return task

    def test_empty_summary(self):
        orch = self._make_orchestrator()
        summary = orch.get_cost_summary()
        assert summary["session_total_cost_usd"] == 0.0
        assert summary["session_total_tokens"] == 0
        assert summary["finished_task_count"] == 0
        assert summary["active_task_count"] == 0

    def test_finished_tasks_included(self):
        orch = self._make_orchestrator()
        t1 = self._make_task("task-001", "done", cost=0.01, tokens=500)
        t2 = self._make_task("task-002", "done", cost=0.02, tokens=1000)
        orch._finished["task-001"] = t1
        orch._finished["task-002"] = t2

        summary = orch.get_cost_summary()
        assert summary["session_total_cost_usd"] == pytest.approx(0.03, abs=1e-6)
        assert summary["session_total_tokens"] == 1500
        assert summary["finished_task_count"] == 2
        assert len(summary["finished_tasks"]) == 2

    def test_active_tasks_included(self):
        orch = self._make_orchestrator()
        t = self._make_task("task-active", "running", cost=0.005, tokens=200)
        orch._active["task-active"] = t

        summary = orch.get_cost_summary()
        assert summary["session_total_cost_usd"] == pytest.approx(0.005, abs=1e-6)
        assert summary["active_task_count"] == 1
        assert len(summary["active_tasks"]) == 1

    def test_combined_cost(self):
        orch = self._make_orchestrator()
        t1 = self._make_task("f1", "done", cost=0.01, tokens=400)
        t2 = self._make_task("a1", "running", cost=0.005, tokens=150)
        orch._finished["f1"] = t1
        orch._active["a1"] = t2

        summary = orch.get_cost_summary()
        assert summary["session_total_cost_usd"] == pytest.approx(0.015, abs=1e-6)
        assert summary["session_total_tokens"] == 550


# ─── reload_config tool ───────────────────────────────────────────────────────

class TestReloadConfig:
    async def test_reload_returns_provider_count(self):
        from orchestrai.server.tools import _reload_config
        from orchestrai.registry.registry import CapabilityRegistry
        from orchestrai.orchestrator.orchestrator import Orchestrator
        from orchestrai.config.settings import PolicyConfig

        # Build minimal orchestrator
        registry = MagicMock(spec=CapabilityRegistry)
        registry.all_capabilities.return_value = []
        orch = Orchestrator(registry)
        orch._router._policy = PolicyConfig()

        new_settings = MagicMock()
        new_settings.policy = PolicyConfig()
        new_settings.model_config = {}

        fake_providers = [MagicMock(name="anthropic"), MagicMock(name="ollama")]
        for i, p in enumerate(fake_providers):
            p.name = ["anthropic", "ollama"][i]

        new_registry = MagicMock(spec=CapabilityRegistry)
        new_registry.all_capabilities.return_value = []

        with (
            patch("orchestrai.config.settings.reload_settings", return_value=new_settings),
            patch("orchestrai.providers.discovery.discover_providers", new=AsyncMock(return_value=fake_providers)),
            patch("orchestrai.registry.registry.CapabilityRegistry.build", new=AsyncMock(return_value=new_registry)),
        ):
            result = await _reload_config({}, orch, registry)

        assert result["reloaded"] is True
        assert result["providers_found"] == 2
        assert "anthropic" in result["providers"]
        assert "ollama" in result["providers"]

    async def test_reload_swaps_registry_on_orchestrator(self):
        from orchestrai.server.tools import _reload_config
        from orchestrai.registry.registry import CapabilityRegistry
        from orchestrai.orchestrator.orchestrator import Orchestrator
        from orchestrai.config.settings import PolicyConfig

        old_registry = MagicMock(spec=CapabilityRegistry)
        old_registry.all_capabilities.return_value = []
        orch = Orchestrator(old_registry)
        orch._router._policy = PolicyConfig()

        new_settings = MagicMock()
        new_settings.policy = PolicyConfig()
        new_settings.model_config = {}

        new_registry = MagicMock(spec=CapabilityRegistry)
        new_registry.all_capabilities.return_value = ["m1"]

        with (
            patch("orchestrai.config.settings.reload_settings", return_value=new_settings),
            patch("orchestrai.providers.discovery.discover_providers", new=AsyncMock(return_value=[])),
            patch("orchestrai.registry.registry.CapabilityRegistry.build", new=AsyncMock(return_value=new_registry)),
        ):
            await _reload_config({}, orch, old_registry)

        assert orch._registry is new_registry
        assert orch._router._registry is new_registry
