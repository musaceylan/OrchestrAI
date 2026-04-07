"""
MCP Tool definitions and dispatch for OrchestrAI.
"""
from __future__ import annotations

from typing import Any

import structlog
from mcp.types import Tool

from orchestrai.orchestrator.judge import run_judge
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.registry.registry import CapabilityRegistry

log = structlog.get_logger()

# ── Tool schemas ──────────────────────────────────────────────────────────────

def build_tools() -> list[Tool]:
    return [
        Tool(
            name="submit_task",
            description=(
                "Submit a software engineering task for orchestrated multi-model execution. "
                "Automatically routes to the best combination of models for planning, coding, "
                "testing, and review."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task description (e.g. 'Add pagination to the users endpoint')",
                    },
                    "repo_root": {
                        "type": "string",
                        "description": "Absolute path to the repository root (enables static analysis and test runs)",
                    },
                    "target_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to focus on",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["parallel_draft", "impl_tester", "planner_coder_reviewer", "bugfix", "refactor", "docs"],
                        "description": "Orchestration mode (auto-detected if omitted)",
                    },
                    "user_preferences": {
                        "type": "object",
                        "description": "Override routing: {preferred_providers: [...], privacy_level: ..., cost_tier: ...}",
                    },
                },
                "required": ["request"],
            },
        ),
        Tool(
            name="inspect_plan",
            description="Get the implementation plan generated for a task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from submit_task"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="inspect_registry",
            description="List all available providers and their model capabilities, grouped by role strengths.",
            inputSchema={
                "type": "object",
                "properties": {
                    "role_filter": {
                        "type": "string",
                        "enum": ["planner", "coder", "tester", "reviewer", "judge"],
                        "description": "Filter by role (optional)",
                    },
                },
            },
        ),
        Tool(
            name="inspect_agents",
            description="Show which models are currently assigned to each role for a given task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="inspect_artifacts",
            description="List or retrieve artifacts (patches, tests, reviews) for a task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "kind": {
                        "type": "string",
                        "enum": ["task_brief", "code_patch", "test_candidate", "review_comments",
                                 "judge_verdict", "final_decision", "implementation_plan",
                                 "tool_result", "repo_summary", "routing_decision"],
                        "description": "Filter by artifact kind (optional)",
                    },
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="inspect_trace",
            description="Get the full execution trace for a task including timing, tokens, and decisions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="compare_candidates",
            description="Compare multiple code candidates from a parallel_draft run and get a judge verdict.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID from a parallel_draft run"},
                    "judge_model": {
                        "type": "string",
                        "description": "Specific model to use as judge (optional, uses best available)",
                    },
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="rerun_with_policy",
            description="Re-run a task with different policy constraints (e.g. local-only, lower cost).",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Original task ID to re-run"},
                    "policy_overrides": {
                        "type": "object",
                        "description": "Policy fields to override: {privacy_level, cost_tier, allowed_providers, denied_providers}",
                    },
                },
                "required": ["task_id", "policy_overrides"],
            },
        ),
        Tool(
            name="list_available_models",
            description="List all models discovered across all active providers with their capabilities.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider_filter": {
                        "type": "string",
                        "description": "Filter by provider name (optional)",
                    },
                },
            },
        ),
        Tool(
            name="probe_providers",
            description="Re-probe all configured providers to check availability and update the registry.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_task_result",
            description="Get the final result of a completed task including the winning patch and verdict.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "include_diff": {
                        "type": "boolean",
                        "description": "Include full unified diff in response (default: true)",
                    },
                },
                "required": ["task_id"],
            },
        ),
    ]


# ── Tool dispatch ─────────────────────────────────────────────────────────────

async def handle_tool(
    name: str,
    arguments: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> Any:
    handlers = {
        "submit_task": _submit_task,
        "inspect_plan": _inspect_plan,
        "inspect_registry": _inspect_registry,
        "inspect_agents": _inspect_agents,
        "inspect_artifacts": _inspect_artifacts,
        "inspect_trace": _inspect_trace,
        "compare_candidates": _compare_candidates,
        "rerun_with_policy": _rerun_with_policy,
        "list_available_models": _list_available_models,
        "probe_providers": _probe_providers,
        "get_task_result": _get_task_result,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return await handler(arguments, orchestrator, registry)
    except Exception as e:
        log.exception("tool.error", tool=name, error=str(e))
        return {"error": str(e), "tool": name}


async def _submit_task(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    task = await orchestrator.submit(
        request=args["request"],
        repo_root=args.get("repo_root"),
        target_files=args.get("target_files"),
        mode=args.get("mode"),
        user_preferences=args.get("user_preferences"),
    )
    result: dict[str, Any] = {
        "task_id": task.id,
        "status": task.status,
        "mode": task.mode,
        "task_type": task.brief.task_type.value,
    }
    if task.error:
        result["error"] = task.error
    if task.final:
        f = task.final
        result["summary"] = f.summary
        result["confidence"] = f.confidence
        result["evidence"] = f.evidence
        if f.review:
            result["review_verdict"] = f.review.overall_verdict
            result["review_concerns"] = f.review.key_concerns
        if f.patch:
            result["patch_provider"] = f.patch.provenance.provider
            result["patch_model"] = f.patch.provenance.model
            result["has_diff"] = bool(f.patch.unified_diff)
        if f.tests:
            result["tests_provider"] = f.tests.provenance.provider
            result["test_framework"] = f.tests.framework
    return result


async def _inspect_plan(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    task_id = args["task_id"]
    # Look in active tasks first, then completed
    active = orchestrator._active.get(task_id)
    if active and active.routing:
        return {
            "task_id": task_id,
            "mode": active.mode,
            "routing_rationale": active.routing.rationale,
            "assignments": active.routing.assignments,
        }
    return {"error": f"Task {task_id} not found in active tasks"}


async def _inspect_registry(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    if registry is None:
        return {"error": "Registry not initialized"}
    role_filter = args.get("role_filter")
    data = registry.to_dict()
    if role_filter:
        from orchestrai.artifacts.schemas import RoleType
        try:
            role = RoleType(role_filter)
            caps = registry.capabilities_for_role(role)
            return {
                "role": role_filter,
                "models": [
                    {
                        "model_id": c.model_id,
                        "provider": c.provider,
                        "strength": c.role_strengths.get(role, 0.0),
                        "cost_tier": c.cost_tier.value,
                        "latency_tier": c.latency_tier.value,
                    }
                    for c in caps
                ],
            }
        except ValueError:
            pass
    return data


async def _inspect_agents(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    task_id = args["task_id"]
    active = orchestrator._active.get(task_id)
    if not active:
        return {"error": f"Task {task_id} not active"}
    assignments = active.routing.assignments if active.routing else []
    return {
        "task_id": task_id,
        "mode": active.mode,
        "assignments": assignments,
        "subtask_count": len(active.subtasks),
        "subtasks": [
            {
                "id": s.id,
                "role": s.role.value,
                "provider": s.provider,
                "model": s.model,
                "status": s.status,
            }
            for s in active.subtasks
        ],
    }


async def _inspect_artifacts(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    from orchestrai.artifacts.store import ArtifactStore
    from orchestrai.artifacts.schemas import ArtifactKind
    task_id = args["task_id"]
    store = ArtifactStore(task_id)
    kind_filter = args.get("kind")
    if kind_filter:
        try:
            kind = ArtifactKind(kind_filter)
            artifacts = store.list_by_kind(kind)
        except ValueError:
            return {"error": f"Unknown artifact kind: {kind_filter}"}
    else:
        artifacts = list(store.all().values())
    return {
        "task_id": task_id,
        "count": len(artifacts),
        "artifacts": [
            {
                "id": a.get("id"),
                "kind": a.get("kind"),
                "provider": a.get("provenance", {}).get("provider"),
                "model": a.get("provenance", {}).get("model"),
                "role": a.get("provenance", {}).get("role"),
            }
            for a in artifacts
        ],
    }


async def _inspect_trace(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    import json
    from pathlib import Path
    task_id = args["task_id"]
    trace_path = Path(f"/tmp/orchestrai/traces/{task_id}.jsonl")
    if not trace_path.exists():
        return {"error": f"No trace found for task {task_id}"}
    events = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return {"task_id": task_id, "event_count": len(events), "events": events}


async def _compare_candidates(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    from orchestrai.artifacts.store import ArtifactStore
    from orchestrai.artifacts.schemas import ArtifactKind, OrchestratedTask
    task_id = args["task_id"]
    store = ArtifactStore(task_id)
    patches = store.list_by_kind(ArtifactKind.CODE_PATCH)
    if not patches:
        return {"error": f"No code patches found for task {task_id}"}

    # Build a minimal task proxy
    active = orchestrator._active.get(task_id)
    if not active:
        return {"error": f"Task {task_id} not in active tasks. Already completed?"}

    if registry is None:
        return {"error": "Registry not initialized"}

    verdict = await run_judge(
        task=active,
        candidates=patches,
        registry=registry,
        judge_model_override=args.get("judge_model"),
    )
    return {
        "task_id": task_id,
        "candidates": len(patches),
        "winner_candidate_id": verdict.winner_candidate_id,
        "winner_rationale": verdict.winner_rationale,
        "confidence": verdict.confidence,
        "scores": verdict.candidate_scores,
        "rejected": verdict.alternatives_rejected,
    }


async def _rerun_with_policy(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    from orchestrai.artifacts.store import ArtifactStore
    from orchestrai.artifacts.schemas import ArtifactKind
    task_id = args["task_id"]
    overrides = args.get("policy_overrides", {})

    store = ArtifactStore(task_id)
    briefs = store.list_by_kind(ArtifactKind.TASK_BRIEF)
    if not briefs:
        return {"error": f"No task brief found for {task_id}"}
    brief = briefs[0]

    # Convert policy overrides to user_preferences format
    user_prefs = {
        "privacy_level": overrides.get("privacy_level"),
        "cost_tier": overrides.get("cost_tier"),
        "preferred_providers": overrides.get("allowed_providers", []),
        "denied_providers": overrides.get("denied_providers", []),
    }
    user_prefs = {k: v for k, v in user_prefs.items() if v}

    task = await orchestrator.submit(
        request=brief.description,
        repo_root=brief.repo_root,
        target_files=brief.target_files,
        user_preferences=user_prefs or None,
    )
    return {
        "original_task_id": task_id,
        "new_task_id": task.id,
        "status": task.status,
        "policy_overrides": overrides,
    }


async def _list_available_models(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    if registry is None:
        return {"error": "Registry not initialized"}
    provider_filter = args.get("provider_filter")
    caps = registry.all_capabilities()
    if provider_filter:
        caps = [c for c in caps if c.provider == provider_filter]
    return {
        "total": len(caps),
        "models": [
            {
                "model_id": c.model_id,
                "provider": c.provider,
                "context_window": c.context_window,
                "cost_tier": c.cost_tier.value,
                "latency_tier": c.latency_tier.value,
                "privacy_level": c.privacy_level.value,
                "strengths": {r.value: round(s, 2) for r, s in c.role_strengths.items()},
                "features": {
                    "streaming": c.supports_streaming,
                    "function_calling": c.supports_function_calling,
                    "vision": c.supports_vision,
                },
            }
            for c in caps
        ],
    }


async def _probe_providers(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    from orchestrai.providers.discovery import discover_providers
    from orchestrai.registry.registry import CapabilityRegistry as CR
    providers = await discover_providers()
    new_registry = await CR.build(providers)
    # Swap in new registry
    orchestrator._registry = new_registry
    orchestrator._router._registry = new_registry
    return {
        "providers_found": len(providers),
        "providers": [p.name for p in providers],
        "total_models": len(new_registry.all_capabilities()),
    }


async def _get_task_result(
    args: dict[str, Any],
    orchestrator: Orchestrator,
    registry: "CapabilityRegistry | None",
) -> dict[str, Any]:
    from orchestrai.artifacts.store import ArtifactStore
    from orchestrai.artifacts.schemas import ArtifactKind
    task_id = args["task_id"]
    include_diff = args.get("include_diff", True)
    store = ArtifactStore(task_id)
    finals = store.list_by_kind(ArtifactKind.FINAL_DECISION)
    if not finals:
        return {"error": f"No final decision found for task {task_id}"}
    final = finals[-1]  # raw dict from store
    result: dict[str, Any] = {
        "task_id": task_id,
        "summary": final.get("summary"),
        "confidence": final.get("confidence"),
        "evidence": final.get("evidence", []),
    }
    patch = final.get("patch")
    if patch:
        prov = patch.get("provenance", {})
        result["patch"] = {
            "provider": prov.get("provider"),
            "model": prov.get("model"),
            "description": patch.get("description"),
        }
        if include_diff:
            result["patch"]["unified_diff"] = patch.get("unified_diff")
    tests = final.get("tests")
    if tests:
        prov = tests.get("provenance", {})
        result["tests"] = {
            "provider": prov.get("provider"),
            "framework": tests.get("framework"),
            "code": (tests.get("test_code") or "")[:2000] or None,
        }
    review = final.get("review")
    if review:
        result["review"] = {
            "verdict": review.get("overall_verdict"),
            "concerns": review.get("key_concerns", []),
            "praise": review.get("praise", []),
        }
    return result
