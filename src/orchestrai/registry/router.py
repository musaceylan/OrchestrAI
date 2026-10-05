"""
Routing engine — given a task and available capabilities, decide which model
gets which role. Uses declarative capability metadata, not hardcoded if/else.
"""
from __future__ import annotations

from typing import Any

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind,
    PrivacyLevel,
    RoleType,
    RoutingDecision,
    TaskType,
)
from orchestrai.config.settings import PolicyConfig
from orchestrai.observability.trace import make_artifact_id
from orchestrai.orchestrator.context import TaskContext
from orchestrai.registry.registry import CapabilityRegistry

log = structlog.get_logger()


# Roles required for each orchestration mode
MODE_ROLES: dict[str, list[RoleType]] = {
    "parallel_draft": [
        RoleType.CODER,
        RoleType.CODER,   # intentionally run 2+ coders in parallel
        RoleType.REVIEWER,
    ],
    "impl_tester": [
        RoleType.CODER,
        RoleType.TESTER,
        RoleType.REVIEWER,
    ],
    "planner_coder_reviewer": [
        RoleType.PLANNER,
        RoleType.CODER,
        RoleType.TESTER,
        RoleType.REVIEWER,
    ],
    "bugfix": [
        RoleType.ANALYZER,
        RoleType.CODER,
        RoleType.TESTER,
        RoleType.REVIEWER,
    ],
    "refactor": [
        RoleType.ANALYZER,
        RoleType.REFACTOR,
        RoleType.REVIEWER,
    ],
    "docs": [
        RoleType.RESEARCHER,
        RoleType.DOCUMENTER,
    ],
}

# Task type → preferred mode
TASK_MODE_MAP: dict[TaskType, str] = {
    TaskType.BUGFIX: "bugfix",
    TaskType.FEATURE: "planner_coder_reviewer",
    TaskType.REFACTOR: "refactor",
    TaskType.REVIEW: "impl_tester",
    TaskType.TEST_GENERATION: "impl_tester",
    TaskType.DOCS: "docs",
    TaskType.RESEARCH: "planner_coder_reviewer",
    TaskType.GENERAL: "planner_coder_reviewer",
}


class RoutingEngine:
    def __init__(self, registry: CapabilityRegistry, policy: PolicyConfig) -> None:
        self._registry = registry
        self._policy = policy

    def recommended_mode(self, task_type: TaskType) -> str:
        return TASK_MODE_MAP.get(task_type, "planner_coder_reviewer")

    def route(
        self,
        task_type: TaskType,
        mode: str,
        task_id: str,
        user_preferences: dict[str, Any] | None = None,
        *,
        context: TaskContext | None = None,
    ) -> RoutingDecision:
        """
        Assign a best-fit model to each role for the given mode.
        Returns a RoutingDecision artifact with full rationale.
        """
        from orchestrai.artifacts.schemas import Provenance

        prefs = user_preferences or {}
        roles = MODE_ROLES.get(mode, MODE_ROLES["planner_coder_reviewer"])

        context = context or TaskContext.resolve(self._policy, prefs)
        eligibility = context.eligibility

        assignments: list[dict[str, Any]] = []
        skipped: list[str] = []
        policy_constraints: list[str] = []
        used_models: set[str] = set()  # avoid assigning same model twice when possible

        if eligibility.local_only:
            policy_constraints.append("local_only_mode=true")
        if eligibility.privacy != PrivacyLevel.PUBLIC:
            policy_constraints.append(f"privacy_required={eligibility.privacy.value}")

        for role in roles:
            candidates = context.candidates(self._registry, role)

            if not candidates:
                skipped.append(role.value)
                log.warning("routing.no_candidate", role=role.value, task_id=task_id)
                continue

            # Prefer models not already assigned, to maximise diversity
            fresh = [c for c in candidates if f"{c.provider}/{c.model_id}" not in used_models]
            chosen = fresh[0] if fresh else candidates[0]

            # Role suffix for parallel_draft disambiguation
            role_label = role.value
            if role == RoleType.CODER and assignments:
                existing_coders = sum(1 for a in assignments if a["role"] == RoleType.CODER.value)
                if existing_coders > 0:
                    role_label = f"coder_{existing_coders + 1}"

            used_models.add(f"{chosen.provider}/{chosen.model_id}")
            assignments.append(
                {
                    "role": role_label,
                    "role_type": role.value,
                    "provider": chosen.provider,
                    "model": chosen.model_id,
                    "strength": round(chosen.strength_for_role(role), 3),
                    "cost_tier": chosen.cost_tier.value,
                    "latency_tier": chosen.latency_tier.value,
                    "privacy_level": chosen.privacy_level.value,
                }
            )

        rationale = self._build_rationale(assignments, policy_constraints, mode, task_type)
        log.info("routing.decided", task_id=task_id, mode=mode, assignments=assignments)

        return RoutingDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.ROUTING_DECISION,
            provenance=Provenance(task_id=task_id),
            task_type=task_type,
            assignments=assignments,
            rationale=rationale,
            skipped_providers=skipped,
            policy_constraints=policy_constraints,
        )

    def _build_rationale(
        self,
        assignments: list[dict[str, Any]],
        constraints: list[str],
        mode: str,
        task_type: TaskType,
    ) -> str:
        lines = [f"Mode: {mode} | Task: {task_type.value}"]
        for a in assignments:
            lines.append(
                f"  {a['role']:20s} → {a['provider']}/{a['model']} "
                f"(strength={a['strength']}, cost={a['cost_tier']}, privacy={a['privacy_level']})"
            )
        if constraints:
            lines.append(f"Policy constraints: {', '.join(constraints)}")
        return "\n".join(lines)
