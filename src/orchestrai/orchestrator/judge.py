"""
Judge — compares multiple candidate patches and selects the best one.
Used when parallel_draft mode generates N candidates, or when you want
an independent model to evaluate competing implementations.
"""
from __future__ import annotations

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind,
    CodePatch,
    JudgeVerdict,
    OrchestratedTask,
    Provenance,
    RoleType,
)
from orchestrai.observability.trace import make_artifact_id
from orchestrai.orchestrator.modes.base_mode import ROLE_SYSTEM_PROMPTS, _extract_json
from orchestrai.providers.base import CompletionRequest
from orchestrai.registry.registry import CapabilityRegistry

log = structlog.get_logger()


async def run_judge(
    task: OrchestratedTask,
    candidates: list[CodePatch],
    registry: CapabilityRegistry,
    judge_model_override: str | None = None,
) -> JudgeVerdict:
    """
    Select best candidate using a judge model.
    Invalid explicit overrides raise before judging. Automatic selection and
    completion failures retain the heuristic fallback.
    """
    override_cap = None
    if judge_model_override is not None:
        override_cap = registry.resolve_model_reference(judge_model_override)
        if not override_cap.available:
            raise ValueError("Judge model override is not available")

    if len(candidates) < 2:
        # Nothing to judge
        winner = candidates[0] if candidates else None
        return JudgeVerdict(
            id=make_artifact_id(),
            kind=ArtifactKind.JUDGE_VERDICT,
            provenance=Provenance(task_id=task.id),
            winner_candidate_id=winner.id if winner else None,
            winner_rationale="Only one candidate available — auto-selected.",
            confidence=0.7,
        )

    # Try to find best judge model (planning + review strength)
    judge_caps = [override_cap] if override_cap else registry.capabilities_for_role(RoleType.JUDGE)
    if not judge_caps:
        return _heuristic_judge(task, candidates)

    judge_cap = judge_caps[0]
    provider = registry.get_provider(judge_cap.provider)
    if provider is None:
        return _heuristic_judge(task, candidates)

    # Build comparison prompt
    comparison = _build_comparison(task, candidates)
    system = ROLE_SYSTEM_PROMPTS[RoleType.JUDGE]

    request = CompletionRequest(
        messages=[{"role": "user", "content": comparison}],
        system=system,
        model=judge_cap.model_id,
        max_tokens=2048,
        temperature=0.1,
        role=RoleType.JUDGE,
        task_id=task.id,
    )

    try:
        response = await provider.complete(request)
        return _parse_judge_response(task, candidates, response.content)
    except Exception as e:
        log.warning("judge.provider_failed", error=str(e))
        return _heuristic_judge(task, candidates)


def _build_comparison(task: OrchestratedTask, candidates: list[CodePatch]) -> str:
    lines = [
        f"Task: {task.brief.description}\n",
        f"Compare these {len(candidates)} candidate implementations:\n",
    ]
    for i, patch in enumerate(candidates):
        lines.append(
            f"--- Candidate {i} (provider: {patch.provenance.provider}, "
            f"model: {patch.provenance.model}) ---\n"
            f"{patch.unified_diff[:2000] or patch.description[:1000]}\n"
        )
    lines.append(
        "\nOutput JSON: {winner_index: int, rationale: str, "
        "scores: {\"0\": float, ...}, concerns: {\"0\": str, ...}}"
    )
    return "\n".join(lines)


def _parse_judge_response(
    task: OrchestratedTask,
    candidates: list[CodePatch],
    content: str,
) -> JudgeVerdict:
    data = _extract_json(content)
    if data:
        try:
            winner_idx = int(data.get("winner_index", 0))
            winner_idx = max(0, min(winner_idx, len(candidates) - 1))
            winner = candidates[winner_idx]
            scores = {
                str(i): float(data.get("scores", {}).get(str(i), 0.5))
                for i in range(len(candidates))
            }
            rationale = data.get("rationale", "")
            rejected = [
                {
                    "candidate_id": c.id,
                    "provider": c.provenance.provider,
                    "concern": data.get("concerns", {}).get(str(i), ""),
                }
                for i, c in enumerate(candidates) if c.id != winner.id
            ]
            return JudgeVerdict(
                id=make_artifact_id(),
                kind=ArtifactKind.JUDGE_VERDICT,
                provenance=Provenance(task_id=task.id),
                winner_candidate_id=winner.id,
                winner_rationale=rationale,
                candidate_scores=scores,
                evidence_used=["judge model comparison"],
                alternatives_rejected=rejected,
                confidence=scores.get(str(winner_idx), 0.8),
            )
        except Exception as e:
            log.warning("judge.parse_failed", error=str(e))
    return _heuristic_judge(task, candidates)


def _heuristic_judge(task: OrchestratedTask, candidates: list[CodePatch]) -> JudgeVerdict:
    """Score candidates heuristically when no judge model is available."""
    scored: list[tuple[float, CodePatch]] = []
    for patch in candidates:
        score = patch.confidence
        # Prefer longer diffs (more complete)
        diff_len = len(patch.unified_diff or "")
        if 100 < diff_len < 5000:
            score += 0.05
        # Prefer from higher-capability providers
        if patch.provenance.provider in ("anthropic",):
            score += 0.05
        scored.append((score, patch))

    scored.sort(key=lambda x: x[0], reverse=True)
    winner_score, winner = scored[0]
    rejected = [
        {"candidate_id": c.id, "provider": c.provenance.provider, "score": s}
        for s, c in scored[1:]
    ]

    return JudgeVerdict(
        id=make_artifact_id(),
        kind=ArtifactKind.JUDGE_VERDICT,
        provenance=Provenance(task_id=task.id),
        winner_candidate_id=winner.id,
        winner_rationale=(
            "Selected by heuristic scoring (confidence + diff quality + provider strength)."
        ),
        candidate_scores={c.id: s for s, c in scored},
        evidence_used=["heuristic scoring"],
        alternatives_rejected=rejected,
        confidence=winner_score,
    )
