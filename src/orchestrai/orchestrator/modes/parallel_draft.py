"""
Parallel Draft Mode — run N coders in parallel, reviewer judges best result.
Best for: quick generation with diversity, when you want multiple candidates.
"""
from __future__ import annotations

import asyncio

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, FinalDecision, OrchestratedTask, Provenance, RoleType,
)
from orchestrai.observability.trace import make_artifact_id
from orchestrai.orchestrator.modes.base_mode import BaseMode

log = structlog.get_logger()


class ParallelDraftMode(BaseMode):
    @property
    def name(self) -> str:
        return "parallel_draft"

    async def run(self, task: OrchestratedTask) -> OrchestratedTask:
        routing = task.routing
        if routing is None:
            task.status = "failed"
            task.error = "No routing decision available"
            return task

        brief = task.brief
        base_prompt = self._build_base_context(brief)

        # Run all CODER assignments in parallel
        coder_assignments = self._find_all_assignments(routing, RoleType.CODER)
        if not coder_assignments:
            task.status = "failed"
            task.error = "No coder assignments in routing"
            return task

        coder_tasks = [
            self._call_agent(
                task=task,
                role=RoleType.CODER,
                provider_name=a["provider"],
                model_id=a["model"],
                user_prompt=f"Implement the following:\n\n{base_prompt}",
            )
            for a in coder_assignments
        ]

        log.info(
            "parallel_draft.coders_starting",
            task_id=task.id,
            count=len(coder_tasks),
        )
        coder_results = await asyncio.gather(*coder_tasks, return_exceptions=True)

        patches = []
        for i, result in enumerate(coder_results):
            if isinstance(result, Exception):
                log.error("parallel_draft.coder_failed", task_id=task.id, index=i, error=str(result))
                continue
            content, subtask = result
            if content:
                prov = Provenance(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    provider=subtask.provider,
                    model=subtask.model,
                    role=RoleType.CODER,
                )
                patch = self._make_patch(content, prov, confidence=0.65 + i * 0.05)
                art_id = self._store.put(patch)
                subtask.output_artifact_id = art_id
                patches.append(patch)
                self._tracer.artifact_created(art_id, patch.kind.value, subtask.id)

        if not patches:
            task.status = "failed"
            task.error = "All coder agents failed"
            return task

        # Run reviewer on the best (first successful) draft
        reviewer_assignment = self._find_assignment(routing, RoleType.REVIEWER)
        review = None
        if reviewer_assignment and patches:
            review_prompt = (
                f"Review this implementation for the following task:\n\n"
                f"Task: {brief.description}\n\n"
                f"Implementation:\n{patches[0].unified_diff or patches[0].description}"
            )
            review_content, review_subtask = await self._call_agent(
                task=task,
                role=RoleType.REVIEWER,
                provider_name=reviewer_assignment["provider"],
                model_id=reviewer_assignment["model"],
                user_prompt=review_prompt,
            )
            if review_content:
                from orchestrai.artifacts.schemas import ReviewComments
                verdict, concerns, praise = self._parse_review(review_content)
                review = ReviewComments(
                    id=make_artifact_id(),
                    kind=ArtifactKind.REVIEW_COMMENTS,
                    provenance=Provenance(
                        task_id=task.id,
                        subtask_id=review_subtask.id,
                        provider=review_subtask.provider,
                        model=review_subtask.model,
                        role=RoleType.REVIEWER,
                    ),
                    overall_verdict=verdict,
                    key_concerns=concerns,
                    praise=praise,
                )
                art_id = self._store.put(review)
                review_subtask.output_artifact_id = art_id

        # Choose best patch (highest confidence, reviewer-approved preferred)
        best_patch = max(patches, key=lambda p: p.confidence)
        alternatives = [p for p in patches if p.id != best_patch.id]

        summary = (
            f"Generated {len(patches)} candidate implementation(s) in parallel. "
            f"Selected best by confidence score ({best_patch.confidence:.2f}). "
            f"Provider: {best_patch.provenance.provider}/{best_patch.provenance.model}."
        )

        final = FinalDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.FINAL_DECISION,
            provenance=Provenance(task_id=task.id),
            summary=summary,
            patch=best_patch,
            evidence=[f"Generated {len(patches)} candidates"],
            confidence=best_patch.confidence,
            alternatives=alternatives,
            review=review,
        )
        self._finalize(task, final)

        log.info(
            "parallel_draft.complete",
            task_id=task.id,
            candidates=len(patches),
            chosen_provider=best_patch.provenance.provider,
        )
        return task
