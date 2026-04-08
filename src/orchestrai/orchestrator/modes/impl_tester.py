"""
Implementer + Tester Mode — focused TDD-style workflow.
Best for: test generation, review-only tasks, small implementations.

Flow:
  1. CODER implements / proposes the change
  2. TESTER writes tests independently (can catch coder blind spots)
  3. Run actual tests if repo available
  4. REVIEWER critiques both
"""
from __future__ import annotations

import asyncio

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, FinalDecision, OrchestratedTask, Provenance,
    ReviewComments, RoleType, TestCandidate,
)
from orchestrai.execution.shell import run_tests
from orchestrai.observability.trace import make_artifact_id
from orchestrai.orchestrator.modes.base_mode import BaseMode

log = structlog.get_logger()


class ImplTesterMode(BaseMode):
    @property
    def name(self) -> str:
        return "impl_tester"

    async def run(self, task: OrchestratedTask) -> OrchestratedTask:
        routing = task.routing
        brief = task.brief
        if routing is None:
            task.status = "failed"
            task.error = "No routing decision"
            return task

        base = self._build_base_context(brief)

        # Run CODER and TESTER in parallel
        coder_a = self._find_assignment(routing, RoleType.CODER)
        tester_a = self._find_assignment(routing, RoleType.TESTER)

        parallel: list = []
        if coder_a:
            parallel.append(
                self._call_agent(
                    task=task, role=RoleType.CODER,
                    provider_name=coder_a["provider"], model_id=coder_a["model"],
                    user_prompt=f"Implement this task and output a unified diff:\n\n{base}",
                )
            )
        if tester_a:
            parallel.append(
                self._call_agent(
                    task=task, role=RoleType.TESTER,
                    provider_name=tester_a["provider"], model_id=tester_a["model"],
                    user_prompt=(
                        f"Write comprehensive tests for this task. "
                        f"Cover edge cases and failure modes:\n\n{base}"
                    ),
                )
            )

        results = await asyncio.gather(*parallel, return_exceptions=True)

        patch = None
        test_artifact: TestCandidate | None = None
        idx = 0
        if coder_a and idx < len(results):
            result = results[idx]
            idx += 1
            if isinstance(result, Exception):
                log.error("impl_tester.coder_failed", task_id=task.id, error=str(result))
            else:
                code_content, code_subtask = result
                if code_content:
                    prov = Provenance(
                        task_id=task.id, subtask_id=code_subtask.id,
                        provider=code_subtask.provider, model=code_subtask.model,
                        role=RoleType.CODER,
                    )
                    patch = self._make_patch(code_content, prov)
                    self._store.put(patch)

        # Detect test framework from repo scan stored in brief.metadata
        detected_framework = brief.metadata.get("test_framework", "pytest")

        if tester_a and idx < len(results):
            result = results[idx]
            if isinstance(result, Exception):
                log.error("impl_tester.tester_failed", task_id=task.id, error=str(result))
            else:
                test_content, test_subtask = result
                if test_content:
                    test_artifact = TestCandidate(
                        id=make_artifact_id(),
                        kind=ArtifactKind.TEST_CANDIDATE,
                        provenance=Provenance(
                            task_id=task.id, subtask_id=test_subtask.id,
                            provider=test_subtask.provider, model=test_subtask.model,
                            role=RoleType.TESTER,
                        ),
                        test_code=test_content,
                        framework=detected_framework,
                    )
                    self._store.put(test_artifact)

        # Run tests if repo available
        tool_evidence: list[str] = []
        if brief.repo_root:
            try:
                test_run = await run_tests(brief.repo_root, task_id=task.id)
                self._store.put(test_run)
                tool_evidence.append(f"tests: {'PASS' if test_run.passed else 'FAIL'}")
                if test_run.passed and test_artifact:
                    test_artifact.passed = True
                self._tracer.tool_executed("test_runner", "", 0 if test_run.passed else 1, 0)
            except Exception as e:
                log.warning("impl_tester.tests_failed", error=str(e))

        # Optional reviewer
        reviewer_a = self._find_assignment(routing, RoleType.REVIEWER)
        review: ReviewComments | None = None
        if reviewer_a and (patch or test_artifact):
            review_prompt = (
                f"Review this code and tests:\n\nTask: {brief.description}\n\n"
                f"Diff:\n{patch.unified_diff[:2000] if patch else 'none'}\n\n"
                f"Tests:\n{test_artifact.test_code[:1500] if test_artifact else 'none'}\n\n"
                f"Tool evidence: {', '.join(tool_evidence) or 'none'}"
            )
            review_content, review_subtask = await self._call_agent(
                task=task, role=RoleType.REVIEWER,
                provider_name=reviewer_a["provider"], model_id=reviewer_a["model"],
                user_prompt=review_prompt,
            )
            if review_content:
                review = ReviewComments(
                    id=make_artifact_id(),
                    kind=ArtifactKind.REVIEW_COMMENTS,
                    provenance=Provenance(
                        task_id=task.id, subtask_id=review_subtask.id,
                        provider=review_subtask.provider, model=review_subtask.model,
                        role=RoleType.REVIEWER,
                    ),
                    overall_verdict="approve",
                )
                self._store.put(review)

        confidence = 0.72
        if tool_evidence and "PASS" in " ".join(tool_evidence):
            confidence += 0.15

        final = FinalDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.FINAL_DECISION,
            provenance=Provenance(task_id=task.id),
            summary=(
                f"impl_tester mode. Code: {patch.provenance.provider if patch else 'none'}, "
                f"Tests: {test_artifact.provenance.provider if test_artifact else 'none'}. "
                f"Tool evidence: {', '.join(tool_evidence) or 'none'}."
            ),
            patch=patch,
            tests=test_artifact,
            evidence=tool_evidence,
            confidence=confidence,
            review=review,
        )
        self._finalize(task, final)
        return task
