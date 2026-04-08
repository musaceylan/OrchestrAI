"""
Planner → Coder → Tester → Reviewer pipeline.
The reference orchestration mode for complex features and bugfixes.

Flow:
  1. PLANNER analyses task, produces structured plan
  2. CODER implements based on plan  [parallel with TESTER if possible]
  3. TESTER writes tests based on plan + code
  4. REVIEWER critiques code + tests
  5. Merger/Judge selects final result
"""
from __future__ import annotations

import asyncio
import json

import structlog

from orchestrai.artifacts.schemas import (
    ArtifactKind, FinalDecision, ImplementationPlan, OrchestratedTask, Provenance,
    ReviewComments, RoleType, TestCandidate,
)
from orchestrai.execution.shell import run_lint, run_tests
from orchestrai.observability.trace import make_artifact_id
from orchestrai.orchestrator.modes.base_mode import BaseMode

log = structlog.get_logger()


class PlannerCoderReviewerMode(BaseMode):
    @property
    def name(self) -> str:
        return "planner_coder_reviewer"

    async def run(self, task: OrchestratedTask) -> OrchestratedTask:
        routing = task.routing
        brief = task.brief

        if routing is None:
            task.status = "failed"
            task.error = "No routing decision"
            return task

        base_context = self._build_base_context(brief)

        # ── Step 1: PLAN ──────────────────────────────────────────────────────
        plan_assignment = self._find_assignment(routing, RoleType.PLANNER)
        plan_text = ""
        plan_artifact: ImplementationPlan | None = None

        if plan_assignment:
            log.info("pcr.planning", task_id=task.id, model=plan_assignment["model"])
            plan_prompt = (
                f"Analyse this software engineering task and produce a structured implementation plan.\n\n"
                f"{base_context}\n\n"
                f"Output JSON with keys: steps (list of {{step, files, approach}}), "
                f"risks (list of strings), estimated_complexity (low|medium|high|very_high)."
            )
            plan_text, plan_subtask = await self._call_agent(
                task=task,
                role=RoleType.PLANNER,
                provider_name=plan_assignment["provider"],
                model_id=plan_assignment["model"],
                user_prompt=plan_prompt,
            )

            if plan_text:
                steps = self._parse_plan_steps(plan_text)
                plan_artifact = ImplementationPlan(
                    id=make_artifact_id(),
                    kind=ArtifactKind.IMPLEMENTATION_PLAN,
                    provenance=Provenance(
                        task_id=task.id,
                        subtask_id=plan_subtask.id,
                        provider=plan_subtask.provider,
                        model=plan_subtask.model,
                        role=RoleType.PLANNER,
                    ),
                    steps=steps,
                    rationale=plan_text[:1000],
                )
                art_id = self._store.put(plan_artifact)
                plan_subtask.output_artifact_id = art_id
                self._tracer.artifact_created(art_id, plan_artifact.kind.value, plan_subtask.id)

        # ── Step 2: CODE + TEST in parallel ───────────────────────────────────
        coder_assignment = self._find_assignment(routing, RoleType.CODER)
        tester_assignment = self._find_assignment(routing, RoleType.TESTER)

        plan_context = f"\n\nImplementation plan:\n{plan_text}" if plan_text else ""

        code_prompt = (
            f"Implement the following task. Output a unified diff.\n\n"
            f"{base_context}{plan_context}"
        )
        test_prompt = (
            f"Write tests for the following task.\n\n"
            f"{base_context}{plan_context}\n\n"
            f"Focus on unit tests for the core logic. Use the appropriate test framework."
        )

        parallel_calls = []
        if coder_assignment:
            parallel_calls.append(
                self._call_agent(
                    task=task,
                    role=RoleType.CODER,
                    provider_name=coder_assignment["provider"],
                    model_id=coder_assignment["model"],
                    user_prompt=code_prompt,
                )
            )
        if tester_assignment:
            parallel_calls.append(
                self._call_agent(
                    task=task,
                    role=RoleType.TESTER,
                    provider_name=tester_assignment["provider"],
                    model_id=tester_assignment["model"],
                    user_prompt=test_prompt,
                )
            )

        log.info("pcr.coder_tester_parallel", task_id=task.id, calls=len(parallel_calls))
        parallel_results = await asyncio.gather(*parallel_calls, return_exceptions=True)

        code_content = ""
        test_content = ""
        code_subtask = None
        test_subtask = None

        idx = 0
        if coder_assignment:
            result = parallel_results[idx]
            if isinstance(result, Exception):
                log.error("pcr.coder_failed", task_id=task.id, error=str(result))
            else:
                code_content, code_subtask = result
            idx += 1
        if tester_assignment:
            result = parallel_results[idx]
            if isinstance(result, Exception):
                log.error("pcr.tester_failed", task_id=task.id, error=str(result))
            else:
                test_content, test_subtask = result

        # Build patch artifact
        patch = None
        if code_content and code_subtask:
            prov = Provenance(
                task_id=task.id, subtask_id=code_subtask.id,
                provider=code_subtask.provider, model=code_subtask.model,
                role=RoleType.CODER,
            )
            patch = self._make_patch(code_content, prov)
            art_id = self._store.put(patch)
            code_subtask.output_artifact_id = art_id
            self._tracer.artifact_created(art_id, patch.kind.value, code_subtask.id)

        # Build test candidate artifact
        test_artifact: TestCandidate | None = None
        if test_content and test_subtask:
            test_artifact = TestCandidate(
                id=make_artifact_id(),
                kind=ArtifactKind.TEST_CANDIDATE,
                provenance=Provenance(
                    task_id=task.id, subtask_id=test_subtask.id,
                    provider=test_subtask.provider, model=test_subtask.model,
                    role=RoleType.TESTER,
                ),
                test_code=test_content,
                framework=brief.metadata.get("test_framework", "pytest"),
                covers_cases=[],
            )
            art_id = self._store.put(test_artifact)
            test_subtask.output_artifact_id = art_id
            self._tracer.artifact_created(art_id, test_artifact.kind.value, test_subtask.id)

        # ── Step 3: Run lint/tests if repo available ───────────────────────────
        tool_evidence: list[str] = []
        if brief.repo_root:
            try:
                lint_result = await run_lint(brief.repo_root, task_id=task.id)
                self._store.put(lint_result)
                if lint_result.passed:
                    tool_evidence.append("lint: PASS")
                else:
                    tool_evidence.append(
                        f"lint: FAIL ({lint_result.error_count} errors, {lint_result.warning_count} warnings)"
                    )
                self._tracer.tool_executed(
                    "lint", lint_result.tool, 0 if lint_result.passed else 1, 0
                )
            except Exception as e:
                log.warning("pcr.lint_failed", error=str(e))

        # ── Step 4: REVIEW ────────────────────────────────────────────────────
        reviewer_assignment = self._find_assignment(routing, RoleType.REVIEWER)
        review: ReviewComments | None = None

        if reviewer_assignment and (patch or test_artifact):
            patch_text = patch.unified_diff if patch else "(no patch generated)"
            test_text = test_artifact.test_code if test_artifact else "(no tests generated)"

            review_prompt = (
                f"Review the following implementation and tests for the task below.\n\n"
                f"Task: {brief.description}\n\n"
                f"Implementation (diff):\n{patch_text[:3000]}\n\n"
                f"Tests:\n{test_text[:2000]}\n\n"
                f"Tool results: {', '.join(tool_evidence) or 'none'}\n\n"
                f"Output JSON with: overall_verdict, comments (list), key_concerns (list), praise (list)."
            )
            review_content, review_subtask = await self._call_agent(
                task=task,
                role=RoleType.REVIEWER,
                provider_name=reviewer_assignment["provider"],
                model_id=reviewer_assignment["model"],
                user_prompt=review_prompt,
            )
            if review_content:
                verdict, concerns, praise = self._parse_review(review_content)
                review = ReviewComments(
                    id=make_artifact_id(),
                    kind=ArtifactKind.REVIEW_COMMENTS,
                    provenance=Provenance(
                        task_id=task.id, subtask_id=review_subtask.id,
                        provider=review_subtask.provider, model=review_subtask.model,
                        role=RoleType.REVIEWER,
                    ),
                    overall_verdict=verdict,
                    key_concerns=concerns,
                    praise=praise,
                )
                art_id = self._store.put(review)
                review_subtask.output_artifact_id = art_id

        # ── Step 5: Assemble final result ─────────────────────────────────────
        confidence = 0.70
        if tool_evidence and "PASS" in " ".join(tool_evidence):
            confidence += 0.10
        if review and review.overall_verdict == "approve":
            confidence += 0.10
        if plan_artifact:
            confidence += 0.05

        summary_parts = [f"Orchestrated in '{self.name}' mode."]
        if plan_artifact:
            summary_parts.append(f"Plan: {len(plan_artifact.steps)} steps.")
        if patch:
            summary_parts.append(f"Implementation: {patch.provenance.provider}/{patch.provenance.model}.")
        if test_artifact:
            summary_parts.append(f"Tests: {test_artifact.provenance.provider}/{test_artifact.provenance.model}.")
        if review:
            summary_parts.append(f"Review verdict: {review.overall_verdict}.")
        if tool_evidence:
            summary_parts.append(f"Tool results: {', '.join(tool_evidence)}.")

        final = FinalDecision(
            id=make_artifact_id(),
            kind=ArtifactKind.FINAL_DECISION,
            provenance=Provenance(task_id=task.id),
            summary=" ".join(summary_parts),
            patch=patch,
            tests=test_artifact,
            evidence=tool_evidence,
            confidence=confidence,
            review=review,
        )
        self._finalize(task, final)

        log.info("pcr.complete", task_id=task.id, confidence=confidence)
        return task

    def _parse_plan_steps(self, text: str) -> list[dict]:
        try:
            import re
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("steps", [])
        except Exception:
            pass
        return [{"step": line.strip()} for line in text.split("\n") if line.strip()]
