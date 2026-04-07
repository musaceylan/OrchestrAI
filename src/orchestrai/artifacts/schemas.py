"""
Typed artifact schemas — the shared language between all agents.

Every piece of information exchanged between orchestrator and agents
is represented as a typed artifact with full provenance tracking.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    BUGFIX = "bugfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    REVIEW = "review"
    TEST_GENERATION = "test_generation"
    DOCS = "docs"
    RESEARCH = "research"
    GENERAL = "general"


class ArtifactKind(str, Enum):
    TASK_BRIEF = "task_brief"
    REPO_SUMMARY = "repo_summary"
    FILE_SUMMARY = "file_summary"
    IMPLEMENTATION_PLAN = "implementation_plan"
    CODE_PATCH = "code_patch"
    TEST_CANDIDATE = "test_candidate"
    COMMAND_OUTPUT = "command_output"
    LINT_OUTPUT = "lint_output"
    TYPE_CHECK_OUTPUT = "type_check_output"
    BENCHMARK_RESULT = "benchmark_result"
    REVIEW_COMMENTS = "review_comments"
    RISK_FLAGS = "risk_flags"
    MERGE_NOTES = "merge_notes"
    FINAL_DECISION = "final_decision"
    ROUTING_DECISION = "routing_decision"
    JUDGE_VERDICT = "judge_verdict"


class RoleType(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    ANALYZER = "analyzer"
    DOCUMENTER = "documenter"
    JUDGE = "judge"
    REFACTOR = "refactor"
    RESEARCHER = "researcher"


class PrivacyLevel(str, Enum):
    PUBLIC = "public"        # any provider OK
    INTERNAL = "internal"    # avoid external telemetry
    CONFIDENTIAL = "confidential"  # local-only
    SECRET = "secret"        # local-only, no logging


class CostTier(str, Enum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


class LatencyTier(str, Enum):
    FAST = "fast"       # <2s first token
    MEDIUM = "medium"   # 2–8s
    SLOW = "slow"       # >8s


class ProviderKind(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPAT = "openai_compat"   # Ollama, vLLM, LM Studio, etc.
    UNKNOWN = "unknown"


# ── Provenance ────────────────────────────────────────────────────────────────

class Provenance(BaseModel):
    """Tracks where an artifact came from."""
    task_id: str
    subtask_id: str | None = None
    agent_run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    role: RoleType | None = None
    created_at: float = Field(default_factory=time.time)
    parent_artifact_ids: list[str] = Field(default_factory=list)
    trace_id: str | None = None


# ── Base Artifact ─────────────────────────────────────────────────────────────

class Artifact(BaseModel):
    """Base class for all artifacts."""
    id: str
    kind: ArtifactKind
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Concrete Artifact Types ───────────────────────────────────────────────────

class TaskBrief(Artifact):
    kind: ArtifactKind = ArtifactKind.TASK_BRIEF
    task_type: TaskType
    description: str
    repo_root: str | None = None
    target_files: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    context_snippets: list[str] = Field(default_factory=list)
    raw_request: str = ""


class RepoSummary(Artifact):
    kind: ArtifactKind = ArtifactKind.REPO_SUMMARY
    language: str | None = None
    frameworks: list[str] = Field(default_factory=list)
    test_framework: str | None = None
    lint_tools: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    key_files: list[str] = Field(default_factory=list)
    total_files: int = 0
    summary_text: str = ""


class FileSummary(Artifact):
    kind: ArtifactKind = ArtifactKind.FILE_SUMMARY
    path: str
    language: str | None = None
    summary: str = ""
    key_symbols: list[str] = Field(default_factory=list)
    line_count: int = 0


class ImplementationPlan(Artifact):
    kind: ArtifactKind = ArtifactKind.IMPLEMENTATION_PLAN
    steps: list[dict[str, Any]] = Field(default_factory=list)
    estimated_complexity: str = "medium"
    risks: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    rationale: str = ""


class CodePatch(Artifact):
    kind: ArtifactKind = ArtifactKind.CODE_PATCH
    unified_diff: str = ""
    files_changed: list[str] = Field(default_factory=list)
    description: str = ""
    approach: str = ""
    confidence: float = 0.0   # 0.0–1.0
    raw_code_blocks: list[dict[str, str]] = Field(default_factory=list)


class TestCandidate(Artifact):
    kind: ArtifactKind = ArtifactKind.TEST_CANDIDATE
    test_code: str = ""
    test_file_path: str | None = None
    framework: str = ""
    covers_cases: list[str] = Field(default_factory=list)
    passed: bool | None = None
    test_output: str = ""


class CommandOutput(Artifact):
    kind: ArtifactKind = ArtifactKind.COMMAND_OUTPUT
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    success: bool = True


class LintOutput(Artifact):
    kind: ArtifactKind = ArtifactKind.LINT_OUTPUT
    tool: str
    issues: list[dict[str, Any]] = Field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    passed: bool = True
    raw_output: str = ""


class TypeCheckOutput(Artifact):
    kind: ArtifactKind = ArtifactKind.TYPE_CHECK_OUTPUT
    tool: str
    errors: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = True
    raw_output: str = ""


class ReviewComments(Artifact):
    kind: ArtifactKind = ArtifactKind.REVIEW_COMMENTS
    comments: list[dict[str, Any]] = Field(default_factory=list)
    overall_verdict: str = ""   # "approve" | "request_changes" | "needs_discussion"
    severity_counts: dict[str, int] = Field(default_factory=dict)
    key_concerns: list[str] = Field(default_factory=list)
    praise: list[str] = Field(default_factory=list)


class RiskFlags(Artifact):
    kind: ArtifactKind = ArtifactKind.RISK_FLAGS
    flags: list[dict[str, Any]] = Field(default_factory=list)
    overall_risk: str = "low"   # "low" | "medium" | "high" | "critical"
    requires_human_review: bool = False


class JudgeVerdict(Artifact):
    kind: ArtifactKind = ArtifactKind.JUDGE_VERDICT
    winner_candidate_id: str | None = None
    winner_rationale: str = ""
    candidate_scores: dict[str, float] = Field(default_factory=dict)
    evidence_used: list[str] = Field(default_factory=list)
    alternatives_rejected: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0


class FinalDecision(Artifact):
    kind: ArtifactKind = ArtifactKind.FINAL_DECISION
    summary: str = ""
    patch: CodePatch | None = None
    tests: TestCandidate | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    judge_verdict: JudgeVerdict | None = None
    alternatives: list[CodePatch] = Field(default_factory=list)
    review: ReviewComments | None = None
    risk: RiskFlags | None = None
    tokens_used: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    duration_ms: float = 0.0


class RoutingDecision(Artifact):
    kind: ArtifactKind = ArtifactKind.ROUTING_DECISION
    task_type: TaskType
    assignments: list[dict[str, Any]] = Field(default_factory=list)   # role -> model
    rationale: str = ""
    skipped_providers: list[str] = Field(default_factory=list)
    policy_constraints: list[str] = Field(default_factory=list)


# ── Task + Subtask ────────────────────────────────────────────────────────────

class SubTask(BaseModel):
    id: str
    role: RoleType
    description: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_id: str | None = None
    provider: str | None = None
    model: str | None = None
    status: str = "pending"   # pending | running | done | failed
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class OrchestratedTask(BaseModel):
    id: str
    trace_id: str
    brief: TaskBrief
    mode: str
    subtasks: list[SubTask] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    routing: RoutingDecision | None = None
    final: FinalDecision | None = None
    status: str = "pending"
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
