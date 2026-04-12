"""
Structured tracing and logging for every orchestration decision.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import structlog

from orchestrai.config.settings import get_settings

log = structlog.get_logger()


def _make_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def make_trace_id() -> str:
    return _make_id("tr-")


def make_task_id() -> str:
    return _make_id("task-")


def make_subtask_id() -> str:
    return _make_id("sub-")


def make_agent_run_id() -> str:
    return _make_id("run-")


def make_artifact_id() -> str:
    return _make_id("art-")


class TraceEvent(dict):
    """A single trace event — serialisable dict subclass."""


class Tracer:
    """Records a structured trace for an orchestrated task."""

    def __init__(self, trace_id: str, task_id: str) -> None:
        self.trace_id = trace_id
        self.task_id = task_id
        self.events: list[TraceEvent] = []
        self._start = time.time()
        settings = get_settings()
        self._trace_dir = Path(settings.observability.trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)

    def _emit(self, kind: str, **kwargs: Any) -> None:
        event = TraceEvent(
            trace_id=self.trace_id,
            task_id=self.task_id,
            kind=kind,
            ts=time.time(),
            elapsed_ms=round((time.time() - self._start) * 1000, 2),
            **kwargs,
        )
        self.events.append(event)
        log.info("trace", **{k: v for k, v in event.items() if k != "trace_id"})

    def task_started(self, mode: str, task_type: str) -> None:
        self._emit("task_started", mode=mode, task_type=task_type)

    def routing_decided(self, assignments: list[dict[str, Any]], rationale: str) -> None:
        self._emit("routing_decided", assignments=assignments, rationale=rationale)

    def subtask_started(self, subtask_id: str, role: str, provider: str, model: str) -> None:
        self._emit(
            "subtask_started",
            subtask_id=subtask_id,
            role=role,
            provider=provider,
            model=model,
        )

    def subtask_finished(
        self,
        subtask_id: str,
        role: str,
        success: bool,
        duration_ms: float,
        tokens: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        self._emit(
            "subtask_finished",
            subtask_id=subtask_id,
            role=role,
            success=success,
            duration_ms=duration_ms,
            tokens=tokens or {},
            error=error,
        )

    def tool_executed(
        self, tool: str, command: str, exit_code: int, duration_ms: float
    ) -> None:
        self._emit(
            "tool_executed",
            tool=tool,
            command=command,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    def artifact_created(self, artifact_id: str, kind: str, agent_run_id: str | None) -> None:
        # Use `artifact_kind` in kwargs to avoid shadowing _emit's positional `kind` param.
        self._emit(
            "artifact_created",
            artifact_id=artifact_id,
            artifact_kind=kind,
            agent_run_id=agent_run_id,
        )

    def judge_ran(self, winner: str | None, confidence: float, candidates: int) -> None:
        self._emit(
            "judge_ran",
            winner=winner,
            confidence=confidence,
            candidates=candidates,
        )

    def task_finished(
        self,
        success: bool,
        total_tokens: dict[str, int],
        cost_usd: float | None,
        error: str | None = None,
    ) -> None:
        self._emit(
            "task_finished",
            success=success,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            total_duration_ms=round((time.time() - self._start) * 1000, 2),
            error=error,
        )

    def save(self) -> Path:
        # Validate task_id to prevent path traversal
        if not self.task_id or "/" in self.task_id or "\\" in self.task_id or ".." in self.task_id:
            raise ValueError(f"Invalid task_id for trace save: {self.task_id!r}")
        path = self._trace_dir / f"{self.task_id}.jsonl"
        resolved = path.resolve()
        if not str(resolved).startswith(str(self._trace_dir.resolve())):
            raise ValueError(f"Path traversal detected for task_id: {self.task_id!r}")
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "events": list(self.events),
            "duration_ms": round((time.time() - self._start) * 1000, 2),
        }


@contextmanager
def task_trace(task_id: str) -> Generator[Tracer, None, None]:
    tracer = Tracer(trace_id=make_trace_id(), task_id=task_id)
    try:
        yield tracer
    finally:
        tracer.save()


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    import logging
    import sys

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
