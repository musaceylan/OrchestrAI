"""
Optional Prometheus metrics — gracefully degrades to structured-log no-ops
if ``prometheus_client`` is not installed.

Install extras to enable:
    pip install orchestrai[metrics]
"""
from __future__ import annotations

from typing import Any

try:
    from prometheus_client import Counter, Histogram
    from prometheus_client import registry as _prom_registry

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False


# ── No-op shims ───────────────────────────────────────────────────────────────

class _NoOpCounter:
    def labels(self, **_: Any) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1) -> None:
        pass


class _NoOpHistogram:
    def labels(self, **_: Any) -> "_NoOpHistogram":
        return self

    def observe(self, amount: float) -> None:
        pass


# ── Factory helpers ───────────────────────────────────────────────────────────

def _counter(name: str, doc: str, labelnames: list[str]) -> Any:
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpCounter()
    try:
        return Counter(name, doc, labelnames)
    except ValueError:
        # Already registered (e.g. during test re-imports) — fetch existing
        return _prom_registry.REGISTRY._names_to_collectors.get(
            name + "_total", _NoOpCounter()
        )


def _histogram(
    name: str,
    doc: str,
    labelnames: list[str],
    buckets: list[float] | None = None,
) -> Any:
    if not _PROMETHEUS_AVAILABLE:
        return _NoOpHistogram()
    kwargs: dict[str, Any] = {}
    if buckets:
        kwargs["buckets"] = buckets
    try:
        return Histogram(name, doc, labelnames, **kwargs)
    except ValueError:
        return _prom_registry.REGISTRY._names_to_collectors.get(name, _NoOpHistogram())


# ── Metric definitions ────────────────────────────────────────────────────────

#: Increment on each task completion — labels: status ("done"|"failed"), mode
tasks_total = _counter(
    "orchestrai_tasks_total",
    "Total orchestrated tasks by status and mode",
    ["status", "mode"],
)

#: Observe task wall-clock duration (seconds) — labels: mode
task_duration_seconds = _histogram(
    "orchestrai_task_duration_seconds",
    "Task end-to-end duration in seconds",
    ["mode"],
    buckets=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0],
)

#: Increment on each LLM call attempt (after retries resolve) — labels: role, provider, status
agent_calls_total = _counter(
    "orchestrai_agent_calls_total",
    "Total agent (LLM) calls by role, provider, and outcome",
    ["role", "provider", "status"],
)

#: Increment by estimated USD cost — labels: provider, model
cost_usd_total = _counter(
    "orchestrai_cost_usd_total",
    "Cumulative estimated cost in USD by provider and model",
    ["provider", "model"],
)

#: Increment on each safety violation caught in a generated diff
safety_violations_total = _counter(
    "orchestrai_safety_violations_total",
    "Total safety violations detected in generated diffs",
    ["kind"],
)

#: Increment on each audit tool call — labels: tool, outcome
tool_calls_total = _counter(
    "orchestrai_tool_calls_total",
    "Total MCP tool calls by tool name and outcome",
    ["tool", "outcome"],
)
