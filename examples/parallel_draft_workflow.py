"""
Example: Run parallel drafts from multiple models and get a judge verdict.

Run with:
    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... python examples/parallel_draft_workflow.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.providers.discovery import discover_providers
from orchestrai.registry.registry import CapabilityRegistry


async def main() -> None:
    providers = await discover_providers()
    if not providers:
        print("No providers available.")
        return

    registry = await CapabilityRegistry.build(providers)
    orchestrator = Orchestrator(registry)

    task = await orchestrator.submit(
        request=(
            "Implement a Python function `levenshtein_distance(a: str, b: str) -> int` "
            "that computes the edit distance between two strings efficiently."
        ),
        mode="parallel_draft",
    )

    print(f"Task {task.id}: {task.status}")
    if task.final:
        print(f"Confidence: {task.final.confidence:.2f}")
        if task.final.patch:
            print(f"Winner: {task.final.patch.provenance.provider}/{task.final.patch.provenance.model}")
            print("\n--- WINNING IMPLEMENTATION ---")
            print(task.final.patch.unified_diff or task.final.patch.description)


if __name__ == "__main__":
    asyncio.run(main())
