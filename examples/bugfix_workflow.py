"""
Example: Orchestrate a bugfix using OrchestrAI directly (without MCP).

Run with:
    ANTHROPIC_API_KEY=... python examples/bugfix_workflow.py
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
    print("Discovering providers...")
    providers = await discover_providers()
    if not providers:
        print("No providers available. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or start Ollama.")
        return

    registry = await CapabilityRegistry.build(providers)
    print(f"Registry: {len(registry.all_capabilities())} models across {len(providers)} providers")

    orchestrator = Orchestrator(registry)
    task = await orchestrator.submit(
        request=(
            "Fix the bug where the login function raises AttributeError "
            "when the user dict is missing the 'email' key. "
            "It should return a 401 error instead of crashing."
        ),
        mode="planner_coder_reviewer",
    )

    print(f"\nTask {task.id}: {task.status}")
    if task.final:
        print(f"Summary: {task.final.summary}")
        print(f"Confidence: {task.final.confidence:.2f}")
        if task.final.patch and task.final.patch.unified_diff:
            print("\n--- PATCH ---")
            print(task.final.patch.unified_diff[:2000])
        if task.final.review:
            print(f"\nReview verdict: {task.final.review.overall_verdict}")
            for concern in task.final.review.key_concerns:
                print(f"  ⚠ {concern}")


if __name__ == "__main__":
    asyncio.run(main())
