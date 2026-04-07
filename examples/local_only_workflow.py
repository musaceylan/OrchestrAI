"""
Example: Privacy-preserving local-only workflow using Ollama.
No data leaves your machine.

Requires Ollama running: ollama serve && ollama pull codellama

Run with:
    python examples/local_only_workflow.py
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
    # Only local providers (Ollama default)
    providers = await discover_providers()
    local_providers = [p for p in providers if getattr(p, "base_url", "").startswith("http://localhost")]

    if not local_providers:
        print("No local providers found. Start Ollama: ollama serve && ollama pull codellama")
        return

    registry = await CapabilityRegistry.build(local_providers)
    orchestrator = Orchestrator(registry)

    task = await orchestrator.submit(
        request="Write a Python function to parse a JWT token without verifying the signature",
        user_preferences={"privacy_level": "secret"},
    )

    print(f"Task {task.id}: {task.status}")
    if task.final and task.final.patch:
        print(f"Model: {task.final.patch.provenance.model}")
        print(task.final.patch.unified_diff or task.final.patch.description)


if __name__ == "__main__":
    asyncio.run(main())
