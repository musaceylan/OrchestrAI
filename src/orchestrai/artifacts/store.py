"""
Artifact store — in-process (+ optional disk) storage with provenance tracking.
All orchestration artifacts flow through here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from orchestrai.artifacts.schemas import Artifact, ArtifactKind
from orchestrai.config.settings import get_settings
from orchestrai.observability.trace import make_artifact_id

log = structlog.get_logger()


class ArtifactStore:
    """
    Thread-safe (asyncio) artifact store.
    Stores in memory for the lifetime of a task, persists to disk.
    """

    def __init__(self, task_id: str) -> None:
        self._task_id = task_id
        self._store: dict[str, dict[str, Any]] = {}
        settings = get_settings()
        self._artifact_dir = Path(settings.observability.artifact_dir) / task_id
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def put(self, artifact: Artifact) -> str:
        """Store an artifact. Returns its ID."""
        if not artifact.id:
            artifact.id = make_artifact_id()
        data = artifact.model_dump(mode="json")
        self._store[artifact.id] = data
        self._persist(artifact.id, data)
        log.info(
            "artifact.stored",
            task_id=self._task_id,
            artifact_id=artifact.id,
            kind=artifact.kind.value,
        )
        return artifact.id

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        return self._store.get(artifact_id)

    def list_by_kind(self, kind: ArtifactKind) -> list[dict[str, Any]]:
        return [v for v in self._store.values() if v.get("kind") == kind.value]

    def all(self) -> dict[str, dict[str, Any]]:
        return dict(self._store)

    def _persist(self, artifact_id: str, data: dict[str, Any]) -> None:
        path = self._artifact_dir / f"{artifact_id}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            log.warning("artifact.persist_error", artifact_id=artifact_id, error=str(e))

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for v in self._store.values():
            k = v.get("kind", "unknown")
            counts[k] = counts.get(k, 0) + 1
        return {"task_id": self._task_id, "total": len(self._store), "by_kind": counts}
