"""Unit tests for artifact store."""
from __future__ import annotations

import pytest

from orchestrai.artifacts.schemas import (
    ArtifactKind,
    CodePatch,
    Provenance,
    RoleType,
)
from orchestrai.artifacts.store import ArtifactStore
from orchestrai.observability.trace import make_artifact_id, make_task_id


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRAI__OBSERVABILITY__ARTIFACT_DIR", str(tmp_path))
    task_id = make_task_id()
    return ArtifactStore(task_id), task_id


def _patch(task_id: str) -> CodePatch:
    return CodePatch(
        id=make_artifact_id(),
        kind=ArtifactKind.CODE_PATCH,
        provenance=Provenance(
            task_id=task_id,
            provider="anthropic",
            model="claude-sonnet-4-6",
            role=RoleType.CODER,
        ),
        unified_diff="--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        description="Fixed the bug",
    )


def test_put_and_get(store):
    s, task_id = store
    p = _patch(task_id)
    art_id = s.put(p)
    assert art_id == p.id

    retrieved = s.get(art_id)
    assert retrieved is not None
    assert retrieved["id"] == p.id
    assert retrieved["kind"] == ArtifactKind.CODE_PATCH.value


def test_list_by_kind(store):
    s, task_id = store
    for _ in range(3):
        s.put(_patch(task_id))
    patches = s.list_by_kind(ArtifactKind.CODE_PATCH)
    assert len(patches) == 3
    assert all(isinstance(p, dict) for p in patches)


def test_all_returns_dict(store):
    s, task_id = store
    s.put(_patch(task_id))
    s.put(_patch(task_id))
    all_arts = s.all()
    assert len(all_arts) == 2
    assert all(isinstance(v, dict) for v in all_arts.values())


def test_summary(store):
    s, task_id = store
    s.put(_patch(task_id))
    summary = s.summary()
    assert summary["total"] == 1
    assert ArtifactKind.CODE_PATCH.value in summary["by_kind"]
