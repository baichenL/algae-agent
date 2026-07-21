from __future__ import annotations

import os
import re
import shutil
import gc
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterator
from uuid import uuid4


DEFAULT_DB_PATH = "data/outputs/algae_system_local.db"
TEST_WORKSPACE_ROOT = Path("data/outputs/test_workspaces")


@dataclass(frozen=True)
class WorkspaceContext:
    id: str
    name: str
    db_path: str
    simulation_only: bool = True
    seed: int = 2025
    time_origin: str = "2025-01-01T00:00:00Z"
    email_mode: str = "test_outbox"


SHARED_WORKSPACE = WorkspaceContext(
    id="shared",
    name="共享实验室",
    db_path=DEFAULT_DB_PATH,
    email_mode="smtp",
)

_current_workspace: ContextVar[WorkspaceContext] = ContextVar("algae_workspace", default=SHARED_WORKSPACE)
_workspaces: dict[str, WorkspaceContext] = {SHARED_WORKSPACE.id: SHARED_WORKSPACE}
_run_workspaces: dict[str, str] = {}
_lock = RLock()


class WorkspaceDatabasePath(os.PathLike[str]):
    """Path-like proxy retained by legacy DB modules while resolving per request."""

    def __fspath__(self) -> str:
        return current_workspace().db_path

    def __str__(self) -> str:
        return self.__fspath__()


DATABASE_PATH = WorkspaceDatabasePath()


def current_workspace() -> WorkspaceContext:
    return _current_workspace.get()


@contextmanager
def workspace_scope(workspace: WorkspaceContext) -> Iterator[WorkspaceContext]:
    token = _current_workspace.set(workspace)
    try:
        yield workspace
    finally:
        _current_workspace.reset(token)


def create_test_workspace(scenario_id: str, *, seed: int = 2025) -> WorkspaceContext:
    safe_scenario = re.sub(r"[^a-z0-9-]+", "-", scenario_id.casefold()).strip("-") or "scenario"
    workspace_id = f"test-{safe_scenario}-{uuid4().hex[:10]}"
    root = TEST_WORKSPACE_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_path = (root / workspace_id / "workspace.sqlite3").resolve()
    if root not in db_path.parents:
        raise RuntimeError("invalid_workspace_path")
    db_path.parent.mkdir(parents=True, exist_ok=False)
    workspace = WorkspaceContext(
        id=workspace_id,
        name=f"隔离测试 · {safe_scenario}",
        db_path=str(db_path),
        seed=int(seed),
    )
    with _lock:
        _workspaces[workspace.id] = workspace
    return workspace


def register_run_workspace(run_id: str, workspace: WorkspaceContext) -> None:
    with _lock:
        _run_workspaces[run_id] = workspace.id


def workspace_for_run(run_id: str) -> WorkspaceContext | None:
    with _lock:
        workspace_id = _run_workspaces.get(run_id)
        return _workspaces.get(workspace_id) if workspace_id else None


def get_workspace(workspace_id: str) -> WorkspaceContext | None:
    with _lock:
        return _workspaces.get(workspace_id)


def list_test_workspaces() -> list[WorkspaceContext]:
    with _lock:
        return [item for item in _workspaces.values() if item.id != SHARED_WORKSPACE.id]


def reset_test_workspace(workspace_id: str) -> WorkspaceContext:
    workspace = get_workspace(workspace_id)
    if not workspace or workspace.id == SHARED_WORKSPACE.id:
        raise KeyError(workspace_id)
    path = Path(workspace.db_path).resolve()
    root = TEST_WORKSPACE_ROOT.resolve()
    if root not in path.parents:
        raise RuntimeError("invalid_workspace_path")
    if path.exists():
        last_error: PermissionError | None = None
        for _ in range(3):
            gc.collect()
            try:
                path.unlink()
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error:
            raise last_error
    with _lock:
        for run_id, owner in list(_run_workspaces.items()):
            if owner == workspace_id:
                _run_workspaces.pop(run_id, None)
    return workspace


def delete_test_workspace(workspace_id: str) -> None:
    workspace = reset_test_workspace(workspace_id)
    directory = Path(workspace.db_path).resolve().parent
    root = TEST_WORKSPACE_ROOT.resolve()
    if root not in directory.parents:
        raise RuntimeError("invalid_workspace_path")
    if directory.exists():
        shutil.rmtree(directory)
    with _lock:
        _workspaces.pop(workspace_id, None)
