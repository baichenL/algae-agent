from __future__ import annotations

import os
import json
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
DEFAULT_CONTROL_DB_PATH = os.getenv(
    "ALGAE_CONTROL_DB_PATH",
    "data/control/algae_control.db",
)
DEFAULT_DOMAIN_DB_PATH = os.getenv(
    "ALGAE_DOMAIN_DB_PATH",
    "data/domain/algae_domain.db",
)
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
    storage_mode: str = "legacy"
    control_db_path: str | None = None
    domain_db_path: str | None = None

    def control_path(self) -> str:
        return (
            str(self.control_db_path)
            if self.storage_mode == "split" and self.control_db_path
            else self.db_path
        )

    def domain_path(self) -> str:
        return (
            str(self.domain_db_path)
            if self.storage_mode == "split" and self.domain_db_path
            else self.db_path
        )


def _shared_storage_settings() -> tuple[str, str, str]:
    explicit_mode = os.getenv("WORKSPACE_STORAGE_MODE")
    control_path = DEFAULT_CONTROL_DB_PATH
    domain_path = DEFAULT_DOMAIN_DB_PATH
    if explicit_mode is not None:
        mode = "split" if explicit_mode.strip().lower() == "split" else "legacy"
        return mode, control_path, domain_path
    manifest_path = Path(
        os.getenv(
            "WORKSPACE_STORAGE_MANIFEST",
            "data/workspace_storage_manifest.json",
        )
    )
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("storage_mode") == "split":
                return (
                    "split",
                    str(manifest["control_db_path"]),
                    str(manifest["domain_db_path"]),
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return "legacy", control_path, domain_path


_SHARED_STORAGE_MODE, _SHARED_CONTROL_PATH, _SHARED_DOMAIN_PATH = (
    _shared_storage_settings()
)


SHARED_WORKSPACE = WorkspaceContext(
    id="shared",
    name="共享实验室",
    db_path=DEFAULT_DB_PATH,
    email_mode="smtp",
    storage_mode=_SHARED_STORAGE_MODE,
    control_db_path=_SHARED_CONTROL_PATH,
    domain_db_path=_SHARED_DOMAIN_PATH,
)

_current_workspace: ContextVar[WorkspaceContext] = ContextVar("algae_workspace", default=SHARED_WORKSPACE)
_storage_target: ContextVar[str] = ContextVar("algae_storage_target", default="auto")
_workspaces: dict[str, WorkspaceContext] = {SHARED_WORKSPACE.id: SHARED_WORKSPACE}
_run_workspaces: dict[str, str] = {}
_lock = RLock()


class WorkspaceDatabasePath(os.PathLike[str]):
    """Path-like proxy retained by legacy DB modules while resolving per request."""

    def __fspath__(self) -> str:
        workspace = current_workspace()
        target = _storage_target.get()
        if target == "domain":
            return workspace.domain_path()
        return workspace.control_path()

    def __str__(self) -> str:
        return self.__fspath__()


DATABASE_PATH = WorkspaceDatabasePath()


class WorkspaceDomainDatabasePath(os.PathLike[str]):
    def __fspath__(self) -> str:
        workspace = current_workspace()
        target = _storage_target.get()
        if target == "control":
            return workspace.control_path()
        return workspace.domain_path()

    def __str__(self) -> str:
        return self.__fspath__()


DOMAIN_DATABASE_PATH = WorkspaceDomainDatabasePath()
CONTROL_DATABASE_PATH = DATABASE_PATH


def current_workspace() -> WorkspaceContext:
    return _current_workspace.get()


@contextmanager
def workspace_scope(workspace: WorkspaceContext) -> Iterator[WorkspaceContext]:
    token = _current_workspace.set(workspace)
    try:
        yield workspace
    finally:
        _current_workspace.reset(token)


@contextmanager
def storage_target_scope(target: str) -> Iterator[str]:
    if target not in {"auto", "control", "domain"}:
        raise ValueError("invalid_storage_target")
    token = _storage_target.set(target)
    try:
        yield target
    finally:
        _storage_target.reset(token)


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
        storage_mode="legacy",
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
    paths = {
        Path(workspace.db_path).resolve(),
        Path(workspace.control_path()).resolve(),
        Path(workspace.domain_path()).resolve(),
    }
    root = TEST_WORKSPACE_ROOT.resolve()
    for path in paths:
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
