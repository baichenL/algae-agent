from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.security import ApiPrincipal
from app.core.session_security import require_session_user_mutation, require_session_viewer
from app.core.workspaces import current_workspace
from app.services.user_memory.catalog import PREDICATES
from app.services.user_memory.store import (
    MemoryRevisionConflict,
    apply_memory,
    archive_memory,
    decide_candidate,
    delete_memory,
    get_settings,
    list_candidates,
    list_memories,
    restore_memory,
    update_memory,
    update_settings,
)


router = APIRouter(prefix="/api/v2", tags=["user-memory"])


class MemoryCreateRequest(BaseModel):
    memory_type: Literal["preference", "profile", "note"]
    predicate: str
    value: Any


class MemoryUpdateRequest(BaseModel):
    value: Any
    expected_revision: int = Field(ge=1)


class MemorySettingsUpdateRequest(BaseModel):
    enabled: bool
    auto_write_low_risk: bool = True


def _identity(principal: ApiPrincipal) -> tuple[str, str]:
    return principal.name, current_workspace().id


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Memory not found.")


@router.get("/memories")
async def memories(
    status: Literal["active", "archived", "superseded"] | None = None,
    memory_type: Literal["preference", "profile", "note"] | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    after_id: int | None = Query(default=None, ge=1),
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    owner_id, workspace_id = _identity(principal)
    items = list_memories(
        owner_id=owner_id, workspace_id=workspace_id, status=status,
        memory_type=memory_type, limit=limit, after_id=after_id,
    )
    return {"memories": items, "next_after_id": items[-1]["id"] if len(items) == limit else None}


@router.post("/memories")
async def create_memory(
    payload: MemoryCreateRequest,
    principal: ApiPrincipal = Depends(require_session_user_mutation),
):
    owner_id, workspace_id = _identity(principal)
    try:
        action, memory = apply_memory(
            owner_id=owner_id, workspace_id=workspace_id,
            memory_type=payload.memory_type, predicate=payload.predicate, value=payload.value,
            actor=principal.name, reason="user_created", confirmation=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"action": action, "memory": memory}


@router.patch("/memories/{memory_id}")
async def patch_memory(
    memory_id: int,
    payload: MemoryUpdateRequest,
    principal: ApiPrincipal = Depends(require_session_user_mutation),
):
    owner_id, workspace_id = _identity(principal)
    try:
        memory = update_memory(
            memory_id, owner_id=owner_id, workspace_id=workspace_id,
            expected_revision=payload.expected_revision, value=payload.value, actor=principal.name,
        )
    except KeyError as exc:
        raise _not_found() from exc
    except MemoryRevisionConflict as exc:
        raise HTTPException(status_code=409, detail="Memory revision conflict; refresh before editing.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"memory": memory}


@router.post("/memories/{memory_id}/archive")
async def archive(memory_id: int, principal: ApiPrincipal = Depends(require_session_user_mutation)):
    owner_id, workspace_id = _identity(principal)
    try:
        return {"memory": archive_memory(memory_id, owner_id=owner_id, workspace_id=workspace_id, actor=principal.name)}
    except KeyError as exc:
        raise _not_found() from exc


@router.post("/memories/{memory_id}/restore")
async def restore(memory_id: int, principal: ApiPrincipal = Depends(require_session_user_mutation)):
    owner_id, workspace_id = _identity(principal)
    try:
        return {"memory": restore_memory(memory_id, owner_id=owner_id, workspace_id=workspace_id, actor=principal.name)}
    except KeyError as exc:
        raise _not_found() from exc


@router.delete("/memories/{memory_id}")
async def remove(memory_id: int, principal: ApiPrincipal = Depends(require_session_user_mutation)):
    owner_id, workspace_id = _identity(principal)
    try:
        delete_memory(memory_id, owner_id=owner_id, workspace_id=workspace_id, actor=principal.name)
    except KeyError as exc:
        raise _not_found() from exc
    return {"status": "deleted", "id": memory_id}


@router.get("/memories/candidates")
async def candidates(
    status: Literal["pending", "accepted", "rejected", "blocked"] = "pending",
    limit: int = Query(default=100, ge=1, le=200),
    principal: ApiPrincipal = Depends(require_session_viewer),
):
    owner_id, workspace_id = _identity(principal)
    return {"candidates": list_candidates(owner_id=owner_id, workspace_id=workspace_id, status=status, limit=limit)}


@router.post("/memories/candidates/{candidate_id}/accept")
async def accept_candidate(candidate_id: int, principal: ApiPrincipal = Depends(require_session_user_mutation)):
    owner_id, workspace_id = _identity(principal)
    try:
        return {"candidate": decide_candidate(candidate_id, owner_id=owner_id, workspace_id=workspace_id, accept=True, actor=principal.name)}
    except KeyError as exc:
        raise _not_found() from exc


@router.post("/memories/candidates/{candidate_id}/reject")
async def reject_candidate(candidate_id: int, principal: ApiPrincipal = Depends(require_session_user_mutation)):
    owner_id, workspace_id = _identity(principal)
    try:
        return {"candidate": decide_candidate(candidate_id, owner_id=owner_id, workspace_id=workspace_id, accept=False, actor=principal.name)}
    except KeyError as exc:
        raise _not_found() from exc


@router.get("/memories/export")
async def export_memories(principal: ApiPrincipal = Depends(require_session_viewer)):
    owner_id, workspace_id = _identity(principal)
    return {
        "schema_version": 1,
        "owner_id": owner_id,
        "workspace_id": workspace_id,
        "settings": get_settings(owner_id, workspace_id),
        "memories": list_memories(owner_id=owner_id, workspace_id=workspace_id, limit=200),
    }


@router.get("/memory-settings")
async def memory_settings(principal: ApiPrincipal = Depends(require_session_viewer)):
    owner_id, workspace_id = _identity(principal)
    return {"settings": get_settings(owner_id, workspace_id), "predicates": sorted(PREDICATES)}


@router.patch("/memory-settings")
async def patch_memory_settings(
    payload: MemorySettingsUpdateRequest,
    principal: ApiPrincipal = Depends(require_session_user_mutation),
):
    owner_id, workspace_id = _identity(principal)
    return {"settings": update_settings(
        owner_id, workspace_id, enabled=payload.enabled,
        auto_write_low_risk=payload.auto_write_low_risk,
    )}
