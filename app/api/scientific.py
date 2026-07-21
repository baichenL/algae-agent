from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.db import scientific as scientific_db
from app.core.security import ApiPrincipal, require_scientist, require_viewer
from app.services.scientific.importer import DatasetImportError, parse_scientific_dataset
from app.services.scientific.service import run_scientific_task
from app.tools.executor import ToolExecutionContext, execute_registered_tool


router = APIRouter(prefix="/scientific", tags=["scientific"])
lab_router = APIRouter(prefix="/lab", tags=["lab"])


class ScientificRunRequest(BaseModel):
    dataset_id: str = Field(min_length=1)
    mode: Literal["diagnose", "diagnose_and_optimize"] = "diagnose_and_optimize"
    target_metric: str | None = None
    direction: Literal["maximize", "minimize"] = "maximize"
    target_batch_ids: list[str] = Field(default_factory=list)
    max_cycles: int = Field(default=2, ge=1, le=2)
    offline_replay: bool = False
    session_id: str = "scientific-api"
    run_seed: int = 2025


@router.post("/datasets/import")
async def import_dataset(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(default=None),
    strain_id: str | None = Form(default=None),
    mapping_json: str | None = Form(default=None),
    sheet_name: str | None = Form(default=None),
    principal: ApiPrincipal = Depends(require_scientist),
):
    try:
        mapping = json.loads(mapping_json) if mapping_json else None
        parsed = parse_scientific_dataset(
            content=await file.read(), filename=file.filename or "dataset",
            dataset_name=dataset_name, strain_id=strain_id, mapping=mapping, sheet_name=sheet_name,
        )
        existing = scientific_db.get_dataset(parsed.dataset["id"], include_measurements=False)
        if existing:
            return {"status": "success", "idempotent": True, "dataset": existing}
        scientific_db.insert_dataset(parsed.dataset, parsed.batches)
        dataset = scientific_db.get_dataset(parsed.dataset["id"], include_measurements=False)
        return {"status": "success", "idempotent": False, "dataset": dataset, "imported_by": principal.name}
    except DatasetImportError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc), **exc.details}) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_mapping_json"}) from exc


@router.get("/datasets")
async def datasets(_: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "datasets": scientific_db.list_datasets()}


@router.get("/datasets/{dataset_id}")
async def dataset(dataset_id: str, _: ApiPrincipal = Depends(require_viewer)):
    item = scientific_db.get_dataset(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="scientific_dataset_not_found")
    return {"status": "success", "dataset": item}


@router.post("/runs", status_code=202)
async def start_scientific_run(payload: ScientificRunRequest, principal: ApiPrincipal = Depends(require_scientist)):
    try:
        result = run_scientific_task(
            dataset_id=payload.dataset_id, mode=payload.mode, target_metric=payload.target_metric,
            direction=payload.direction, target_batch_ids=payload.target_batch_ids,
            max_cycles=payload.max_cycles, session_id=payload.session_id,
            run_seed=payload.run_seed,
            offline_replay=payload.offline_replay, create_pending=payload.mode == "diagnose_and_optimize",
        )
        return {"status": "success", "scientific_run": result, "started_by": principal.name}
    except ValueError as exc:
        if str(exc) == "scientific_dataset_not_found":
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
async def scientific_run(run_id: str, _: ApiPrincipal = Depends(require_viewer)):
    run = scientific_db.get_scientific_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="scientific_run_not_found")
    return {"status": "success", "scientific_run": run}


@router.post("/runs/{run_id}/proposal", status_code=202)
async def request_proposal(run_id: str, principal: ApiPrincipal = Depends(require_scientist)):
    result = await execute_registered_tool(
        "experiment_proposal_request",
        {"scientific_run_id": run_id},
        context=ToolExecutionContext(caller="frontend_form", source="scientific_api", session_id=principal.name),
    )
    if result.get("status") in {"error", "blocked"}:
        raise HTTPException(status_code=422, detail=result.get("message") or result)
    return result


@lab_router.get("/devices")
async def devices(_: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "devices": scientific_db.list_lab_devices()}
