# app/api/workflow.py
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.database import get_algae_status
from app.core.security import ApiPrincipal, require_approver, require_scientist, require_viewer
from app.schemas.algae import (
    ManualSubcultureRequest,
    StatusQueryRequest,
    SubcultureRequest,
)
from app.services.strains import strain_service

router = APIRouter()


class SimulationRunRequest(BaseModel):
    strain_id: str = Field("Chlorella_01", min_length=1, max_length=100)
    generation_number: int = Field(1, ge=1, le=100000)
    interaction_mode: Literal["verification", "demo"] = "demo"
    fault_scenario: Optional[str] = None
    wavelength_nm: float = Field(680.0, ge=190.0, le=1100.0)
    simulated_absorbance: float = Field(0.8, ge=0.0, le=4.0)
    # Backward-compatible field used by the previous simulation UI.
    fault: Optional[str] = None


class ManualTaskResolutionRequest(BaseModel):
    resolution: Literal["completed", "failed"]
    note: Optional[str] = Field(None, max_length=500)


@router.post("/workflow/simulations", status_code=status.HTTP_202_ACCEPTED)
async def start_subculture_simulation(payload: SimulationRunRequest, _: ApiPrincipal = Depends(require_scientist)):
    """Start a visual-only run that never mutates the production strain DB."""
    from app.services.workflows.simulation_service import simulation_runs

    try:
        run = simulation_runs.create(
            strain_id=payload.strain_id,
            generation_number=payload.generation_number,
            interaction_mode=payload.interaction_mode,
            fault_scenario=payload.fault_scenario or payload.fault,
            wavelength_nm=payload.wavelength_nm,
            simulated_absorbance=payload.simulated_absorbance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return run


@router.post("/workflow/simulations/{run_id}/manual-tasks/{task_id}/resolve")
async def resolve_subculture_manual_task(
    run_id: str,
    task_id: str,
    payload: ManualTaskResolutionRequest,
    _: ApiPrincipal = Depends(require_approver),
):
    from app.services.workflows.simulation_service import (
        ManualTaskConflict,
        SimulationRunNotFound,
        simulation_runs,
    )

    try:
        return simulation_runs.resolve_manual_task(
            run_id,
            task_id,
            resolution=payload.resolution,
            note=payload.note,
        )
    except SimulationRunNotFound as exc:
        raise HTTPException(status_code=404, detail="Simulation run not found") from exc
    except ManualTaskConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/workflow/simulations/{run_id}")
async def get_subculture_simulation(run_id: str, _: ApiPrincipal = Depends(require_viewer)):
    from app.services.workflows.simulation_service import simulation_runs

    run = simulation_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Simulation run not found")
    return run


@router.post("/workflow/subculture", status_code=status.HTTP_202_ACCEPTED)
async def run_subculture_workflow(payload: SubcultureRequest, _: ApiPrincipal = Depends(require_scientist)):
    """Compatibility alias for a visual-only simulation run."""
    from app.services.workflows.simulation_service import simulation_runs

    try:
        run = simulation_runs.create(
            strain_id=payload.strain_id,
            generation_number=payload.generation_number,
        )
        return {
            **run,
            "execution_mode": "simulation",
            "physical_execution": False,
            "persisted": False,
            "deprecated_endpoint": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"传代模拟任务创建失败: {str(e)}"
        )

@router.post("/workflow/force_subculture")
async def force_subculture_endpoint(
    payload: Optional[ManualSubcultureRequest] = None,
    _: ApiPrincipal = Depends(require_approver),
):
    strain = "Chlorella_01"
    if payload and payload.strain_id:
        strain = payload.strain_id

    status_data = get_algae_status(strain)
    if not status_data:
        raise HTTPException(status_code=404, detail=f"未找到品系 {strain} 的生存记录")
        
    try:
        pending_id = strain_service.create_pending_workflow_subculture(
            strain,
            requester="human_api",
            source_message="POST /workflow/force_subculture",
        )
        return {
            "status": "pending",
            "action": "workflow_subculture",
            "pending_id": pending_id,
            "strain_id": strain,
            "require_confirmation": True,
            "confirm_endpoint": "/api/v1/strain/confirm",
            "msg": "已创建高风险传代待审批请求；当前没有执行硬件或修改正式状态。",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"传代审批请求创建失败: {str(e)}")

@router.post("/status/query")
async def query_subculture_status(payload: StatusQueryRequest, _: ApiPrincipal = Depends(require_viewer)):
    status_data = get_algae_status(payload.strain_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="未找到该藻种的生存记录")
    
    return {
        "status": "success",
        "strain_id": payload.strain_id,
        "current_generation": status_data.get("generation_number", 1),
        "days_since_last_subculture": status_data.get("days_since_last_subculture", 0),
        "last_inoculation_time": status_data.get("last_subculture_time", ""),
        "report": f"当前是第 {status_data.get('generation_number', 1)} 代，距离上次传代 {status_data.get('days_since_last_subculture', 0)} 天，最后操作时间：{status_data.get('last_subculture_time', '未知')}"
    }
