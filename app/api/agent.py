from fastapi import APIRouter, HTTPException

from app.services.observability.agent_trace import (
    build_agent_run_trace,
    list_agent_run_summaries,
)
from app.services.learning.store import (
    list_agent_learning_reviews,
    list_agent_memories,
    list_agent_skills,
)

router = APIRouter(prefix="/agent")


@router.get("/runs")
async def list_agent_runs(
    session_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
):
    return {
        "status": "success",
        "runs": list_agent_run_summaries(
            session_id=session_id,
            status=status,
            limit=limit,
        ),
    }


@router.get("/runs/{agent_run_id}")
async def get_agent_run_trace(agent_run_id: str):
    result = build_agent_run_trace(agent_run_id)
    if not result:
        raise HTTPException(status_code=404, detail="agent_run_not_found")
    return result


@router.get("/memories")
async def list_memories(
    scope: str | None = None,
    status: str | None = None,
    limit: int = 50,
):
    return {
        "status": "success",
        "memories": list_agent_memories(scope=scope, status=status, limit=limit),
    }


@router.get("/skills")
async def list_skills(
    status: str | None = None,
    limit: int = 50,
):
    return {
        "status": "success",
        "skills": list_agent_skills(status=status, limit=limit),
    }


@router.get("/learning/reviews")
async def list_learning_reviews(
    status: str | None = None,
    agent_run_id: str | None = None,
    limit: int = 50,
):
    return {
        "status": "success",
        "reviews": list_agent_learning_reviews(
            status=status,
            agent_run_id=agent_run_id,
            limit=limit,
        ),
    }
