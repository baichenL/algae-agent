from __future__ import annotations

from typing import Any

from langgraph.types import Command

from app.core import database
from app.services.agent_runtime.checkpoint import get_compiled_graph
from app.services.agent_runtime.events import record_run_event


async def process_resume_job(job_id: int) -> dict[str, Any]:
    job = database.claim_resume_job(job_id)
    if not job:
        return {"status": "skipped", "reason": "job_not_claimable", "job_id": job_id}
    pending_id = int(job["pending_id"])
    agent_run_id = job.get("agent_run_id")
    graph_thread_id = job.get("graph_thread_id")
    try:
        app = get_compiled_graph()
        if app is None:
            raise RuntimeError("agent_runtime_graph_not_initialized")
        record_run_event(
            agent_run_id,
            event_type="approval_resume_requested",
            layer="agent_runtime",
            payload={
                "pending_id": pending_id,
                "graph_thread_id": graph_thread_id,
                "approval_version": job.get("approval_version"),
                "job_id": job_id,
                "attempt": job.get("attempt_count"),
            },
        )
        result = await app.ainvoke(
            Command(
                resume={
                    "pending_id": pending_id,
                    "approval_version": job.get("approval_version"),
                    "job_id": job_id,
                }
            ),
            config={"configurable": {"thread_id": graph_thread_id}},
        )
        database.mark_resume_job_succeeded(job_id)
        database.mark_pending_resume_completed(pending_id, {"graph_result": "resumed"})
        record_run_event(
            agent_run_id,
            event_type="approval_resume_job_succeeded",
            layer="agent_runtime",
            payload={"pending_id": pending_id, "graph_thread_id": graph_thread_id, "job_id": job_id},
        )
        return {"status": "success", "job_id": job_id, "pending_id": pending_id, "graph_result": result}
    except Exception as exc:
        database.mark_resume_job_failed(job_id, str(exc))
        database.mark_pending_resume_failed(pending_id, str(exc))
        record_run_event(
            agent_run_id,
            event_type="approval_resume_job_failed",
            layer="agent_runtime",
            payload={
                "pending_id": pending_id,
                "graph_thread_id": graph_thread_id,
                "job_id": job_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return {"status": "error", "job_id": job_id, "pending_id": pending_id, "reason": str(exc)}


async def process_pending_resume_jobs(*, limit: int = 20) -> list[dict[str, Any]]:
    results = []
    for job in database.list_pending_resume_jobs(limit=limit):
        results.append(await process_resume_job(int(job["id"])))
    return results


async def review_pending_and_resume(
    pending_id: int,
    *,
    approve: bool,
    reviewed_by: str = "human",
    review_reason: str | None = None,
) -> dict[str, Any]:
    review = database.review_pending_action_with_resume_job(
        pending_id,
        approve=approve,
        reviewed_by=reviewed_by,
        review_reason=review_reason,
    )
    if review.get("status") != "success":
        return review
    if not review.get("graph_thread_id"):
        return {**review, "durable_resume": False}
    job_id = review.get("job_id")
    if not job_id:
        return {**review, "durable_resume": True, "resume": {"status": "skipped", "reason": "missing_job"}}
    resume = await process_resume_job(int(job_id))
    return {**review, "durable_resume": True, "resume": resume}
