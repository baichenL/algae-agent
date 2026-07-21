from __future__ import annotations

from app.core import database
from app.core.db import workflow_runs


def inspect_workflow_activity(
    *,
    session_id: str,
    strain_id: str | None = None,
    pending_id: int | None = None,
) -> dict:
    if pending_id is not None:
        pending = database.get_pending_action(pending_id)
        pendings = [pending] if pending else []
    else:
        pendings = []
        for item in database.list_pending_actions(status="all"):
            payload = item.get("payload") or {}
            data = payload.get("data") or {}
            if payload.get("type") != "workflow_subculture":
                continue
            if strain_id and data.get("strain_id") != strain_id:
                continue
            if not strain_id and data.get("session_id") != session_id:
                continue
            pendings.append(item)
    runs = []
    for pending in pendings:
        run = workflow_runs.get_workflow_run_by_pending(pending.get("id"))
        if run:
            runs.append(run)
    return {
        "pending": pendings,
        "workflow_runs": runs,
        "pending_count": len(pendings),
        "run_count": len(runs),
        "strain_id": strain_id,
        "pending_id": pending_id,
    }
