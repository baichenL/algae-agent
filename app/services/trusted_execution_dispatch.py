from __future__ import annotations

import os
from typing import Any

from app.core.workspaces import WorkspaceContext


def dispatch_trusted_execution(
    background_tasks: Any,
    approval_id: int,
    workflow_run_id: int | None = None,
    workspace: WorkspaceContext | None = None,
    operation_id: str | None = None,
) -> None:
    """Wake an execution path without importing worker capabilities in production.

    Production workers poll the durable execution queue. Tests and explicit local
    development may execute inline, in which case the privileged module is loaded
    lazily and only after the inline mode check.
    """
    inline = (
        os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test"
        or os.getenv("TRUSTED_WORKER_INLINE", "false").casefold()
        in {"1", "true", "yes", "on"}
    )
    if not inline:
        return

    from app.services.approval_execution_service import execute_approved_pending

    background_tasks.add_task(
        execute_approved_pending,
        approval_id,
        workflow_run_id,
        workspace,
        operation_id,
    )
