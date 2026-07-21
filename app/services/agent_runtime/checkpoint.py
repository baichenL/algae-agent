from __future__ import annotations

import os
from typing import Any, Callable

import aiosqlite
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.db.connection import connect
from app.core.time_utils import time_days_ago


CHECKPOINT_DB_PATH = os.path.join("data", "outputs", "agent_checkpoints.sqlite")

_compiled_graph: Any | None = None
_checkpointer: Any | None = None
_checkpoint_connection: aiosqlite.Connection | None = None
_deps_factory: Callable[[], Any] | None = None


def graph_thread_id_for_run(agent_run_id: str) -> str:
    return f"agent-run:{agent_run_id}"


def graph_thread_id_for_task(task_id: str) -> str:
    return f"agent-task:{task_id}"


def set_runtime_deps_factory(factory: Callable[[], Any]) -> None:
    global _deps_factory
    _deps_factory = factory


def get_runtime_deps() -> Any:
    if _deps_factory is None:
        raise RuntimeError("agent_runtime_deps_factory_not_configured")
    return _deps_factory()


def set_compiled_graph(graph: Any) -> None:
    global _compiled_graph
    _compiled_graph = graph


def get_compiled_graph() -> Any | None:
    return _compiled_graph


def get_checkpointer() -> Any | None:
    return _checkpointer


async def initialize_agent_runtime_graph(builder: Callable[..., Any]) -> Any:
    global _compiled_graph, _checkpointer, _checkpoint_connection
    os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
    _checkpoint_connection = await aiosqlite.connect(CHECKPOINT_DB_PATH)
    _checkpointer = AsyncSqliteSaver(_checkpoint_connection)
    await _checkpointer.setup()
    _compiled_graph = builder(checkpointer=_checkpointer)
    return _compiled_graph


async def shutdown_agent_runtime_graph() -> None:
    global _compiled_graph, _checkpointer, _checkpoint_connection
    _compiled_graph = None
    _checkpointer = None
    if _checkpoint_connection is not None:
        await _checkpoint_connection.close()
    _checkpoint_connection = None


def build_fallback_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def _completed_checkpoint_threads(*, retention_days: int, max_completed_threads: int) -> list[str]:
    cutoff = time_days_ago(retention_days)
    with connect(row_factory=True) as conn:
        rows = conn.execute(
            """
            SELECT ar.graph_thread_id,
                   MAX(COALESCE(ar.finished_at, ar.started_at, '')) AS finished_at
            FROM agent_runs AS ar
            LEFT JOIN conversation_tasks AS task ON task.id = ar.task_id
            WHERE ar.graph_thread_id IS NOT NULL
              AND ar.status NOT IN ('running', 'waiting_approval')
              AND (
                    ar.task_id IS NULL
                    OR task.status IN ('completed', 'cancelled', 'failed', 'expired')
                  )
              AND NOT EXISTS (
                    SELECT 1 FROM agent_runs AS active_run
                    WHERE active_run.graph_thread_id = ar.graph_thread_id
                      AND active_run.status IN ('running', 'waiting_approval')
                  )
            GROUP BY ar.graph_thread_id
            ORDER BY finished_at DESC
            """
        ).fetchall()
    thread_ids: list[str] = []
    for index, row in enumerate(rows):
        finished_at = row["finished_at"] or ""
        over_age = bool(finished_at and finished_at < cutoff)
        over_count = index >= max_completed_threads
        if over_age or over_count:
            thread_ids.append(row["graph_thread_id"])
    return thread_ids


async def cleanup_agent_runtime_checkpoints(
    *,
    retention_days: int = 7,
    max_completed_threads: int = 500,
) -> dict[str, int]:
    thread_ids = _completed_checkpoint_threads(
        retention_days=retention_days,
        max_completed_threads=max_completed_threads,
    )
    deleted_threads = 0
    if not thread_ids:
        return {"deleted_threads": 0}
    if _checkpointer is not None and hasattr(_checkpointer, "adelete_thread"):
        for thread_id in thread_ids:
            await _checkpointer.adelete_thread(thread_id)
            deleted_threads += 1
        return {"deleted_threads": deleted_threads}

    close_after = False
    conn = _checkpoint_connection
    if conn is None:
        if not os.path.exists(CHECKPOINT_DB_PATH):
            return {"deleted_threads": 0}
        conn = await aiosqlite.connect(CHECKPOINT_DB_PATH)
        close_after = True
    try:
        for thread_id in thread_ids:
            await conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            deleted_threads += 1
        await conn.commit()
    finally:
        if close_after:
            await conn.close()
    return {"deleted_threads": deleted_threads}
