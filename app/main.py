import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core import database
from app.core.database import init_db
from app.core.tasks import daily_schedule_monitor
from app.core.time_utils import time_days_ago
from app.services.agent_runtime.approval_resume import process_pending_resume_jobs
from app.services.agent_runtime.checkpoint import (
    cleanup_agent_runtime_checkpoints,
    initialize_agent_runtime_graph,
    shutdown_agent_runtime_graph,
)
from app.services.agent_runtime.graph_runtime import build_agent_runtime_graph
from app.core.db import operations
from app.services.chat.legacy_task_migration import import_legacy_task_states
from app.services.user_memory.migration import import_legacy_user_memories
from app.services.user_memory.curator import curate_all_user_memories
from app.services.rag.retrieval.reranker import prewarm_reranker, reranker_runtime_status
from app.core.model_registry import model_runtime_status, probe_model_capabilities
from app.services.rag.embedding_service import embedding_runtime_status, probe_embedding_capability
from app.core.db.connection import connect, connect_domain_readonly
from app.core.workspaces import current_workspace


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    import_legacy_task_states()
    import_legacy_user_memories()
    curate_all_user_memories()
    operations.fail_interrupted_operations()
    await initialize_agent_runtime_graph(build_agent_runtime_graph)
    await process_pending_resume_jobs()
    if (
        os.getenv("ALGAE_AUTH_MODE", "required").casefold() == "test"
        or os.getenv("TRUSTED_WORKER_INLINE", "false").casefold() in {"1", "true", "yes", "on"}
    ):
        from app.services.approval_execution_service import recover_approved_executions

        await recover_approved_executions()
    await cleanup_agent_runtime_checkpoints(retention_days=7, max_completed_threads=500)
    database.cleanup_completed_resume_jobs(older_than=time_days_ago(7))

    try:
        app.state.model_capabilities = await asyncio.wait_for(
            asyncio.to_thread(probe_model_capabilities),
            timeout=float(os.getenv("MODEL_CAPABILITY_PROBE_TIMEOUT_SECONDS", "30")),
        )
    except asyncio.TimeoutError:
        app.state.model_capabilities = {
            "status": "unavailable",
            "probe": "timeout",
            "roles": {},
        }
    try:
        app.state.embedding_capability = await asyncio.wait_for(
            asyncio.to_thread(probe_embedding_capability),
            timeout=float(os.getenv("RAG_EMBEDDING_PROBE_TIMEOUT_SECONDS", "20")),
        )
    except asyncio.TimeoutError:
        app.state.embedding_capability = {
            "status": "degraded",
            "degraded": True,
            "degraded_reason": "embedding_probe_timeout",
        }

    try:
        app.state.rag_reranker = await asyncio.wait_for(asyncio.to_thread(prewarm_reranker), timeout=60)
    except asyncio.TimeoutError:
        app.state.rag_reranker = reranker_runtime_status(reason="prewarm_timeout_60s")

    monitor_task = asyncio.create_task(daily_schedule_monitor())
    scheduler_task = None
    if os.getenv("TRUSTED_WORKER_INLINE", "false").casefold() in {"1", "true", "yes", "on"}:
        from app.services.notifications.scheduler_service import email_scheduler_loop

        scheduler_task = asyncio.create_task(email_scheduler_loop())

    yield

    for task in (monitor_task, scheduler_task):
        if task is None:
            continue
        task.cancel()
    for task in (monitor_task, scheduler_task):
        if task is None:
            continue
        try:
            await task
        except asyncio.CancelledError:
            pass
    await shutdown_agent_runtime_graph()


app = FastAPI(
    title="Algae Agent API",
    description="Algae lab agent API with SQLite, LangGraph, and approval boundaries.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health/live", include_in_schema=False)
async def health_live():
    """Process liveness probe; intentionally does not depend on optional services."""
    return {"status": "ok"}


@app.get("/health/ready", include_in_schema=False)
async def health_ready():
    """Readiness probe for storage and configured model capabilities."""
    checks: dict[str, dict] = {}
    ready = True

    try:
        with connect() as conn:
            conn.execute("SELECT 1").fetchone()
        checks["database"] = {"status": "ready"}
    except Exception as exc:
        ready = False
        checks["database"] = {"status": "unavailable", "reason": type(exc).__name__}

    if current_workspace().storage_mode == "split":
        try:
            with connect_domain_readonly() as conn:
                conn.execute("SELECT 1 FROM algae_status LIMIT 1").fetchone()
            checks["domain_database"] = {"status": "ready", "mode": "read_only"}
        except Exception as exc:
            ready = False
            checks["domain_database"] = {
                "status": "unavailable",
                "reason": type(exc).__name__,
            }

    reranker_enabled = os.getenv("RAG_RERANKER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    if reranker_enabled:
        reranker = reranker_runtime_status()
        checks["reranker"] = reranker
        if reranker.get("status") != "ready":
            ready = False
    else:
        checks["reranker"] = {"status": "disabled"}

    models = getattr(app.state, "model_capabilities", None) or model_runtime_status()
    checks["models"] = models
    if models.get("status") != "ready":
        ready = False

    embedding = {
        **embedding_runtime_status(),
        **(getattr(app.state, "embedding_capability", None) or {}),
    }
    checks["embedding"] = embedding
    # Retrieval is an optional evidence source. Its failure is visible but
    # degrades readiness instead of making all database-backed work unavailable.

    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if ready else 503, content=payload)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        item.strip()
        for item in os.getenv(
            "ALGAE_ALLOWED_ORIGINS",
            "http://127.0.0.1:8501,http://localhost:8501,http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if item.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return RedirectResponse(url="/app")


app.include_router(api_router)

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if WEB_DIST.exists():
    assets = WEB_DIST / "assets"
    if assets.exists():
        app.mount("/app/assets", StaticFiles(directory=assets), name="web-assets")

    @app.get("/app", include_in_schema=False)
    @app.get("/app/{path:path}", include_in_schema=False)
    async def react_app(path: str = ""):
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/app", include_in_schema=False)
    async def react_app_unbuilt():
        return {
            "status": "frontend_not_built",
            "message": "Build the React workspace in web/ before opening /app.",
        }
