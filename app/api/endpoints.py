# app/api/endpoints.py
# 统一注册所有 API 路由，方便在 main.py 中统一 include
from fastapi import APIRouter

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.email import router as email_router
from app.api.rag import router as rag_router
from app.api.strain import router as strain_router
from app.api.workflow import router as workflow_router
from app.api.scientific import lab_router, router as scientific_router
from app.api.v2 import router as v2_router
from app.api.memory import router as memory_router

router = APIRouter()
legacy_router = APIRouter(prefix="/api/v1")
legacy_router.include_router(agent_router)
legacy_router.include_router(chat_router)
legacy_router.include_router(workflow_router)
legacy_router.include_router(strain_router)
legacy_router.include_router(email_router)
legacy_router.include_router(rag_router)
legacy_router.include_router(scientific_router)
legacy_router.include_router(lab_router)
router.include_router(legacy_router)
router.include_router(v2_router)
router.include_router(memory_router)
