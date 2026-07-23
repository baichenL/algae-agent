from fastapi import APIRouter, Depends, HTTPException

from app.core.security import ApiPrincipal, require_scientist, require_viewer
from app.core.database import (
    get_rag_index_status,
    list_rag_documents,
    list_rag_graph_neighbors,
    list_rag_knowledge_sources,
    list_rag_relations,
    list_rag_trace_logs,
)
from app.models.rag_schema import RagQueryRequest
from app.services.rag.embedding_service import embedding_runtime_status
from app.services.rag.eval.hybrid_demo import build_hybrid_demo_report
from app.services.rag.eval.mini_eval import run_mini_eval, seed_default_mini_eval_cases
from app.services.rag.eval.retrieval_eval import run_retrieval_eval
from app.services.rag.retrieval.reranker import reranker_runtime_status
from app.services.rag.service import answer_rag_question

router = APIRouter()


@router.post("/rag/query")
async def rag_query(payload: RagQueryRequest, _: ApiPrincipal = Depends(require_viewer)):
    try:
        result = answer_rag_question(payload)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(exc)}")


@router.get("/rag/sources")
async def rag_sources(status: str | None = None, _: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "sources": list_rag_documents(status=status)}


@router.get("/rag/index/status")
async def rag_index_status(_: ApiPrincipal = Depends(require_viewer)):
    return {
        "status": "success",
        "index": get_rag_index_status(),
        "embedding": embedding_runtime_status(),
        "reranker": reranker_runtime_status(),
    }


@router.get("/rag/sources/registry")
async def rag_source_registry(doc_type: str | None = None, _: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "sources": list_rag_knowledge_sources(doc_type=doc_type)}


@router.get("/rag/trace")
async def rag_trace(limit: int = 20, _: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "traces": list_rag_trace_logs(limit=limit)}


@router.post("/rag/eval/mini")
async def rag_mini_eval(_: ApiPrincipal = Depends(require_scientist)):
    seed_default_mini_eval_cases()
    return {"status": "success", "eval": run_mini_eval()}


@router.post("/rag/eval/retrieval")
async def rag_retrieval_eval(top_k: int = 5, _: ApiPrincipal = Depends(require_scientist)):
    seed_default_mini_eval_cases()
    return {"status": "success", "eval": run_retrieval_eval(top_k=top_k)}


@router.post("/rag/eval/hybrid-demo")
async def rag_hybrid_demo(payload: RagQueryRequest, _: ApiPrincipal = Depends(require_scientist)):
    return {
        "status": "success",
        "demo": build_hybrid_demo_report(payload.question, top_k=payload.top_k),
    }


@router.get("/rag/evidence/relations")
async def rag_evidence_relations(entity_id: str | None = None, _: ApiPrincipal = Depends(require_viewer)):
    return {"status": "success", "relations": list_rag_relations(entity_id=entity_id)}


@router.get("/rag/evidence/graph")
async def rag_evidence_graph(
    entity_id: str,
    depth: int = 1,
    limit: int = 50,
    _: ApiPrincipal = Depends(require_viewer),
):
    return {
        "status": "success",
        "graph": list_rag_graph_neighbors(entity_id=entity_id, depth=depth, limit=limit),
    }
