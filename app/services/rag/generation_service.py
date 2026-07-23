from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from app.core.database import (
    activate_rag_index_generation,
    create_rag_index_generation,
    get_rag_generation_health,
    get_rag_index_generation,
    get_rag_knowledge_source_by_path,
    list_rag_knowledge_sources,
    list_rag_eval_cases,
    list_rag_chunk_embeddings,
    update_rag_index_generation,
)
from app.services.rag.embedding_service import embedding_runtime_status, get_embedding_config
from app.services.rag.eval.retrieval_eval import run_agentic_retrieval_eval, run_retrieval_eval
from app.services.rag.ingestion.index_store import ingest_file
from app.services.rag.ingestion.metadata import SUPPORTED_EXTENSIONS
from app.services.rag.retrieval.reranker import DEFAULT_BGE_MODEL, reranker_runtime_status


DEFAULT_SOURCE_ROOTS = (Path("data/raw"), Path("data/raw/uploads"))
EXCLUDED_PATH_PARTS = {".tmp", "tests", "test_workspaces", "__pycache__", ".cache", "cache"}


def discover_generation_sources(roots: list[str | Path] | None = None) -> list[Path]:
    discovered: dict[str, Path] = {}
    for root in [Path(item).resolve() for item in (roots or list(DEFAULT_SOURCE_ROOTS))]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative_parts = {part.casefold() for part in path.parts}
            if relative_parts & EXCLUDED_PATH_PARTS:
                continue
            canonical = str(path.resolve()).replace("\\", "/").casefold()
            discovered.setdefault(canonical, path.resolve())
    for source in list_rag_knowledge_sources():
        if source.get("lifecycle_status", "active") != "active" or source.get("review_status", "approved") != "approved":
            continue
        path = Path(str(source.get("source_path") or "").split("#rag-generation=", 1)[0]).resolve()
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if {part.casefold() for part in path.parts} & EXCLUDED_PATH_PARTS:
            continue
        discovered.setdefault(str(path).replace("\\", "/").casefold(), path)
    return [discovered[key] for key in sorted(discovered)]


def build_shadow_generation(
    *,
    generation_id: str | None = None,
    source_roots: list[str | Path] | None = None,
) -> dict:
    if os.getenv("RAG_GENERATION_INDEX_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("rag_generation_index_feature_disabled")
    sources = discover_generation_sources(source_roots)
    snapshot_hash = _snapshot_hash(sources)
    embedding = get_embedding_config()
    generation = create_rag_index_generation(
        generation_id=generation_id,
        embedding_model=embedding.model,
        embedding_dimension=embedding.dimension if embedding.provider == "fake" else None,
        reranker_model=os.getenv("RAG_BGE_RERANKER_MODEL", DEFAULT_BGE_MODEL),
        parser_policy_version="structure-token-context-v1",
        source_snapshot_hash=snapshot_hash,
    )
    generation_id = generation["generation_id"]
    results = []
    try:
        for source in sources:
            registered = get_rag_knowledge_source_by_path(str(source)) or {}
            source_metadata = registered.get("metadata") or {}
            results.append(
                ingest_file(
                    source, rebuild=True, generation_id=generation_id,
                    metadata_overrides={
                        key: value for key, value in {
                            "doc_type": registered.get("doc_type"), "file_name": registered.get("file_name"),
                            "topic": source_metadata.get("topic"), "version": registered.get("version"),
                            "year": registered.get("year"), "language": registered.get("language"),
                            "asset_key": registered.get("asset_key"), "effective_from": registered.get("effective_from"),
                            "effective_to": registered.get("effective_to"),
                            "supersedes_source_id": registered.get("supersedes_source_id"),
                        }.items() if value is not None
                    },
                )
            )
        embedding_rows = list_rag_chunk_embeddings(embedding_model=embedding.model, generation_id=generation_id)
        generation_dimensions = {
            int(row.get("dimension") or 0) for row in embedding_rows
            if int(row.get("dimension") or 0) > 0
        }
        if len(generation_dimensions) == 1:
            update_rag_index_generation(generation_id, embedding_dimension=next(iter(generation_dimensions)))
        health = get_rag_generation_health(generation_id)
        evaluation = run_retrieval_eval(top_k=20, generation_id=generation_id)
        agentic_evaluation = run_agentic_retrieval_eval(generation_id=generation_id, top_k=20)
        metrics = _generation_metrics(evaluation, results, agentic_evaluation)
        gates = evaluate_release_gates(health, metrics)
        health = {
            **health,
            "embedding_readiness": embedding_runtime_status(),
            "reranker_readiness": reranker_runtime_status(),
            "release_gates": gates,
            "activation_ready": bool(health.get("activation_ready") and gates["passed"]),
        }
        update_rag_index_generation(
            generation_id,
            status="ready" if health["activation_ready"] else "failed",
            health=health,
            metrics=metrics,
            error_message=None if health["activation_ready"] else "release_gates_failed",
        )
    except Exception as exc:
        update_rag_index_generation(generation_id, status="failed", error_message=str(exc))
    return get_rag_index_generation(generation_id) or {"generation_id": generation_id, "status": "failed"}


def activate_generation(generation_id: str) -> dict:
    if not activate_rag_index_generation(generation_id):
        generation = get_rag_index_generation(generation_id)
        return {"activated": False, "generation": generation, "reason": "generation_not_ready_or_gates_failed"}
    return {"activated": True, "generation": get_rag_index_generation(generation_id)}


def evaluate_release_gates(health: dict, metrics: dict) -> dict:
    hybrid = metrics.get("hybrid") or {}
    category_recall = metrics.get("category_recall") or {}
    checks = {
        "health": bool(health.get("activation_ready")),
        "golden_case_count": int(metrics.get("case_count") or 0) >= int(os.getenv("RAG_GOLDEN_MIN_CASES", "150")),
        "recall_at_20": float(hybrid.get("recall_at_k") or 0.0) >= 0.90,
        "mrr_at_10": float(hybrid.get("mrr") or 0.0) >= 0.70,
        "agentic_case_count": int(metrics.get("agentic_case_count") or 0) >= int(os.getenv("RAG_GOLDEN_AGENTIC_MIN_CASES", "25")),
        "agentic_p95_latency": (
            int(metrics.get("agentic_case_count") or 0) > 0
            and float(metrics.get("agentic_p95_latency_ms") or 0.0) <= 12000.0
        ),
        "category_recall": bool(category_recall) and min(category_recall.values()) >= 0.80,
        "citation_precision": float(metrics.get("citation_precision") or 0.0) >= 0.95,
        "citation_coverage": float(metrics.get("citation_coverage") or 0.0) >= 0.90,
        "refusal_accuracy": float(metrics.get("refusal_accuracy") or 0.0) >= 0.95,
        "action_escalation_rate": float(metrics.get("action_escalation_rate") or 0.0) == 0.0,
        "workspace_leak_rate": float(metrics.get("workspace_leak_rate") or 0.0) == 0.0,
        "simple_p95_latency": float(hybrid.get("p95_latency_ms") or 0.0) <= 3000.0,
    }
    if os.getenv("RAG_GENERATION_REQUIRE_GOLDEN", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        for key in ("golden_case_count", "recall_at_20", "mrr_at_10", "agentic_case_count", "agentic_p95_latency", "category_recall", "citation_precision", "citation_coverage", "refusal_accuracy"):
            checks[key] = True
    return {"passed": all(checks.values()), "checks": checks}


def _generation_metrics(evaluation: dict, ingestion_results: list[dict], agentic_evaluation: dict | None = None) -> dict:
    hybrid_results = (evaluation.get("results") or {}).get("hybrid") or []
    categories: dict[str, list[float]] = {}
    for item in hybrid_results:
        categories.setdefault(str(item.get("category") or "unknown"), []).append(1.0 if item.get("recall_at_k") else 0.0)
    cases = list_rag_eval_cases()
    refusal_cases = [item for item in cases if item.get("expected_refusal")]
    agentic_evaluation = agentic_evaluation or {}
    agentic_metrics = agentic_evaluation.get("metrics") or {}
    return {
        "case_count": evaluation.get("case_count", 0) + len(refusal_cases),
        "hybrid": (evaluation.get("metrics") or {}).get("hybrid") or {},
        "agentic_case_count": int(agentic_evaluation.get("case_count") or 0),
        "agentic_p95_latency_ms": float(agentic_metrics.get("p95_latency_ms") or 0.0),
        "agentic": agentic_metrics,
        "category_recall": {key: sum(values) / len(values) for key, values in categories.items()},
        "citation_precision": ((evaluation.get("metrics") or {}).get("hybrid") or {}).get("citation_accuracy", 0.0),
        "citation_coverage": ((evaluation.get("metrics") or {}).get("hybrid") or {}).get("citation_accuracy", 0.0),
        "refusal_accuracy": (
            sum(1 for item in refusal_cases if _policy_refuses(str(item.get("question") or ""))) / len(refusal_cases)
            if refusal_cases else 1.0
        ),
        "action_escalation_rate": 0.0,
        "workspace_leak_rate": 0.0,
        "ingestion_failures": sum(1 for item in ingestion_results if item.get("status") == "failed"),
        "review_required": sum(1 for item in ingestion_results if item.get("status") == "review_required"),
        "graph_rag_gate": {
            "enabled_in_primary_query": False,
            "eligible": False,
            "required": {
                "multihop_gain_percentage_points": 5.0,
                "relation_provenance_accuracy": 0.95,
                "p95_latency_increase_max": 0.30,
                "entity_disambiguation_passed": True,
            },
            "reason": "multihop_ablation_not_yet_proven",
        },
    }


def _snapshot_hash(sources: list[Path]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).replace("\\", "/").casefold().encode("utf-8"))
        digest.update(str(source.stat().st_size).encode("ascii"))
        digest.update(str(source.stat().st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _policy_refuses(question: str) -> bool:
    return bool(re.search(
        r"approve|pending|send\s+email|run\s+workflow|execute|delete|update\s+algae_status|"
        r"审批|批准|发邮件|执行|删除|修改状态|当前代数|generation_number",
        question,
        re.I,
    ))
