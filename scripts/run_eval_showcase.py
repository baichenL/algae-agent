from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation.metrics import (
    bootstrap_ci,
    bootstrap_binary_classification_ci,
    bootstrap_classification_ci,
    classification_metrics,
    percentile,
    wilson_ci,
)
from app.services.evaluation.reporting import (
    SCHEMA_VERSION,
    metric_point,
    not_run_metric,
    save_report_bundle,
    utc_now_iso,
)
from scripts.run_agent_eval import configure_temp_db, load_cases, run_case, run_eval
from scripts.run_context_eval import run_context_eval
from scripts.run_scientific_eval import run_eval as run_scientific_eval


TAU_BENCH = "https://arxiv.org/abs/2406.12045"
BFCL = "https://gorilla.cs.berkeley.edu/leaderboard.html"
BEIR = "https://openreview.net/pdf?id=wCu6T5xFjeJ"
RAGAS = "https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/"
ARES = "https://aclanthology.org/2024.naacl-long.20/"
ALCE = "https://arxiv.org/abs/2305.14627"
SKLEARN_CLASSIFICATION = "https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics"
SKLEARN_REGRESSION = "https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _ratio_metric(
    *,
    key: str,
    name: str,
    successes: int,
    total: int,
    standard: str,
    reference: str,
    definition: str,
    evidence_mode: str = "deterministic",
    target: float | None = None,
) -> dict[str, Any]:
    return metric_point(
        key=key,
        display_name=name,
        value=successes / total if total else None,
        unit="ratio",
        numerator=successes if total else None,
        denominator=total if total else None,
        ci95=wilson_ci(successes, total),
        standard=standard,
        reference_url=reference,
        evidence_mode=evidence_mode,
        project_target=target,
        definition=definition,
    )


def _continuous_metric(
    *,
    key: str,
    name: str,
    values: list[float],
    unit: str,
    standard: str,
    reference: str,
    definition: str,
    reducer: Callable[[list[float]], float] = mean,
) -> dict[str, Any]:
    return metric_point(
        key=key,
        display_name=name,
        value=reducer(values) if values else None,
        unit=unit,
        denominator=len(values) if values else None,
        ci95=bootstrap_ci(values, statistic=reducer),
        standard=standard,
        reference_url=reference,
        definition=definition,
    )


def _agent_suite() -> dict[str, Any]:
    runtime_cases = load_cases()
    zh_path = PROJECT_ROOT / "tests" / "fixtures" / "agent_eval_cases_real_zh.json"
    zh_cases = load_cases(zh_path)
    runtime = run_eval(runtime_cases, mode="runtime")
    chinese = run_eval(zh_cases, mode="runtime")
    runtime_results = [
        {**item, "slice_language": "mixed_runtime"}
        for item in runtime["results"]
    ]
    chinese_results = [
        {**item, "slice_language": "zh"}
        for item in chinese["results"]
    ]
    results = runtime_results + chinese_results
    passed = sum(bool(item["passed"]) for item in results)

    route_rows = [
        item
        for item in results
        if item.get("expected", {}).get("route_kind") is not None
        and item.get("actual", {}).get("route_kind") is not None
    ]
    classification = classification_metrics(
        [str(item["expected"]["route_kind"]) for item in route_rows],
        [str(item["actual"]["route_kind"]) for item in route_rows],
    )
    expected_routes = [str(item["expected"]["route_kind"]) for item in route_rows]
    predicted_routes = [str(item["actual"]["route_kind"]) for item in route_rows]
    metrics = [
        _ratio_metric(
            key="task_success_rate",
            name="Task Success Rate / Pass¹",
            successes=passed,
            total=len(results),
            standard="τ-bench Pass¹",
            reference=TAU_BENCH,
            definition="成功完成一次完整任务轨迹的用例比例。",
        ),
        not_run_metric(
            key="provider_pass_at_3",
            display_name="Provider Canary Pass³",
            unit="ratio",
            standard="τ-bench Pass^k",
            reference_url=TAU_BENCH,
            evidence_mode="llm_judge",
            reason="默认 deterministic showcase 不调用真实 Provider；需单独 canary 配置与三次独立轨迹",
            definition="每个任务在三次独立在线运行中全部成功的比例。",
        ),
    ]
    for key, name in (
        ("accuracy", "Router Accuracy"),
        ("macro_precision", "Router Macro Precision"),
        ("macro_recall", "Router Macro Recall"),
        ("macro_f1", "Router Macro-F1"),
    ):
        value = classification[key]
        metrics.append(
            metric_point(
                key=f"router_{key}",
                display_name=name,
                value=value,
                unit="ratio",
                numerator=classification["correct_count"] if key == "accuracy" else None,
                denominator=classification["sample_count"],
                ci95=(
                    wilson_ci(
                        classification["correct_count"],
                        classification["sample_count"],
                    )
                    if key == "accuracy"
                    else bootstrap_classification_ci(
                        expected_routes,
                        predicted_routes,
                        key,
                        labels=classification["labels"],
                    )
                ),
                standard="Standard multiclass classification metrics",
                reference_url=SKLEARN_CLASSIFICATION,
                definition="按路由类别计算；Macro 指标对每个类别等权平均。",
            )
        )
    failures = [
        {
            "id": item["case_id"],
            "expected": item["expected"],
            "actual": item["actual"],
            "failures": item["failures"],
        }
        for item in results
        if not item["passed"]
    ]
    return {
        "key": "agent",
        "display_name": "Agent 整体",
        "status": "completed",
        "description": "离线 deterministic harness；不代表在线模型能力。",
        "sample_count": len(results),
        "metrics": metrics,
        "details": {
            "runtime_case_count": len(runtime["results"]),
            "chinese_case_count": len(chinese["results"]),
            "router": classification,
            "replan_diagnostic": {
                "name": "Replan observed rate",
                "value": runtime.get("replan_observed_rate"),
                "headline_metric": False,
            },
            "failure_slices": _failure_slices(results),
            "failures": failures,
        },
    }


def _failure_slices(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in results:
        expected = item.get("expected") or {}
        buckets.setdefault(
            ("category", str(expected.get("route_kind") or "unlabeled")),
            [],
        ).append(item)
        buckets.setdefault(
            ("language", str(item.get("slice_language") or "unlabeled")),
            [],
        ).append(item)
        buckets.setdefault(
            ("difficulty", str(item.get("difficulty") or "unlabeled")),
            [],
        ).append(item)
    return [
        {
            "dimension": dimension,
            "slice": label,
            "total": len(items),
            "failed": sum(not item.get("passed") for item in items),
        }
        for (dimension, label), items in sorted(buckets.items())
    ]


def _tool_suite() -> dict[str, Any]:
    fixture = json.loads(
        (PROJECT_ROOT / "tests" / "fixtures" / "tool_eval_cases.json").read_text(encoding="utf-8")
    )
    cases_by_id = {item["id"]: item for item in load_cases()}
    rows = []
    for spec in fixture["cases"]:
        case = cases_by_id[spec["agent_case_id"]]
        result = run_case(case, mode=spec["mode"])
        expected = result.expected
        actual = result.actual
        expected_action = expected.get("action")
        actual_action = actual.get("action")
        expected_args = case.get("args") or expected.get("action_args")
        actual_args = actual.get("action_args") or case.get("args")
        should_call = bool(spec["should_call"])
        predicted_call = actual_action not in {
            None,
            "chat",
            "routing_clarification",
            "workflow_request_clarification",
            "pending_form",
        }
        ast_match = expected_action == actual_action and (
            expected_args is None or expected_args == actual_args
        )
        terminal_match = expected.get("terminal_status") == actual.get("terminal_status")
        rows.append(
            {
                "id": spec["id"],
                "selection": expected_action == actual_action,
                "ast": ast_match,
                "executable": (
                    bool(actual.get("tool_executed")) == bool(spec["expected_executable"])
                    if "expected_executable" in spec
                    else None
                ),
                "relevance": should_call == predicted_call,
                "multi_turn": bool(spec.get("multi_turn")),
                "state": terminal_match,
                "response": (
                    expected.get("response_action") == actual.get("response_action")
                    if expected.get("response_action") is not None
                    else terminal_match
                ),
                "expected": expected,
                "actual": actual,
                "failures": result.failures,
            }
        )
    definitions = (
        ("tool_selection_accuracy", "Tool Selection Accuracy", "selection", rows),
        ("ast_exact_match_accuracy", "AST Exact Match Accuracy", "ast", rows),
        (
            "executable_accuracy",
            "Executable Accuracy",
            "executable",
            [item for item in rows if item["executable"] is not None],
        ),
        ("relevance_detection_accuracy", "Relevance Detection Accuracy", "relevance", rows),
        (
            "multi_turn_state_accuracy",
            "Multi-turn State Accuracy",
            "state",
            [item for item in rows if item["multi_turn"]],
        ),
        (
            "multi_turn_response_accuracy",
            "Multi-turn Response Accuracy",
            "response",
            [item for item in rows if item["multi_turn"]],
        ),
    )
    metrics = [
        _ratio_metric(
            key=key,
            name=name,
            successes=sum(bool(item[field]) for item in selected),
            total=len(selected),
            standard="BFCL-compatible local evaluation",
            reference=BFCL,
            definition="按本地固定 Tool Gold 的结构化期望与运行结果计算；不是 BFCL 官方榜单成绩。",
        )
        for key, name, field, selected in definitions
    ]
    return {
        "key": "tool",
        "display_name": "Tool Calling",
        "status": "completed",
        "sample_count": len(rows),
        "description": "BFCL-compatible local evaluation；并非官方 BFCL leaderboard score。",
        "metrics": metrics,
        "details": {"dataset_id": fixture["dataset_id"], "cases": rows},
    }


def _rag_suite(work_dir: Path) -> dict[str, Any]:
    gold_path = PROJECT_ROOT / "tests" / "fixtures" / "interview_rag_v1.json"
    if not gold_path.exists():
        return _not_run_rag_suite("interview-rag-v1 Gold 文件缺失")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    cases = gold.get("cases") or []
    reviewed = [item for item in cases if item.get("human_reviewed") is True]
    if len(cases) != 50 or len(reviewed) != 50:
        return _not_run_rag_suite(
            f"Gold 候选 {len(cases)} 条，其中人工审核 {len(reviewed)}/50；正式检索分数禁止生成"
        )
    configure_temp_db(work_dir / "rag_eval.sqlite3")
    os.environ["RAG_EMBEDDING_ENABLED"] = "true"
    os.environ["RAG_EMBEDDING_PROVIDER"] = "fake"
    os.environ["RAG_EMBEDDING_MODEL"] = "fake-interview-rag-v1"
    os.environ["RAG_RERANKER_ENABLED"] = "true"
    os.environ["RAG_RERANKER_BACKEND"] = "fake"
    from app.core.database import upsert_rag_eval_cases
    from app.services.rag.ingestion.index_store import ingest_file

    corpus_root = PROJECT_ROOT / "tests" / "fixtures" / "rag_gold_corpus"
    ingestion = []
    for path in sorted(corpus_root.rglob("*.md")):
        relative = path.relative_to(corpus_root).as_posix()
        doc_type = (
            "paper"
            if relative.startswith("papers/")
            else "experiment_data"
            if relative.startswith("datasets/")
            else "media_recipe"
            if "recipe" in path.name.casefold()
            else "manual"
        )
        ingestion.append(
            ingest_file(
                path,
                metadata_overrides={"doc_type": doc_type, "topic": path.stem},
            )
        )
    failed_ingestion = [item for item in ingestion if item.get("status") == "failed"]
    if failed_ingestion:
        return _not_run_rag_suite(
            f"Gold corpus ingestion failed for {len(failed_ingestion)} source(s)"
        )
    upsert_rag_eval_cases(
        [
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_route": "knowledge_query",
                "required_evidence_types": [],
                "expected_answer_contains": [],
                "expected_evidence_labels": case["relevant_evidence_set"],
                "expected_refusal": not bool(case["relevant_evidence_set"]),
                "notes": json.dumps(
                    {
                        "expected_status": case["expected_status"],
                        "capability_tags": case["capability_tags"],
                        "difficulty": case["difficulty"],
                        "language": case["language"],
                    },
                    ensure_ascii=False,
                ),
            }
            for case in cases
        ]
    )
    from app.services.rag.eval.retrieval_eval import run_retrieval_eval

    report = run_retrieval_eval(top_k=20, include_reranker=True)
    metrics = []
    modes = report.get("metrics") or {}
    primary = modes.get("hybrid_reranker") or {}
    for key, name in (
        ("precision_at_5", "Precision@5"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_20", "Recall@20"),
        ("mrr_at_10", "MRR@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("hit_rate_at_5", "Hit Rate@5"),
    ):
        values = [
            float(item[key])
            for item in (report.get("results") or {}).get("hybrid_reranker", [])
            if item.get(key) is not None
        ]
        metrics.append(
            metric_point(
                key=key,
                display_name=name,
                value=primary.get(key),
                unit="ratio",
                denominator=len(values) or None,
                ci95=(
                    wilson_ci(int(sum(values)), len(values))
                    if key == "hit_rate_at_5" and values
                    else bootstrap_ci(values)
                ),
                standard="BEIR retrieval metric" if key != "hit_rate_at_5" else "Information retrieval hit rate",
                reference_url=BEIR,
                definition="使用完整 relevant evidence set；nDCG 使用 0–3 级相关性。",
            )
        )
    return {
        "key": "rag",
        "display_name": "RAG Retrieval",
        "status": "completed",
        "sample_count": report.get("case_count"),
        "metrics": metrics,
        "details": {
            "gold_id": gold["dataset_id"],
            "gold_version": gold["version"],
            "ablation": modes,
            "results": report.get("results"),
            "embedding": report.get("embedding"),
            "ingestion": ingestion,
        },
    }


def _not_run_rag_suite(reason: str) -> dict[str, Any]:
    metrics = [
        not_run_metric(
            key=key,
            display_name=name,
            unit="ratio",
            standard="BEIR retrieval metric",
            reference_url=BEIR,
            evidence_mode="deterministic",
            reason=reason,
            definition="需要 50 条已人工审核且证据集合完整的 interview-rag-v1。",
        )
        for key, name in (
            ("precision_at_5", "Precision@5"),
            ("recall_at_5", "Recall@5"),
            ("recall_at_20", "Recall@20"),
            ("mrr_at_10", "MRR@10"),
            ("ndcg_at_10", "nDCG@10"),
            ("hit_rate_at_5", "Hit Rate@5"),
        )
    ]
    return {
        "key": "rag",
        "display_name": "RAG Retrieval",
        "status": "not_run",
        "status_reason": reason,
        "sample_count": 0,
        "metrics": metrics,
        "details": {},
    }


def _rag_generation_suite(*, judge: bool) -> dict[str, Any]:
    provider_ready = bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ALGAE_OPENAI_API_KEY")
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
    )
    reason = (
        "未启用 --judge"
        if not judge
        else "OpenAI-compatible Provider API Key 不可用"
        if not provider_ready
        else "Judge adapter 尚未获得 20 组人工 claim–evidence 校准标签"
    )
    metrics = [
        not_run_metric(
            key=key,
            display_name=name,
            unit="ratio",
            standard=standard,
            reference_url=reference,
            evidence_mode="llm_judge",
            reason=reason,
            definition=definition,
        )
        for key, name, standard, reference, definition in (
            ("context_precision", "Context Precision", "RAGAS", RAGAS, "相关上下文在检索上下文中的精确度。"),
            ("context_recall", "Context Recall", "RAGAS", RAGAS, "参考答案所需事实被上下文覆盖的比例。"),
            ("faithfulness", "Faithfulness", "RAGAS/ARES", ARES, "回答 claims 能由上下文支持的比例。"),
            ("answer_relevance", "Answer Relevance", "RAGAS", RAGAS, "回答与问题意图的相关程度。"),
            ("citation_precision", "Citation Precision", "ALCE", ALCE, "引用能够支持其对应 claim 的比例。"),
            ("citation_recall", "Citation Recall / Completeness", "ALCE", ALCE, "需要引用的 claims 中得到引用支持的比例。"),
            ("judge_cohen_kappa", "Judge–Human Cohen’s κ", "Cohen’s kappa", SKLEARN_CLASSIFICATION, "Judge 与人工标签扣除随机一致后的协议度。"),
        )
    ]
    return {
        "key": "rag_generation",
        "display_name": "RAG Generation 与 Citation",
        "status": "not_run",
        "status_reason": reason,
        "sample_count": 0,
        "metrics": metrics,
        "details": {"calibration_threshold": 0.7, "headline_eligible": False},
    }


def _scientific_suite(work_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configure_temp_db(work_dir / "scientific_eval.sqlite3")
    report = run_scientific_eval(work_dir=work_dir)
    raw = report["metrics"]
    standard_specs = [
        ("data_quality_precision", "Data Quality Precision", "ratio", SKLEARN_CLASSIFICATION),
        ("data_quality_recall", "Data Quality Recall", "ratio", SKLEARN_CLASSIFICATION),
        ("data_quality_f1", "Data Quality F1", "ratio", SKLEARN_CLASSIFICATION),
        ("growth_rate_mae", "Growth Rate MAE", "count", SKLEARN_REGRESSION),
        ("growth_rate_rmse", "Growth Rate RMSE", "count", SKLEARN_REGRESSION),
        ("candidate_cause_hit_at_1", "Candidate Cause Hit@1", "ratio", BEIR),
        ("candidate_cause_hit_at_3", "Candidate Cause Hit@3", "ratio", BEIR),
        ("candidate_cause_mrr", "Candidate Cause MRR", "ratio", BEIR),
        ("offline_replay_simple_regret", "Simple Regret", "regret", "https://arxiv.org/abs/1807.02811"),
        ("constraint_violation_rate", "Constraint Violation Rate", "ratio", "https://arxiv.org/abs/1807.02811"),
        ("task_success_rate", "Task Success Rate", "ratio", TAU_BENCH),
    ]
    metrics = []
    for key, name, unit, reference in standard_specs:
        reducer: Callable[[list[float]], float] = mean
        if key == "offline_replay_simple_regret":
            values = list(raw["full_regrets_by_seed"])
        elif key == "growth_rate_mae":
            values = [
                abs(float(predicted) - float(expected))
                for expected, predicted in zip(
                    raw["growth_rate_expected"],
                    raw["growth_rate_predicted"],
                )
            ]
        elif key == "growth_rate_rmse":
            values = [
                (float(predicted) - float(expected)) ** 2
                for expected, predicted in zip(
                    raw["growth_rate_expected"],
                    raw["growth_rate_predicted"],
                )
            ]
            reducer = lambda items: sum(items) ** 0.5 / len(items) ** 0.5
        elif key == "candidate_cause_hit_at_1":
            values = [float(rank == 1) for rank in raw["candidate_cause_ranks"]]
        elif key == "candidate_cause_hit_at_3":
            values = [
                float(rank is not None and rank <= 3)
                for rank in raw["candidate_cause_ranks"]
            ]
        elif key == "candidate_cause_mrr":
            values = [
                1.0 / rank if rank else 0.0
                for rank in raw["candidate_cause_ranks"]
            ]
        else:
            values = [float(raw[key])] if raw.get(key) is not None else []
        denominator = (
            (
                raw["data_quality_tp"] + raw["data_quality_fp"]
                if key == "data_quality_precision"
                else raw["data_quality_tp"] + raw["data_quality_fn"]
                if key == "data_quality_recall"
                else len(raw["data_quality_labels"])
            )
            if key.startswith("data_quality")
            else len(values)
        )
        classification_key = {
            "data_quality_precision": "precision",
            "data_quality_recall": "recall",
            "data_quality_f1": "f1",
        }.get(key)
        ci95 = (
            (
                wilson_ci(raw["data_quality_tp"], denominator)
                if key in {"data_quality_precision", "data_quality_recall"}
                else bootstrap_binary_classification_ci(
                    raw["data_quality_labels"],
                    raw["data_quality_predictions"],
                    classification_key,
                )
            )
            if classification_key
            else (
                wilson_ci(int(round(sum(values))), len(values))
                if key in {
                    "candidate_cause_hit_at_1",
                    "candidate_cause_hit_at_3",
                    "constraint_violation_rate",
                    "task_success_rate",
                }
                and values
                else bootstrap_ci(values, statistic=reducer)
            )
        )
        metrics.append(
            metric_point(
                key=key,
                display_name=name,
                value=reducer(values) if values else None,
                unit=unit,
                numerator=(
                    raw["data_quality_tp"]
                    if key in {"data_quality_precision", "data_quality_recall"}
                    else sum(values)
                    if unit == "ratio" and values and key != "data_quality_f1"
                    else None
                ),
                denominator=denominator,
                ci95=ci95,
                standard=(
                    "Standard classification metric"
                    if key.startswith("data_quality")
                    else "Standard regression metric"
                    if key.startswith("growth_rate")
                    else "Standard ranking metric"
                    if key.startswith("candidate_cause")
                    else "Constrained optimization metric"
                    if key in {"offline_replay_simple_regret", "constraint_violation_rate"}
                    else "τ-bench task success"
                ),
                reference_url=reference,
                definition="按固定 seed 保留原始结果，并报告 bootstrap 95% CI。",
            )
        )
    comparisons = [
        {
            "key": "scientific_replan_ablation",
            "display_name": "Full closed loop vs no-replan",
            "suite_key": "scientific",
            "metric_key": "task_success_rate",
            "variants": [
                {"name": "full_closed_loop", "value": raw["full_loop_failure_recovery_rate"]},
                {"name": "no_replan", "value": raw["no_replan_failure_recovery_rate"]},
            ],
        },
        {
            "key": "scientific_optimizer_ablation",
            "display_name": "Optimizer vs random selection",
            "suite_key": "scientific",
            "metric_key": "offline_replay_simple_regret",
            "variants": [
                {"name": "optimizer", "value": raw["offline_replay_simple_regret"]},
                {"name": "random", "value": raw["random_selection_simple_regret_median"]},
            ],
        },
    ]
    return (
        {
            "key": "scientific",
            "display_name": "Scientific 闭环",
            "status": "completed" if report["acceptance_passed"] else "failed",
            "sample_count": report["scenario_count"],
            "metrics": metrics,
            "details": {
                "citation_coverage": raw["citation_coverage"],
                "cases": report["cases"],
                "baselines": report["baselines"],
            },
        },
        comparisons,
    )


def _memory_suite(work_dir: Path, *, judge: bool) -> dict[str, Any]:
    configure_temp_db(work_dir / "memory_eval.sqlite3")
    fixture = json.loads(
        (PROJECT_ROOT / "tests" / "fixtures" / "user_memory_eval_cases.json").read_text(encoding="utf-8")
    )
    basic = [item for item in fixture if item["tier"] == "basic_recall"]
    multi = [item for item in fixture if item["tier"] == "multi_conversation"]
    proactive = [item for item in fixture if item["tier"] == "proactive_service"]
    from app.services.user_memory.store import apply_memory
    from app.services.user_memory.retriever import retrieve_user_memories

    owner = "eval-owner"
    workspace = "eval-workspace"
    for item in basic:
        apply_memory(
            owner_id=owner,
            workspace_id=workspace,
            memory_type="preference",
            predicate=item["predicate"],
            value=item["value"],
            actor="evaluation",
        )
    retrieval_rows = []
    for item in multi:
        required = item.get("required") or []
        if not required or item.get("isolation"):
            continue
        retrieved = retrieve_user_memories(
            owner_id=owner,
            workspace_id=workspace,
            query=item.get("query") or item.get("turn") or "",
            limit=5,
            record_usage=False,
        )
        predicates = [row["predicate"] for row in retrieved]
        first_rank = next(
            (
                index
                for index, predicate in enumerate(predicates, start=1)
                if predicate in set(required)
            ),
            None,
        )
        retrieval_rows.append(
            {
                "id": item["id"],
                "required": required,
                "retrieved": predicates,
                "recall_at_5": len(set(required) & set(predicates[:5])) / len(set(required)),
                "mrr_at_5": 1.0 / first_rank if first_rank and first_rank <= 5 else 0.0,
            }
        )

    leaked = retrieve_user_memories(
        owner_id="other-owner",
        workspace_id=workspace,
        query="language report units notification visualization",
        limit=20,
        record_usage=False,
    )
    isolation_total = sum(bool(item.get("isolation")) for item in multi)
    leakage_rate = len(leaked) / max(1, len(basic))
    metrics = [
        not_run_metric(
            key=key,
            display_name=name,
            unit="ratio",
            standard="Standard extraction metric",
            reference_url=SKLEARN_CLASSIFICATION,
            evidence_mode="llm_judge",
            reason=(
                "Provider suite 未启用"
                if not judge
                else "抽取结果需要真实 Provider 和人工标签；当前 runner 未发送在线请求"
            ),
        )
        for key, name in (
            ("extraction_exact_match", "Extraction Exact Match"),
            ("extraction_precision", "Extraction Precision"),
            ("extraction_recall", "Extraction Recall"),
            ("extraction_f1", "Extraction F1"),
        )
    ]
    for key, name in (("recall_at_5", "Memory Recall@5"), ("mrr_at_5", "Memory MRR@5")):
        values = [item[key] for item in retrieval_rows]
        metrics.append(
            _continuous_metric(
                key=key,
                name=name,
                values=values,
                unit="ratio",
                standard="Standard information retrieval metric",
                reference=BEIR,
                definition="在隔离 owner/workspace 的临时 SQLite 中检索。",
            )
        )
    metrics.extend(
        [
            metric_point(
                key="data_leakage_rate",
                display_name="Data Leakage Rate",
                value=leakage_rate,
                unit="ratio",
                numerator=len(leaked),
                denominator=len(basic),
                ci95=wilson_ci(len(leaked), len(basic)),
                standard="Cross-tenant isolation failure rate",
                reference_url="https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/",
                definition="其他 owner 能检索到目标 owner 记忆的比例；越低越好。",
            ),
            not_run_metric(
                key="proactive_task_success_rate",
                display_name="Proactive Safety Task Success Rate",
                unit="ratio",
                standard="τ-bench-style task success",
                reference_url=TAU_BENCH,
                evidence_mode="deterministic",
                reason="fixture 已定义 20 条期望，但尚无逐条执行主动服务链路的 deterministic harness",
                definition="必须运行实际主动服务链路；不能用 fixture 中期望标签的数量作为代理分数。",
            ),
        ]
    )
    return {
        "key": "memory",
        "display_name": "User Memory",
        "status": "completed",
        "sample_count": len(fixture),
        "metrics": metrics,
        "details": {
            "retrieval_cases": retrieval_rows,
            "isolation_case_count": isolation_total,
            "fixture_tiers": {"basic": len(basic), "multi": len(multi), "proactive": len(proactive)},
        },
    }


def _context_suite() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = run_context_eval()
    rows = report["results"]
    full_tokens = [float(item["variant_tokens"]["full_pipeline"]) for item in rows]
    no_compression_tokens = [float(item["variant_tokens"]["no_compression"]) for item in rows]
    compression_ratios = [
        full / no_compression
        for full, no_compression in zip(full_tokens, no_compression_tokens)
        if no_compression
    ]
    full_latency = [float(item["variant_latency_ms"]["full_pipeline"]) for item in rows]
    metrics = [
        _continuous_metric(
            key="input_tokens",
            name="Input Tokens",
            values=full_tokens,
            unit="tokens",
            standard="Tokenizer input token count",
            reference="https://platform.openai.com/tokenizer",
            definition="Full pipeline 送入模型前的输入 token 估算。",
        ),
        _continuous_metric(
            key="compression_ratio",
            name="Compression Ratio",
            values=compression_ratios,
            unit="ratio",
            standard="Compression ratio",
            reference="https://en.wikipedia.org/wiki/Data_compression_ratio",
            definition="full pipeline tokens / no-compression tokens；仅为效率指标。",
        ),
        metric_point(
            key="p50_latency",
            display_name="P50 Latency",
            value=percentile(full_latency, 0.50),
            unit="milliseconds",
            denominator=len(full_latency),
            ci95=bootstrap_ci(full_latency, statistic=lambda values: float(percentile(values, 0.50))),
            standard="Latency percentile",
            reference_url="https://opentelemetry.io/docs/specs/otel/metrics/",
            definition="Context assembly 墙钟耗时中位数。",
        ),
        metric_point(
            key="p95_latency",
            display_name="P95 Latency",
            value=percentile(full_latency, 0.95),
            unit="milliseconds",
            denominator=len(full_latency),
            ci95=bootstrap_ci(full_latency, statistic=lambda values: float(percentile(values, 0.95))),
            standard="Latency percentile",
            reference_url="https://opentelemetry.io/docs/specs/otel/metrics/",
            definition="Context assembly 墙钟耗时第 95 百分位。",
        ),
    ]
    comparison = {
        "key": "context_compression_ablation",
        "display_name": "Full context vs no-compression",
        "suite_key": "context",
        "metric_key": "input_tokens",
        "variants": [
            {
                "name": "full_pipeline",
                "tokens_mean": mean(full_tokens),
                "p95_latency_ms": percentile(full_latency, 0.95),
            },
            {
                "name": "no_compression",
                "tokens_mean": mean(no_compression_tokens),
                "p95_latency_ms": percentile(
                    [float(item["variant_latency_ms"]["no_compression"]) for item in rows],
                    0.95,
                ),
            },
        ],
    }
    return (
        {
            "key": "context",
            "display_name": "Context 与效率",
            "status": report["status"],
            "sample_count": report["case_count"],
            "description": "效率指标不能单独证明任务质量提升。",
            "metrics": metrics,
            "details": {"cases": rows},
        },
        [comparison],
    )


def build_report(*, judge: bool = False, work_dir: Path) -> dict[str, Any]:
    generated_at = utc_now_iso()
    report_id = generated_at.replace("-", "").replace(":", "").replace(".", "").replace("Z", "Z")
    agent = _agent_suite()
    tool = _tool_suite()
    rag = _rag_suite(work_dir)
    rag_generation = _rag_generation_suite(judge=judge)
    scientific, scientific_comparisons = _scientific_suite(work_dir)
    memory = _memory_suite(work_dir, judge=judge)
    context, context_comparisons = _context_suite()
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": f"eval-{report_id}",
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "dataset_versions": {
            "agent": "agent-eval-v2",
            "tool": "interview-tool-v1",
            "rag": "interview-rag-v1",
            "scientific": "scientific-offline-v2",
            "memory": "user-memory-60-v1",
            "context": "context-8-scenarios-v1",
        },
        "environment": {
            "embedding_backend": os.getenv("RAG_EMBEDDING_PROVIDER", "disabled/fake"),
            "reranker_backend": os.getenv("RAG_RERANKER_BACKEND", "lexical/fake"),
            "judge_model": os.getenv("RAG_EVAL_JUDGE_MODEL") if judge else None,
            "judge_enabled": judge,
            "embedding_used": rag.get("status") == "completed",
            "reranker_used": rag.get("status") == "completed",
            "judge_used": rag_generation.get("status") == "completed",
        },
        "suites": [agent, tool, rag, rag_generation, scientific, memory, context],
        "comparisons": [
            *scientific_comparisons,
            *context_comparisons,
            {
                "key": "rag_retrieval_ablation",
                "display_name": "FTS vs Vector vs Hybrid vs Hybrid + Reranker",
                "suite_key": "rag",
                "metric_keys": ["recall_at_20", "mrr_at_10", "ndcg_at_10"],
                "variants": (rag.get("details") or {}).get("ablation") or {},
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic interview evaluation report bundle."
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval_reports")
    parser.add_argument("--judge", action="store_true", help="Enable optional online judge suite.")
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Skip Playwright screenshot/PDF generation (useful in minimal CI images).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a deterministic suite fails or is not runnable.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="algae_showcase_eval_", ignore_cleanup_errors=True) as tmp:
        work_dir = Path(tmp)
        configure_temp_db(work_dir / "showcase.sqlite3")
        report = build_report(judge=args.judge, work_dir=work_dir)
    paths = save_report_bundle(report, args.output_dir)
    if not args.skip_capture:
        capture_script = PROJECT_ROOT / "scripts" / "capture_eval_report.mjs"
        screenshot_path = args.output_dir.resolve() / "latest.png"
        pdf_path = args.output_dir.resolve() / "latest.pdf"
        try:
            subprocess.run(
                [
                    "node",
                    str(capture_script),
                    "--input",
                    str(paths["html"]),
                    "--screenshot",
                    str(screenshot_path),
                    "--pdf",
                    str(pdf_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                timeout=120,
            )
            paths["screenshot"] = screenshot_path
            paths["pdf"] = pdf_path
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(
                f"warning: report capture was not generated ({type(exc).__name__}); "
                "JSON/HTML/Markdown remain valid",
                file=sys.stderr,
            )
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "paths": {key: str(value) for key, value in paths.items()},
                "suite_statuses": {
                    item["key"]: item["status"] for item in report["suites"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    deterministic = [
        item
        for item in report["suites"]
        if item["key"] != "rag_generation"
    ]
    failed = [item for item in deterministic if item["status"] not in {"completed", "passed"}]
    missing_required_metrics = [
        (suite["key"], metric["key"])
        for suite in deterministic
        for metric in suite.get("metrics") or []
        if metric.get("evidence_mode") == "deterministic"
        and metric.get("status") != "completed"
    ]
    raise SystemExit(1 if args.strict and (failed or missing_required_metrics) else 0)


if __name__ == "__main__":
    main()
