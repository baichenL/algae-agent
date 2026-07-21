from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.context import ContextAssemblyFeatures, assemble_model_input
from app.services.context.budget import canonical_json, estimate_tokens


def _snapshot(pending_id: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="context-eval",
        created_at="2026-07-19T12:00:00+08:00",
        strains=[
            {
                "strain_id": f"Chlorella_{index:02d}",
                "generation_number": index,
                "days_since_last_subculture": index + 2,
                "subculture_due": index > 4,
            }
            for index in range(1, 16)
        ],
        pending_actions=[{"id": pending_id, "action_type": "subculture_workflow", "status": "pending"}],
        recent_experiments=[{"id": index, "status": "complete", "summary": "growth observation " * 8} for index in range(5)],
        target_strain={"strain_id": "Chlorella_01", "generation_number": 8},
        selected_agent_memories=[],
        selected_user_memories=[],
    )


def _default_cases() -> list[dict[str, Any]]:
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"historical turn {index}: " + ("microalgae experiment context " * 12),
        }
        for index in range(40)
    ]
    return [
        {
            "id": "long_workflow",
            "request_kind": "workflow_write",
            "route_kind": "workflow",
            "message": "为 Chlorella_01 创建传代审批请求",
            "history": history,
        },
        {
            "id": "long_knowledge",
            "request_kind": "knowledge_query",
            "route_kind": "knowledge_query",
            "message": "根据本地证据说明 TAP 培养基适用条件",
            "history": history,
        },
    ]


def _envelope(case: dict[str, Any], features: ContextAssemblyFeatures, *, pending_id: int = 42):
    return assemble_model_input(
        request_kind=case["request_kind"],
        route_kind=case["route_kind"],
        current_user_message=case["message"],
        conversation_history=case["history"],
        context_snapshot=_snapshot(pending_id),
        system_instruction="Conservative laboratory agent; policy and tool observations are authoritative.",
        task_protocol={"task": "Complete the routed request safely.", "output": "typed result"},
        features=features,
    )


def _legacy_tokens(case: dict[str, Any]) -> int:
    messages = [
        {"role": "system", "content": "Conservative laboratory agent."},
        {"role": "system", "content": canonical_json(vars(_snapshot()))},
        *case["history"],
        {"role": "user", "content": case["message"]},
    ]
    return estimate_tokens(messages)


def run_context_eval(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    results = []
    for case in cases or _default_cases():
        variants = {
            "full_pipeline": ContextAssemblyFeatures(),
            "no_status_bar": ContextAssemblyFeatures(status_bar=False),
            "no_dynamic_skills": ContextAssemblyFeatures(dynamic_skills=False),
            "no_compression": ContextAssemblyFeatures(compression=False),
        }
        variant_tokens = {}
        full = None
        for name, features in variants.items():
            envelope = _envelope(case, features)
            variant_tokens[name] = estimate_tokens(envelope.to_messages(case["message"]))
            if name == "full_pipeline":
                full = envelope
        changed = _envelope(case, ContextAssemblyFeatures(), pending_id=99)
        baseline = _legacy_tokens(case)
        reduction = 1.0 - (variant_tokens["full_pipeline"] / max(1, baseline))
        results.append({
            "case_id": case["id"],
            "baseline_tokens": baseline,
            "variant_tokens": variant_tokens,
            "full_pipeline_reduction": round(reduction, 4),
            "prefix_stable_across_dynamic_change": full.budget_report.prefix_hash == changed.budget_report.prefix_hash,
            "compression_record_count": len(full.compression_records),
            "active_skill_count": len(full.active_skills),
            "protected_pending_present": "42" in canonical_json(full.dynamic_context) or "42" in full.status_bar.render(),
        })
    average_reduction = sum(item["full_pipeline_reduction"] for item in results) / max(1, len(results))
    return {
        "status": "passed" if all(
            item["prefix_stable_across_dynamic_change"] and item["protected_pending_present"]
            for item in results
        ) else "failed",
        "case_count": len(results),
        "average_full_pipeline_reduction": round(average_reduction, 4),
        "meets_25pct_reduction_target": average_reduction >= 0.25,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Context Assembly ablation eval.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_context_eval()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["status"] == "passed" and report["meets_25pct_reduction_target"] else 1)


if __name__ == "__main__":
    main()
