from __future__ import annotations

from app.models.rag_evidence_schema import EvidenceUnit, QueryFrame


REQUIRED_EVIDENCE_TYPES = {
    "component_amount": {"recipe_component"},
    "component": {"recipe_component"},
    "component_group": {"recipe_component"},
    "procedure": {"sop_fact"},
    "method_used": {"paper_claim"},
    "column": {"table_column"},
    "environment_variables": {"table_column"},
    "column_value": {"table_column"},
    "max_value": {"data_value"},
    "min_value": {"data_value"},
    "mean_value": {"data_value"},
    "count_value": {"data_value"},
    "trend": {"data_value"},
    "group_mean": {"data_value"},
    "best_condition": {"data_value"},
}


def verify_required_evidence(frame: QueryFrame, evidence: list[EvidenceUnit]) -> dict:
    required = REQUIRED_EVIDENCE_TYPES.get(frame.target_attribute or "", set())
    present = {item.fact_type for item in evidence}
    missing = sorted(required - present)
    if not required:
        return {
            "status": "not_required",
            "required_evidence_types": [],
            "present_evidence_types": sorted(present),
            "missing_evidence_types": [],
            "rewrite_recommended": False,
        }
    status = "sufficient" if not missing else "insufficient"
    return {
        "status": status,
        "required_evidence_types": sorted(required),
        "present_evidence_types": sorted(present),
        "missing_evidence_types": missing,
        "rewrite_recommended": bool(missing),
        "rewrite_query": build_verification_rewrite(frame, missing) if missing else None,
    }


def build_verification_rewrite(frame: QueryFrame, missing: list[str] | None = None) -> str:
    target = frame.target_value or frame.target_entity or ""
    attribute = frame.target_attribute or ""
    missing_text = " ".join(missing or [])
    if "recipe_component" in missing_text:
        return f"{target} recipe component amount medium composition"
    if "sop_fact" in missing_text:
        return f"{target} SOP manual protocol procedure step {attribute}"
    if "paper_claim" in missing_text:
        return f"{target} paper literature method claim {attribute}"
    if "table_column" in missing_text:
        return f"{target} experiment data table column schema"
    if "data_value" in missing_text:
        return f"{target} experiment data numeric value aggregation {attribute}"
    return f"{frame.original_question} {attribute} evidence"
