import re
from dataclasses import dataclass, field

from app.models.rag_evidence_schema import AnswerabilityResult, EvidencePlan, EvidenceUnit, QueryFrame


TYPE_PRIORITY = {
    "recipe_component": 0,
    "table_column": 1,
    "data_value": 2,
    "sop_fact": 3,
    "paper_claim": 4,
    "text_chunk": 5,
}


@dataclass
class EvidenceSelection:
    selected: list[EvidenceUnit]
    rejected_ids: list[str] = field(default_factory=list)
    coverage: str = "unknown"
    missing_evidence: list[str] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def select_evidence_for_answer(
    frame: QueryFrame,
    plan: EvidencePlan,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
    max_items: int = 12,
) -> EvidenceSelection:
    if not evidence:
        return EvidenceSelection(
            selected=[],
            coverage="none",
            missing_evidence=answerability.missing_evidence or plan.required_capabilities,
            debug={"reason": "no_evidence"},
        )

    if answerability.usable_evidence_ids:
        wanted = set(answerability.usable_evidence_ids)
        candidates = [item for item in evidence if item.evidence_id in wanted]
    else:
        candidates = evidence

    keep_many = bool(
        answerability.closed_world_negative
        or frame.target_attribute in {"component", "column", "entity_overview"}
    )
    limit = max_items if not keep_many else max(max_items, 25)
    ranked = sorted(
        _dedupe(candidates),
        key=lambda item: (
            TYPE_PRIORITY.get(item.fact_type, 99),
            -_relevance_score(frame, item),
            item.source_file,
            str(item.location),
            item.evidence_id,
        ),
    )
    selected = _select_with_type_coverage(ranked, plan, limit)
    selected_ids = {item.evidence_id for item in selected}
    rejected_ids = [item.evidence_id for item in evidence if item.evidence_id not in selected_ids]
    coverage = _coverage(answerability, selected)
    return EvidenceSelection(
        selected=selected,
        rejected_ids=rejected_ids,
        coverage=coverage,
        missing_evidence=answerability.missing_evidence,
        debug={
            "candidate_count": len(evidence),
            "selected_count": len(selected),
            "rejected_count": len(rejected_ids),
            "coverage": coverage,
            "closed_world_negative": answerability.closed_world_negative,
        },
    )


def _dedupe(evidence: list[EvidenceUnit]) -> list[EvidenceUnit]:
    seen = set()
    deduped = []
    for item in evidence:
        key = (
            item.fact_type,
            item.source_file,
            item.entity,
            item.attribute,
            item.value,
            item.text_span,
            tuple(sorted(item.location.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _relevance_score(frame: QueryFrame, item: EvidenceUnit) -> float:
    query_terms = _terms(" ".join([frame.original_question, frame.target_value or "", frame.target_attribute or ""]))
    evidence_terms = _terms(
        " ".join(
            str(part or "")
            for part in [
                item.source_file,
                item.entity,
                item.attribute,
                item.value,
                item.text_span,
                item.metadata.get("group_name"),
                item.metadata.get("inferred_role"),
            ]
        )
    )
    if not query_terms or not evidence_terms:
        return float(item.confidence or 0.0)
    overlap = len(query_terms & evidence_terms)
    exact_target_bonus = 0.0
    target = (frame.target_value or "").lower()
    if target and target in (item.value or "").lower():
        exact_target_bonus = 2.0
    if target and target in (item.attribute or "").lower():
        exact_target_bonus = max(exact_target_bonus, 1.5)
    return overlap + exact_target_bonus + float(item.confidence or 0.0)


def _coverage(answerability: AnswerabilityResult, selected: list[EvidenceUnit]) -> str:
    if answerability.status == "answered" and selected:
        return "sufficient"
    if answerability.status == "partial" and selected:
        return "partial"
    if answerability.status == "not_found" and selected:
        return "background_only"
    return answerability.status


def _select_with_type_coverage(
    ranked: list[EvidenceUnit],
    plan: EvidencePlan,
    limit: int,
) -> list[EvidenceUnit]:
    fact_type_by_preference = {
        "recipe_component": "recipe_component",
        "table_schema": "table_column",
        "data_analysis": "data_value",
        "sop_fact": "sop_fact",
        "paper_fact": "paper_claim",
        "text_chunk": "text_chunk",
    }
    selected = []
    selected_ids = set()
    for preference in plan.preferred_evidence_types:
        fact_type = fact_type_by_preference.get(preference)
        item = next((candidate for candidate in ranked if candidate.fact_type == fact_type), None)
        if item and item.evidence_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.evidence_id)
    for item in ranked:
        if len(selected) >= limit:
            break
        if item.evidence_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.evidence_id)
    return selected


def _terms(text: str) -> set[str]:
    raw = str(text or "").lower().replace("�", "u").replace("�", "u")
    return {
        item.strip("_-+")
        for item in re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", raw)
        if item.strip("_-+")
    }
