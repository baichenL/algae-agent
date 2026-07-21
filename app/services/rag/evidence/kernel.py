import os

from app.models.rag_evidence_schema import AnswerabilityResult, GroundedAnswer
from app.services.rag.evidence.answerability import check_answerability, select_usable_evidence
from app.services.rag.evidence.answer_preferences import apply_answer_preferences
from app.services.rag.evidence.composer import compose_grounded_answer
from app.services.rag.evidence.claim_planner import build_claim_plan
from app.services.rag.evidence.citation_validator import validate_grounded_answer
from app.services.rag.evidence.decomposer import decompose_question
from app.services.rag.evidence.evidence_selector import select_evidence_for_answer
from app.services.rag.evidence.evidence_sufficiency import assess_evidence_sufficiency, requires_direct_support
from app.services.rag.evidence.evidence_store import retrieve_evidence
from app.services.rag.evidence.planner import build_evidence_plan
from app.services.rag.evidence.query_frame import is_blocked_frame, parse_query_frame
from app.services.rag.evidence.query_understanding import parse_question_aspects
from app.services.rag.evidence.semantic_query import (
    expand_group_subqueries,
    parse_semantic_query,
    semantic_subquery_to_frame,
)
from app.services.rag.evidence.verification import verify_required_evidence
from app.models.rag_evidence_schema import AtomicQuestion


def answer_with_evidence_kernel(
    question: str,
    top_k: int = 5,
    source_constraints: list[str] | None = None,
) -> GroundedAnswer | None:
    semantic = expand_group_subqueries(parse_semantic_query(question, source_constraints))
    semantic_items = []
    for subquery in semantic.subqueries:
        if subquery.confidence < 0.8:
            continue
        frame = semantic_subquery_to_frame(subquery, source_constraints)
        if frame is not None:
            semantic_items.append((subquery, frame))
    if len(semantic_items) > 1:
        handled = []
        for subquery, frame in semantic_items:
            answer = _answer_single_question(
                subquery.original_text,
                top_k=top_k,
                source_constraints=source_constraints,
                frame=frame,
                semantic_subquery=subquery.model_dump(),
            )
            if answer is not None:
                handled.append(
                    (
                        AtomicQuestion(
                            question_id=subquery.subquery_id,
                            text=_semantic_subquery_label(subquery),
                            original_span=subquery.original_text,
                        ),
                        answer,
                    )
                )
        if handled:
            result = _compose_multi_answer(handled)
            result.debug["semantic_query"] = semantic.model_dump()
            return result
    if len(semantic_items) == 1:
        subquery, frame = semantic_items[0]
        return _answer_single_question(
            question,
            top_k=top_k,
            source_constraints=source_constraints,
            frame=frame,
            semantic_subquery=subquery.model_dump(),
        )

    atomics = decompose_question(question)
    if len(atomics) > 1:
        answers = [
            (atomic, _answer_single_question(atomic.text, top_k=top_k, source_constraints=source_constraints))
            for atomic in atomics
        ]
        handled = [(atomic, answer) for atomic, answer in answers if answer is not None]
        if handled:
            return _compose_multi_answer(handled)
        return None
    return _answer_single_question(question, top_k=top_k, source_constraints=source_constraints)


def _semantic_subquery_label(subquery) -> str:
    if subquery.relation == "has_component_group":
        group = next(
            (item for item in subquery.entities if item.entity_type == "recipe_group"),
            None,
        )
        if group:
            return group.surface
    return subquery.original_text


def _answer_single_question(
    question: str,
    top_k: int = 5,
    source_constraints: list[str] | None = None,
    frame=None,
    semantic_subquery: dict | None = None,
) -> GroundedAnswer | None:
    frame = frame or parse_query_frame(question, source_constraints)
    if not _m2_should_handle(frame):
        return None

    plan = build_evidence_plan(frame)
    evidence = retrieve_evidence(frame, plan, top_k=top_k)
    answerability = check_answerability(frame, plan, evidence)
    usable = select_usable_evidence(frame, answerability, evidence)
    selection = select_evidence_for_answer(frame, plan, answerability, usable or evidence)
    verification = verify_required_evidence(frame, selection.selected)
    retry_debug = None
    if verification.get("rewrite_recommended") and verification.get("rewrite_query"):
        retry_evidence = retrieve_evidence(
            frame.model_copy(update={"original_question": verification["rewrite_query"]}),
            plan,
            top_k=top_k,
        )
        if retry_evidence:
            merged_evidence = _dedupe_evidence([*evidence, *retry_evidence])
            retry_answerability = check_answerability(frame, plan, merged_evidence)
            retry_usable = select_usable_evidence(frame, retry_answerability, merged_evidence)
            retry_selection = select_evidence_for_answer(frame, plan, retry_answerability, retry_usable or merged_evidence)
            retry_verification = verify_required_evidence(frame, retry_selection.selected)
            retry_debug = {
                "rewrite_query": verification["rewrite_query"],
                "retrieved_evidence_count": len(retry_evidence),
                "verification_status": retry_verification.get("status"),
            }
            if retry_verification.get("status") == "sufficient":
                evidence = merged_evidence
                answerability = retry_answerability
                usable = retry_usable
                selection = retry_selection
                verification = retry_verification
    aspects = parse_question_aspects(question, frame)
    sufficiency = assess_evidence_sufficiency(aspects, selection.selected, answerability, frame)
    effective_answerability = answerability
    if (
        requires_direct_support(aspects)
        and answerability.status != "answered"
        and sufficiency.status == "sufficient"
        and sufficiency.direct_evidence_ids
    ):
        effective_answerability = AnswerabilityResult(
            status="answered",
            reason="evidence_sufficiency_direct_support",
            usable_evidence_ids=sufficiency.direct_evidence_ids,
            missing_evidence=[],
            closed_world_negative=False,
        )
    answer = compose_grounded_answer(
        frame,
        effective_answerability,
        selection.selected,
        aspects=aspects,
        sufficiency=sufficiency,
    )
    answer = apply_answer_preferences(answer, semantic_subquery)
    answer.claims = build_claim_plan(answer, frame, sufficiency)
    citation_validation = _validate_citations(answer)
    if citation_validation and not citation_validation["valid"] and answer.answerability.status == "answered":
        answer.answerability = AnswerabilityResult(
            status="partial",
            reason="citation_validation_failed",
            usable_evidence_ids=answer.answerability.usable_evidence_ids,
            missing_evidence=[item["code"] for item in citation_validation["issues"] if item.get("severity") == "error"],
            closed_world_negative=answer.answerability.closed_world_negative,
        )
        answer.uncertainty.append("Citation validation found unresolved evidence binding issues; treat this answer as partial.")
    answer.debug = {
        "query_frame": frame.model_dump(),
        "question_aspects": aspects.model_dump(),
        "evidence_plan": plan.model_dump(),
        "answerability": answer.answerability.model_dump(),
        "raw_answerability": answerability.model_dump(),
        "evidence_sufficiency": sufficiency.model_dump(),
        "retrieved_evidence_count": len(evidence),
        "selected_evidence_count": len(selection.selected),
        "evidence_selection": selection.debug,
        "agentic_verification": verification,
        "agentic_verification_retry": retry_debug,
        "semantic_subquery": semantic_subquery,
        "citation_validation": citation_validation or {"valid": True, "issues": []},
    }
    return answer


def _dedupe_evidence(evidence):
    deduped = []
    seen = set()
    for item in evidence:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        deduped.append(item)
    return deduped


def _compose_multi_answer(handled: list[tuple[object, GroundedAnswer]]) -> GroundedAnswer:
    lines = []
    evidence = []
    uncertainty = []
    claims = []
    statuses = []
    missing = []
    debug_items = []
    for index, (atomic, answer) in enumerate(handled, start=1):
        question_text = getattr(atomic, "text", None) or getattr(atomic, "original_span", "")
        lines.append(f"{index}. {question_text}\n{answer.direct_answer}")
        evidence.extend(answer.evidence)
        uncertainty.extend(answer.uncertainty)
        claims.extend(answer.claims)
        statuses.append(answer.answerability.status)
        missing.extend(answer.answerability.missing_evidence)
        debug_items.append(
            {
                "question_id": getattr(atomic, "question_id", f"q{index}"),
                "text": getattr(atomic, "text", ""),
                "debug": answer.debug,
            }
        )

    if all(status == "answered" for status in statuses):
        status = "answered"
    elif any(status == "blocked" for status in statuses):
        status = "blocked"
    elif any(status in {"answered", "partial"} for status in statuses):
        status = "partial"
    else:
        status = "not_found"

    deduped_uncertainty = []
    for item in uncertainty:
        if item and item not in deduped_uncertainty:
            deduped_uncertainty.append(item)

    return GroundedAnswer(
        answerability=AnswerabilityResult(
            status=status,
            reason="multi_question_aggregated",
            missing_evidence=list(dict.fromkeys(missing)),
        ),
        direct_answer="\n\n".join(lines),
        evidence=evidence,
        uncertainty=deduped_uncertainty,
        citations=[],
        claims=claims,
        debug={
            "multi_question": True,
            "atomic_questions": debug_items,
            "answerability": {
                "status": status,
                "reason": "multi_question_aggregated",
                "missing_evidence": list(dict.fromkeys(missing)),
            },
        },
    )


def _m2_should_handle(frame) -> bool:
    if is_blocked_frame(frame):
        return True
    if frame.evidence_requirement == "structured_schema" and frame.target_attribute in {
        "column",
        "environment_variables",
        "column_equivalence",
        "column_value",
        "component",
        "component_amount",
        "component_group",
    }:
        return True
    if frame.evidence_requirement == "structured_data_analysis":
        return True
    if frame.evidence_requirement == "structured_overview":
        return True
    if frame.target_attribute in {"standard_equivalence", "protocol_material_overview"}:
        return True
    if frame.target_entity in {"manual", "paper"}:
        return True
    if frame.question_type == "suitability":
        return True
    return False


def _validate_citations(answer: GroundedAnswer) -> dict | None:
    if os.getenv("RAG_CITATION_VALIDATION_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    return validate_grounded_answer(answer).model_dump()
