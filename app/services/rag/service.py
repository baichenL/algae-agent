import re

import time
import uuid

from app.core.database import insert_rag_query_log, insert_rag_retrieval_trace_rows, insert_rag_trace_log
from app.models.rag_schema import (
    RagAnswer,
    RagAnswerSegment,
    RagCitation,
    RagEvidence,
    RagQueryRequest,
    RagQueryResponse,
)
from app.services.rag.retrieval.retriever import retrieve_chunks
from app.services.rag.evidence.controlled_composer import enrich_controlled_answer
from app.services.observability.error_events import record_error_event
from app.services.rag.evidence.adapters.chunk_adapter import chunks_to_evidence_units
from app.services.rag.evidence.evidence_sufficiency import assess_evidence_sufficiency, requires_direct_support
from app.services.rag.evidence.kernel import answer_with_evidence_kernel
from app.services.rag.evidence.query_frame import parse_query_frame
from app.services.rag.evidence.query_understanding import parse_question_aspects


BLOCKED_KEYWORDS = [
    "执行传代",
    "批准",
    "审批",
    "删除品系",
    "修改状态",
    "发送邮件",
    "发邮件",
    "重置 generation",
    "更新 algae_status",
    "update algae_status",
    "approve pending",
    "delete strain",
    "send email",
    "send an email",
    "run workflow",
    "execute workflow",
    "execute subculture",
    "修改培养方案",
    "更新培养方案",
    "调整培养方案",
    "直接修改",
    "directly modify",
]

FACT_LAYER_KEYWORDS = [
    "当前藻种状态",
    "当前状态",
    "当前代数",
    "generation_number",
    "pending",
    "提醒状态",
    "reminder_cycle",
]


def answer_rag_question(request: RagQueryRequest) -> RagQueryResponse:
    started_at = time.perf_counter()
    trace_id = f"rag:{uuid.uuid4().hex}"
    question = request.question.strip()
    blocked_reason = _blocked_reason(question)
    if blocked_reason:
        # RAG is the knowledge layer; execution, approval, writes, and email stay in controlled tools/workflows.
        query_log_id = insert_rag_query_log(
            question=question,
            answer=None,
            citations=[],
            uncertainty=[],
            blocked_reason=blocked_reason,
        )
        insert_rag_trace_log(
            query_id=trace_id,
            user_query=question,
            route="rag_blocked",
            answerability={"status": "blocked", "reason": blocked_reason},
            latency_ms=_elapsed_ms(started_at),
            status="blocked",
        )
        return RagQueryResponse(
            status="blocked",
            blocked=True,
            blocked_reason=blocked_reason,
            query_log_id=query_log_id,
            debug={"trace_id": trace_id},
        )

    try:
        m2_response = answer_with_evidence_kernel(
            question,
            top_k=request.top_k,
            source_constraints=request.doc_types or None,
        )
    except Exception as exc:
        record_error_event(
            layer="rag",
            component="rag_service",
            operation="answer_with_evidence_kernel",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"question": question, "top_k": request.top_k, "doc_types": request.doc_types},
        )
        m2_response = None
    if m2_response:
        response = _build_response_from_m2(question, m2_response)
        _record_rag_trace(trace_id, question, "evidence_kernel", response, m2_response.debug or {}, started_at)
        response.debug = {**(response.debug or {}), "trace_id": trace_id}
        return response

    requested_top_k = max(int(request.top_k or 5), 1)
    retrieval_top_k = requested_top_k
    if (
        _is_manual_operation_query(question)
        or _is_recipe_query(question)
        or _is_paper_query(question)
        or _is_experiment_data_query(question)
    ):
        retrieval_top_k = max(requested_top_k * 3, 12)
    try:
        chunks = retrieve_chunks(question, top_k=retrieval_top_k, doc_types=request.doc_types or None)
        chunks = _select_answer_chunks(question, chunks, requested_top_k)
        if _should_reject_low_relevance_chunks(question, chunks):
            chunks = []
        citations = [_build_citation(index + 1, chunk) for index, chunk in enumerate(chunks)]
        evidence = [_build_evidence(index + 1, chunk, question) for index, chunk in enumerate(chunks)]
        uncertainty = _build_uncertainty(question, chunks)
        conclusion = _build_conclusion(question, chunks)
        sufficiency_debug = _assess_fallback_sufficiency(question, chunks, request.doc_types or None)
    except Exception as exc:
        record_error_event(
            layer="rag",
            component="rag_service",
            operation="fallback_retrieval_or_answer_build",
            severity="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"question": question, "top_k": retrieval_top_k, "doc_types": request.doc_types},
        )
        raise
    if sufficiency_debug and sufficiency_debug["should_gate"]:
        conclusion = _build_insufficient_direct_conclusion(sufficiency_debug["sufficiency"])
        evidence, citations = _filter_and_present_background_evidence(
            evidence,
            citations,
            sufficiency_debug,
        )
        uncertainty = [
            *uncertainty,
            "Retrieved chunks were treated as background only because they do not cover every required semantic aspect.",
            "For formal experimental parameters, use lab SOP, recipe documents, or approved project records rather than background papers alone.",
        ]
    answer = RagAnswer(
        conclusion=conclusion,
        evidence=evidence,
        uncertainty=uncertainty,
        debug=sufficiency_debug.get("debug", {}) if sufficiency_debug else {},
    )
    answer.debug = {
        **(answer.debug or {}),
        "retrieval_channels": _retrieval_channels_from_chunks(chunks),
        "retrieved_ids": [chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")],
        "reranked_ids": [chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")],
        "retrieval_hit_trace": [
            chunk.get("retrieval_hit")
            for chunk in chunks
            if chunk.get("retrieval_hit")
        ],
    }
    answer = enrich_controlled_answer(question, answer, citations)
    query_log_id = insert_rag_query_log(
        question=question,
        answer=answer.model_dump_json(),
        citations=[item.model_dump() for item in citations],
        uncertainty=uncertainty,
    )
    response = RagQueryResponse(
        status="success",
        blocked=False,
        answer=answer,
        citations=citations,
        query_log_id=query_log_id,
    )
    _record_rag_trace(trace_id, question, "m1_fallback", response, answer.debug or {}, started_at)
    response.debug = {**(response.debug or {}), "trace_id": trace_id}
    return response


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _retrieval_channels_from_chunks(chunks: list[dict]) -> list[str]:
    channels = {
        channel
        for chunk in chunks
        for channel in (chunk.get("retrieval_channels") or [])
    }
    return sorted(channels)


def _record_rag_trace(
    trace_id: str,
    question: str,
    route: str,
    response: RagQueryResponse,
    debug: dict,
    started_at: float,
) -> None:
    citations = [item.model_dump() for item in response.citations]
    retrieved_ids = debug.get("retrieved_ids") or [
        item.get("chunk_id") or item.get("evidence_id")
        for item in citations
        if item.get("chunk_id") or item.get("evidence_id")
    ]
    selected_ids = debug.get("selected_evidence_ids") or []
    if response.answer:
        selected_ids = selected_ids or [
            f"{item.doc_type}:{item.file_name}:{item.location}"
            for item in response.answer.evidence
        ]
    retrieval_channels = debug.get("retrieval_channels") or []
    answerability = dict(debug.get("answerability") or {})
    if debug.get("citation_validation") is not None:
        answerability["citation_validation"] = debug.get("citation_validation")
    insert_rag_trace_log(
        query_id=trace_id,
        user_query=question,
        route=route,
        query_frame=debug.get("query_frame") or {},
        retrieval_channels=retrieval_channels,
        retrieved_ids=retrieved_ids,
        reranked_ids=debug.get("reranked_ids") or retrieved_ids,
        selected_evidence_ids=selected_ids,
        answerability=answerability,
        sufficiency=debug.get("evidence_sufficiency") or {},
        citations=citations,
        latency_ms=_elapsed_ms(started_at),
        status=response.status,
    )
    _record_retrieval_hit_trace(trace_id, response)


def _record_retrieval_hit_trace(trace_id: str, response: RagQueryResponse) -> None:
    if not response.answer:
        return
    rows = []
    for hit in (response.answer.debug or {}).get("retrieval_hit_trace") or []:
        rows.append(
            {
                "stage": "selected",
                "evidence_id": hit.get("evidence_id"),
                "document_id": hit.get("document_id"),
                "rank": hit.get("final_rank"),
                "sparse_score": hit.get("sparse_score"),
                "dense_score": hit.get("dense_score"),
                "fusion_score": hit.get("fusion_score"),
                "rerank_score": hit.get("rerank_score"),
                "selected": True,
                "rejection_reason": hit.get("rejection_reason"),
                "metadata": hit.get("metadata") or {},
            }
        )
    for item in response.answer.evidence:
        metadata = {}
        evidence_id = None
        if hasattr(item, "model_dump"):
            payload = item.model_dump()
            metadata = payload.get("metadata") or {}
        hit = metadata.get("retrieval_hit")
        if not hit:
            continue
        evidence_id = hit.get("evidence_id")
        rows.append(
            {
                "stage": "selected",
                "evidence_id": evidence_id,
                "document_id": hit.get("document_id"),
                "rank": hit.get("final_rank"),
                "sparse_score": hit.get("sparse_score"),
                "dense_score": hit.get("dense_score"),
                "fusion_score": hit.get("fusion_score"),
                "rerank_score": hit.get("rerank_score"),
                "selected": True,
                "rejection_reason": hit.get("rejection_reason"),
                "metadata": hit.get("metadata") or {},
            }
        )
    if not rows:
        return
    try:
        insert_rag_retrieval_trace_rows(trace_id, rows)
    except Exception as exc:
        record_error_event(
            layer="rag",
            component="rag_service",
            operation="insert_rag_retrieval_trace_rows",
            severity="warning",
            error_type=type(exc).__name__,
            error_message=str(exc),
            metadata={"trace_id": trace_id, "row_count": len(rows)},
        )


def _assess_fallback_sufficiency(
    question: str,
    chunks: list[dict],
    source_constraints: list[str] | None,
) -> dict | None:
    frame = parse_query_frame(question, source_constraints)
    aspects = parse_question_aspects(question, frame)
    units = chunks_to_evidence_units(chunks)
    sufficiency = assess_evidence_sufficiency(aspects, units, answerability=None, frame=frame)
    should_gate = (
        requires_direct_support(aspects)
        and sufficiency.status in {"background_only", "insufficient", "contradicted"}
    )
    return {
        "should_gate": should_gate,
        "sufficiency": sufficiency,
        "debug": {
            "query_frame": frame.model_dump(),
            "question_aspects": aspects.model_dump(),
            "evidence_sufficiency": sufficiency.model_dump(),
        },
    }


def _build_insufficient_direct_conclusion(sufficiency) -> str:
    missing = ", ".join(sufficiency.missing_aspects) or "required aspects"
    if sufficiency.background_evidence_ids:
        return (
            "Current knowledge base has no direct evidence for this conclusion. "
            f"The retrieved chunks are background only and are missing: {missing}."
        )
    return (
        "Current knowledge base has no direct evidence for this conclusion. "
        f"No retrieved evidence covers the required aspects: {missing}."
    )


def _filter_and_present_background_evidence(
    evidence: list[RagEvidence],
    citations: list[RagCitation],
    sufficiency_debug: dict,
) -> tuple[list[RagEvidence], list[RagCitation]]:
    sufficiency = sufficiency_debug["sufficiency"]
    debug = sufficiency_debug.get("debug") or {}
    aspects = debug.get("question_aspects") or {}
    assessments = {
        item.evidence_id: item.model_dump()
        for item in sufficiency.assessments
        if item.evidence_id
    }
    wanted = set(sufficiency.background_evidence_ids or [])
    wanted.update(item.replace("chunk:", "", 1) for item in list(wanted) if item.startswith("chunk:"))
    wanted.update(f"chunk:{item}" for item in list(wanted) if not item.startswith("chunk:"))
    source_to_citation = {item.source_id: item for item in citations}

    filtered_evidence = []
    filtered_source_ids = set()
    for item in evidence:
        citation = source_to_citation.get(item.source_id)
        chunk_id = citation.chunk_id if citation else ""
        evidence_id = f"chunk:{chunk_id}"
        if chunk_id not in wanted and evidence_id not in wanted:
            continue
        assessment = assessments.get(evidence_id) or assessments.get(chunk_id) or {}
        filtered_source_ids.add(item.source_id)
        filtered_evidence.append(
            item.model_copy(
                update={
                    "quote_summary": _format_evidence_presentation_summary(
                        raw_text=item.quote_summary,
                        doc_type=item.doc_type,
                        aspects=aspects,
                        assessment=assessment,
                    )
                }
            )
        )

    filtered_citations = [item for item in citations if item.source_id in filtered_source_ids]
    return filtered_evidence, filtered_citations


def _format_evidence_presentation_summary(
    raw_text: str,
    doc_type: str,
    aspects: dict,
    assessment: dict,
) -> str:
    if assessment and not assessment.get("supports_direct_answer") and assessment.get("supports_background"):
        support = _background_support_phrase(raw_text, doc_type)
        missing = _format_missing_aspects(assessment.get("missing_aspects") or [], aspects)
        return f"背景资料：{support}；缺失内容：{missing}。"
    return _summarize_content(raw_text, max_length=180)


def _background_support_phrase(raw_text: str, doc_type: str) -> str:
    text = raw_text or ""
    lower = text.lower()
    if "tap" in lower and ("sucrose" in lower or "??" in text):
        if "recovery" in lower or "??" in text:
            return "????? 40 mM ?? TAP ????????????"
        return "????? 40 mM ?? TAP ???"
    if "tap" in lower and ("hygromycin" in lower or "???" in text):
        return "???????? TAP ??????????"
    if "tap" in lower and doc_type == "media_recipe":
        return "????? TAP ???????????"
    if "tap" in lower:
        return "????? TAP ???? TAP ??????"
    return _summarize_content(text, max_length=90)


def _format_missing_aspects(missing_aspects: list[str], aspects: dict) -> str:
    if not missing_aspects:
        return "?????????????????"
    labels = []
    for item in missing_aspects:
        labels.append(_aspect_label(item, aspects))
    return "?".join(label for label in labels if label)


def _aspect_label(aspect: str, aspects: dict) -> str:
    if aspect == "subject":
        return f"???{aspects.get('subject') or '????'}"
    if aspect == "condition":
        condition = aspects.get("condition") or "????"
        if condition == "dark culture":
            condition = "????"
        return f"???{condition}"
    if aspect == "relation":
        relation = aspects.get("relation") or "????"
        labels = {
            "suitability": "????",
            "support": "????",
            "migration": "??/????",
            "relationship": "?????",
        }
        return labels.get(relation, relation)
    if aspect == "target":
        return f"???{aspects.get('target') or '??'}"
    return aspect


def _build_response_from_m2(question: str, grounded_answer) -> RagQueryResponse:
    if grounded_answer.answerability.status == "blocked":
        query_log_id = insert_rag_query_log(
            question=question,
            answer=None,
            citations=[],
            uncertainty=grounded_answer.uncertainty,
            blocked_reason=grounded_answer.answerability.reason,
        )
        return RagQueryResponse(
            status="blocked",
            blocked=True,
            blocked_reason=grounded_answer.answerability.reason,
            query_log_id=query_log_id,
        )

    debug = grounded_answer.debug or {}
    debug["retrieval_channels"] = sorted(
        {
            channel
            for item in grounded_answer.evidence
            for channel in ((item.metadata or {}).get("retrieval_channels") or [])
        }
    )
    debug["retrieved_ids"] = [item.evidence_id for item in grounded_answer.evidence]
    debug["selected_evidence_ids"] = [item.evidence_id for item in grounded_answer.evidence]
    debug["retrieval_hit_trace"] = [
        (item.metadata or {}).get("retrieval_hit")
        for item in grounded_answer.evidence
        if (item.metadata or {}).get("retrieval_hit")
    ]
    aspects = debug.get("question_aspects") or {}
    sufficiency = debug.get("evidence_sufficiency") or {}
    assessments = {
        item.get("evidence_id"): item
        for item in (sufficiency.get("assessments") or [])
        if item.get("evidence_id")
    }

    evidence = []
    citations = []
    evidence_to_source_id = {}
    evidence_to_summary = {}
    citation_key_to_source_id = {}
    for item in grounded_answer.evidence:
        location = _format_m2_location(item.location)
        assessment = assessments.get(item.evidence_id) or {}
        citation_payload = item.citation or {}
        citation_key = (
            item.source_file,
            citation_payload.get("source_path") or item.source_file,
            item.source_type,
            item.location.get("section"),
            item.location.get("page_number"),
            item.location.get("sheet_name"),
            item.location.get("row_start"),
            item.location.get("row_end"),
        )
        source_id = citation_key_to_source_id.get(citation_key)
        if source_id is None:
            source_id = len(citations) + 1
            citation_key_to_source_id[citation_key] = source_id
            citations.append(
                RagCitation(
                    source_id=source_id,
                    chunk_id=citation_payload.get("chunk_id") or item.evidence_id,
                    file_name=item.source_file,
                    source_path=citation_payload.get("source_path") or item.source_file,
                    doc_type=item.source_type,
                    title=item.entity,
                    section=item.location.get("section"),
                    page_number=item.location.get("page_number"),
                    sheet_name=item.location.get("sheet_name"),
                    row_start=item.location.get("row_start"),
                    row_end=item.location.get("row_end"),
                    topic=item.metadata.get("inferred_role") or item.metadata.get("topic"),
                    version=item.metadata.get("version") or citation_payload.get("version"),
                    year=item.metadata.get("year") or citation_payload.get("year"),
                    language=item.metadata.get("language") or citation_payload.get("language"),
                    evidence_type=item.evidence_type or item.fact_type,
                    confidence=item.confidence,
                    extraction_method=item.extraction_method or item.metadata.get("extraction_method"),
                    organism=item.metadata.get("organism"),
                    medium=item.metadata.get("medium"),
                    task_type=item.metadata.get("task_type"),
                    equipment=item.metadata.get("equipment"),
                    measurement=item.metadata.get("measurement"),
                )
            )
        evidence_to_source_id[item.evidence_id] = source_id
        quote_summary = _format_evidence_presentation_summary(
            raw_text=item.text_span or item.value or item.attribute or item.fact_type,
            doc_type=item.source_type,
            aspects=aspects,
            assessment=assessment,
        )
        evidence_to_summary[item.evidence_id] = quote_summary
        evidence.append(
            RagEvidence(
                source_id=source_id,
                file_name=item.source_file,
                doc_type=item.source_type,
                location=location,
                quote_summary=quote_summary,
            )
        )

    uncertainty = list(grounded_answer.uncertainty)
    answer = RagAnswer(
        conclusion=grounded_answer.direct_answer,
        evidence=evidence,
        uncertainty=uncertainty,
        debug=debug,
    )
    for claim in grounded_answer.claims:
        citation_ids = [
            evidence_to_source_id[evidence_id]
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_to_source_id
        ]
        segment = RagAnswerSegment(text=claim.text, citation_ids=citation_ids)
        if claim.claim_type == "fact":
            answer.facts.append(segment)
        elif claim.claim_type == "background":
            background_text = next(
                (
                    evidence_to_summary[evidence_id]
                    for evidence_id in claim.evidence_ids
                    if evidence_id in evidence_to_summary
                ),
                claim.text,
            )
            if background_text.startswith("?????"):
                background_text = background_text.removeprefix("?????")
            answer.facts.append(
                RagAnswerSegment(text=f"背景资料：{background_text}", citation_ids=citation_ids)
            )
        elif claim.claim_type == "suggestion":
            answer.suggestions.append(segment)
        elif claim.claim_type == "explanation":
            answer.explanations.append(segment)
        elif claim.claim_type == "uncertainty" and claim.text not in answer.uncertainty:
            answer.uncertainty.append(claim.text)
    answer = enrich_controlled_answer(question, answer, citations)
    query_log_id = insert_rag_query_log(
        question=question,
        answer=answer.model_dump_json(),
        citations=[item.model_dump() for item in citations],
        uncertainty=uncertainty,
    )
    return RagQueryResponse(
        status="success",
        blocked=False,
        answer=answer,
        citations=citations,
        query_log_id=query_log_id,
        debug=debug,
    )


def _format_query_frame_debug(debug: dict) -> str | None:
    if not debug:
        return None
    if debug.get("multi_question"):
        statuses = []
        for item in debug.get("atomic_questions") or []:
            frame = ((item.get("debug") or {}).get("query_frame") or {})
            answerability = ((item.get("debug") or {}).get("answerability") or {})
            if frame:
                statuses.append(
                    f"{item.get('question_id')}: {frame.get('question_type')}/{frame.get('target_attribute')} -> "
                    f"{answerability.get('status')}:{answerability.get('reason')}"
                )
        return "QueryFrame: " + "；".join(statuses) if statuses else None
    frame = debug.get("query_frame") or {}
    answerability = debug.get("answerability") or {}
    if not frame:
        return None
    return (
        "QueryFrame: "
        f"type={frame.get('question_type')}, target={frame.get('target_entity')}.{frame.get('target_attribute')}, "
        f"value={frame.get('target_value')}, evidence={frame.get('evidence_requirement')}; "
        f"answerability={answerability.get('status')}:{answerability.get('reason')}"
    )


def _format_m2_location(location: dict) -> str:
    if location.get("page_number"):
        return f"page {location['page_number']}"
    if location.get("sheet_name"):
        if location.get("column_index"):
            return f"sheet {location['sheet_name']}, column {location['column_index']}"
        if location.get("row_start") and location.get("row_end"):
            return f"sheet {location['sheet_name']}, rows {location['row_start']}-{location['row_end']}"
        return f"sheet {location['sheet_name']}"
    if location.get("section"):
        return str(location["section"])
    if location.get("column_index"):
        return f"column {location['column_index']}"
    return "evidence"


def _blocked_reason(question: str) -> str | None:
    normalized = question.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword.lower() in normalized:
            return "rag_read_only_boundary"
    return None


def _build_citation(source_id: int, chunk: dict) -> RagCitation:
    metadata = chunk.get("metadata") or {}
    return RagCitation(
        source_id=source_id,
        chunk_id=chunk["chunk_id"],
        file_name=chunk["file_name"],
        source_path=chunk["source_path"],
        doc_type=chunk["doc_type"],
        title=chunk.get("title"),
        section=chunk.get("section"),
        page_number=chunk.get("page_number"),
        sheet_name=chunk.get("sheet_name"),
        row_start=chunk.get("row_start"),
        row_end=chunk.get("row_end"),
        topic=chunk.get("topic"),
        version=chunk.get("version") or metadata.get("version"),
        year=chunk.get("year") or metadata.get("year"),
        language=chunk.get("language") or metadata.get("language"),
        evidence_type="text_chunk",
        confidence=chunk.get("hybrid_score"),
        extraction_method="chunk_retrieval",
        organism=metadata.get("organism"),
        medium=metadata.get("medium"),
        task_type=metadata.get("task_type"),
        equipment=metadata.get("equipment"),
        measurement=metadata.get("measurement"),
    )


def _build_evidence(source_id: int, chunk: dict, question: str = "") -> RagEvidence:
    if _is_manual_operation_query(question):
        quote_summary = _summarize_manual_operation_chunk(chunk)
    elif _is_recipe_query(question):
        quote_summary = _summarize_recipe_chunk(chunk)
    elif chunk.get("doc_type") == "paper":
        quote_summary = _summarize_paper_chunk(chunk)
    elif chunk.get("doc_type") == "experiment_data":
        quote_summary = _summarize_experiment_chunk(chunk)
    else:
        quote_summary = _summarize_content(chunk["content"])
    return RagEvidence(
        source_id=source_id,
        file_name=chunk["file_name"],
        doc_type=chunk["doc_type"],
        location=_format_location(chunk),
        quote_summary=quote_summary,
    )


def _build_conclusion(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "当前知识库没有检索到与该问题直接相关的可引用资料。"

    if _is_manual_operation_query(question):
        manual_summary = _build_manual_operation_conclusion(question, chunks)
        if manual_summary:
            return manual_summary

    if _is_tap_applicability_query(question):
        return _build_tap_applicability_conclusion(chunks)

    if _is_tap_sterilization_query(question):
        return _build_tap_sterilization_conclusion(question, chunks)

    recipe_conclusion = _build_recipe_conclusion(question, chunks)
    if recipe_conclusion:
        return recipe_conclusion

    paper_conclusion = _build_paper_conclusion(question, chunks)
    if paper_conclusion:
        return paper_conclusion

    experiment_conclusion = _build_experiment_data_conclusion(chunks)
    if experiment_conclusion:
        return experiment_conclusion

    summaries = []
    seen = set()
    for chunk in chunks[:4]:
        summary = _summarize_content(chunk["content"], max_length=160)
        if summary and summary not in seen:
            summaries.append(summary)
            seen.add(summary)
    summary_text = "?".join(summaries)
    if _mentions_fact_layer(question):
        return f"??????????????????????????pending ??????? SQLite ????????????????{summary_text}"
    return f"??????????????????{summary_text}"


def _select_answer_chunks(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    if _is_manual_operation_query(question):
        manual_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "manual"]
        if re.search(r"tap", question or "", re.I):
            tap_chunks = [
                chunk for chunk in manual_chunks if "tap" in (chunk.get("content") or "").lower()
            ]
            if tap_chunks:
                manual_chunks = tap_chunks
        operation_chunks = [
            chunk for chunk in manual_chunks if _manual_operation_labels(chunk.get("content") or "")
        ]
        if operation_chunks:
            manual_chunks = operation_chunks
        focused_manual_chunks = _filter_manual_chunks_by_focus(question, manual_chunks)
        if focused_manual_chunks:
            manual_chunks = focused_manual_chunks
        if manual_chunks:
            manual_chunks.sort(key=_manual_chunk_order)
            return manual_chunks[: max(int(top_k or 5), 1)]
    if not _is_recipe_query(question):
        if _is_experiment_data_query(question):
            experiment_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "experiment_data"]
            return experiment_chunks[: max(int(top_k or 5), 1)] if experiment_chunks else chunks
        if _is_paper_query(question):
            paper_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "paper"]
            return _select_diverse_paper_chunks(paper_chunks or chunks, top_k)
        return chunks
    recipe_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "media_recipe"]
    table_chunks = [
        chunk for chunk in recipe_chunks if "table" in str(chunk.get("section") or "").lower()
    ]
    if len(table_chunks) >= 3:
        focused_recipe_chunks = _filter_recipe_chunks_by_focus(question, table_chunks)
        if focused_recipe_chunks:
            focused_recipe_chunks.sort(key=_recipe_chunk_order)
            return focused_recipe_chunks[: max(int(top_k or 5), 1)]
        table_chunks.sort(key=_recipe_chunk_order)
        return table_chunks[: max(int(top_k or 5), 1)]
    if len(recipe_chunks) >= 3:
        recipe_chunks.sort(key=_recipe_chunk_order)
        return recipe_chunks[: max(int(top_k or 5), 1)]
    return chunks


def _should_reject_low_relevance_chunks(question: str, chunks: list[dict]) -> bool:
    if not chunks:
        return False
    if (
        _is_manual_operation_query(question)
        or _is_recipe_query(question)
        or _is_paper_query(question)
        or _is_experiment_data_query(question)
        or _is_tap_applicability_query(question)
        or _is_tap_sterilization_query(question)
    ):
        return False

    evidence_text = " ".join(
        str(part or "")
        for chunk in chunks[:5]
        for part in [
            chunk.get("file_name"),
            chunk.get("title"),
            chunk.get("topic"),
            chunk.get("content"),
            chunk.get("section"),
            (chunk.get("metadata") or {}).get("topic"),
        ]
    ).casefold()
    required_terms = _required_topic_terms(question)
    if required_terms:
        return not all(term.casefold() in evidence_text for term in required_terms)
    terms = _question_topic_terms(question)
    if not terms:
        return False
    return not any(term.casefold() in evidence_text for term in terms)


def _required_topic_terms(question: str) -> list[str]:
    text = question or ""
    required = []
    for term in ["浑河", "沈阳", "Hun River", "Shenyang"]:
        if term.lower() in text.lower():
            required.append(term)
    return required


def _question_topic_terms(question: str) -> list[str]:
    text = question or ""
    terms = []
    domain_terms = [
        "浑河",
        "沈阳",
        "微藻",
        "藻类",
        "淡水",
        "河流",
        "水体",
        "绿藻",
        "蓝藻",
        "硅藻",
        "microalgae",
        "algae",
        "freshwater",
        "river",
    ]
    for term in domain_terms:
        if re.search(re.escape(term), text, re.I):
            terms.append(term)
    for term in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text):
        if term.lower() not in {"what", "which", "list", "common", "about"}:
            terms.append(term)
    return list(dict.fromkeys(terms))


def _is_recipe_query(question: str) -> bool:
    if _is_manual_operation_query(question):
        return False
    if _is_tap_applicability_query(question) or _is_tap_sterilization_query(question):
        return False
    text = question or ""
    has_medium = bool(re.search(r"tap|鍩瑰吇鍩簗medium", text, re.I))
    asks_recipe = bool(
        re.search(r"閰嶆柟|缁勫垎|鎴愬垎|缁勬垚|鍖呭惈|鍚湁鍝簺|鍖呮嫭鍝簺|recipe|composition|component", text, re.I)
    )
    return asks_recipe and (has_medium or bool(re.search(r"閰嶆柟|recipe", text, re.I)))


def _is_tap_applicability_query(question: str) -> bool:
    text = question or ""
    return bool(re.search(r"tap|鍩瑰吇鍩簗medium", text, re.I)) and bool(
        re.search(r"閫傜敤浜巪閫傚悎|鎵€鏈墊鍏ㄩ儴|all|any|Chlorella|灏忕悆钘粅鍝佺郴", text, re.I)
    )


def _is_tap_sterilization_query(question: str) -> bool:
    text = question or ""
    return bool(re.search(r"tap|鍩瑰吇鍩簗medium", text, re.I)) and bool(
        re.search(r"鐏弻|楂樺帇|121|20\s*鍒嗛挓|autoclave|sterili", text, re.I)
    )


def _is_paper_query(question: str) -> bool:
    return bool(
        re.search(
            r"璁烘枃|paper|鏂囩尞|literature|machine learning|deep learning|data-driven|growth prediction|forecasting",
            question or "",
            re.I,
        )
    )


def _is_experiment_data_query(question: str) -> bool:
    return bool(
        re.search(
            r"实验数据|表格|xlsx|OD750|growth curve|生长曲线|biomass|生物量|数据",
            question or "",
            re.I,
        )
    )


def _is_manual_operation_query(question: str) -> bool:
    return bool(
        re.search(
            r"瀹為獙鎵嬪唽|鎵嬪唽|manual|sop|protocol|鎿嶄綔|姝ラ|娴佺▼|鎬庝箞|濡備綍|鐢靛嚮|杞寲|骞虫澘|澶嶈嫃",
            question or "",
            re.I,
        )
    )


def _recipe_focus(question: str) -> str:
    text = question or ""
    if re.search(r"磷酸盐|phosphate|K2HPO4|KH2PO4|K2HPO4|KH2PO4", text, re.I):
        return "phosphate"
    if re.search(r"Hutner|寰噺|trace|EDTA|ZnSO4|FeSO4|閲戝睘", text, re.I):
        return "trace"
    if re.search(r"鐩愭憾娑瞸姘簮|NH4Cl|MgSO4|CaCl2|salt", text, re.I):
        return "salts"
    if re.search(r"宸ヤ綔娑瞸鍐颁箼閰竱Tris|姣嶆恫.*鍔犲叆|1\s*L", text, re.I):
        return "working"
    return "all"


def _build_recipe_conclusion(question: str, chunks: list[dict]) -> str | None:
    recipe_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "media_recipe"]
    if not recipe_chunks:
        return None

    table_by_section = {str(chunk.get("section") or ""): chunk for chunk in recipe_chunks}
    working = _parse_pipe_pairs(_content_for_section(table_by_section, "table 4"))
    salts = _parse_pipe_pairs(_content_for_section(table_by_section, "table 1"))
    phosphate = _parse_pipe_pairs(_content_for_section(table_by_section, "table 2"))
    trace = _parse_trace_table(_content_for_section(table_by_section, "table 3"))

    focus = _recipe_focus(question)
    if focus == "phosphate":
        if not phosphate:
            return "根据当前 TAP 配方文档，未找到磷酸盐溶液的结构化表格。"
        lines = ["根据当前 TAP 配方文档，TAP 培养基的磷酸盐溶液（母液2）包含："]
        lines.extend(_format_pair_bullets(phosphate, suffix="（母液用量）"))
        lines.append("工作液中加入磷酸盐溶液（母液2）1 mL/L。")
        return "\n".join(lines)

    if focus == "trace":
        if not trace:
            return "根据当前 TAP 配方文档，未找到 Hutner's 微量元素的结构化表格。"
        lines = ["根据当前 TAP 配方文档，Hutner's 微量元素（母液3，工作液加入 1 mL/L）包含："]
        lines.extend(_format_pair_bullets(trace, suffix="（母液用量）"))
        return "\n".join(lines)

    if focus == "salts":
        if not salts:
            return "根据当前 TAP 配方文档，未找到盐溶液的结构化表格。"
        lines = ["根据当前 TAP 配方文档，盐溶液（母液1，工作液加入 10 mL/L）包含："]
        lines.extend(_format_pair_bullets(salts, suffix="（母液用量）"))
        return "\n".join(lines)

    if focus == "working":
        if not working:
            return "根据当前 TAP 配方文档，未找到工作液配制表格。"
        lines = ["根据当前 TAP 配方文档，1 L TAP 工作液配制包括："]
        lines.extend(_format_pair_bullets(working))
        return "\n".join(lines)

    lines = [
        "TAP（Tris-Acetate-Phosphate）培养基在当前知识库中作为莱茵衣藻相关培养基记录。",
        "根据当前 TAP 配方文档，其核心组分可整理如下：",
    ]
    if working:
        lines.append("1. 工作液配制")
        lines.extend(_format_pair_bullets(working))
    if salts:
        lines.append("2. 盐溶液（母液1，工作液加入 10 mL/L）")
        lines.extend(_format_pair_bullets(salts, suffix="（母液用量）"))
    if phosphate:
        lines.append("3. 磷酸盐溶液（母液2，工作液加入 1 mL/L）")
        lines.extend(_format_pair_bullets(phosphate, suffix="（母液用量）"))
    if trace:
        lines.append("4. Hutner's 微量元素（母液3，工作液加入 1 mL/L）")
        lines.extend(_format_pair_bullets(trace, suffix="（母液用量）"))

    lines.append("配制要点：")
    lines.append("- 工作液最终定容至 1 L。")
    lines.append("- 母液浓度和最终工作液浓度不要混为一谈；上述用量按来源表格原文呈现。")
    return "\n".join(lines)


def _build_tap_applicability_conclusion(chunks: list[dict]) -> str:
    doc_types = {chunk.get("doc_type") for chunk in chunks}
    if "media_recipe" in doc_types or "manual" in doc_types:
        return (
            "不能确认 TAP 培养基适用于所有 Chlorella 品系。\n"
            "当前知识库中的 TAP 依据主要指向莱茵衣藻相关配方或实验手册操作；这些来源不能自动外推到所有小球藻或所有 Chlorella 品系。\n"
            "正式实验中应以对应品系的实验室 SOP、历史培养记录或审批后的培养方案为准。"
        )
    return (
        "根据当前知识库，未找到足够依据判断 TAP 培养基是否适用于所有 Chlorella 品系。"
    )


def _build_tap_sterilization_conclusion(question: str, chunks: list[dict]) -> str:
    sterilization_hits = []
    asks_specific_condition = bool(re.search(r"121|20\s*min|20\s*鍒嗛挓", question or "", re.I))
    for chunk in chunks:
        content = chunk.get("content") or ""
        if re.search(r"121|20\s*min|20\s*分钟|autoclave|sterili", content, re.I):
            sterilization_hits.append(chunk)
    if sterilization_hits and asks_specific_condition:
        summary = "；".join(_summarize_content(chunk["content"], max_length=120) for chunk in sterilization_hits[:3])
        return f"当前知识库中检索到与灭菌相关的依据：{summary}"
    return (
        "根据当前命中的 TAP 配方和手册内容，不能确认该配方明确要求 121°C 灭菌 20 分钟。\n"
        "当前知识库可以确认 TAP 的组分和部分相关操作，但未给出足够可引用依据支持这个具体灭菌条件。"
    )


def _build_paper_conclusion(question: str, chunks: list[dict]) -> str | None:
    paper_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "paper"]
    if not paper_chunks:
        return None

    papers = []
    seen = set()
    for chunk in paper_chunks:
        file_name = chunk.get("file_name") or ""
        if file_name in seen:
            continue
        seen.add(file_name)
        papers.append((file_name, _paper_display_title(chunk), _paper_topic_tags(chunk)))

    lines = ["根据当前本地论文库，命中以下与问题相关的论文："]
    for index, (file_name, title, tags) in enumerate(papers[:5], start=1):
        tag_text = f"；相关主题：{', '.join(tags)}" if tags else ""
        lines.append(f"{index}. {title}{tag_text}。")
    if re.search(r"machine learning|机器学习|data-driven|deep learning", question or "", re.I):
        lines.append("总体判断：这些来源与 microalgae cultivation 的机器学习、数据驱动建模、预测或优化方法相关。")
    lines.append("注意：以上是基于本地论文库文件名、元数据和命中页面的归纳，不等同于完整系统综述。")
    return "\n".join(lines)


def _build_experiment_data_conclusion(chunks: list[dict]) -> str | None:
    data_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "experiment_data"]
    if not data_chunks:
        return None

    sheets = []
    seen = set()
    has_biomass = False
    has_od750 = False
    for chunk in data_chunks:
        location = _format_location(chunk)
        if location not in seen:
            sheets.append(location)
            seen.add(location)
        content = chunk.get("content") or ""
        has_biomass = has_biomass or "biomass" in content.lower()
        has_od750 = has_od750 or "od750" in content.lower()

    lines = ["根据当前本地实验数据，已检索到微藻生长曲线相关表格记录。"]
    if has_biomass:
        lines.append("表格字段包含 biomass g/L，可用于生物量随时间变化的生长曲线分析。")
    if not has_od750:
        lines.append("当前命中的表格中未看到 OD750 字段；因此不能把这些数据直接说成 OD750 记录。")
    lines.append(f"可引用位置：{'; '.join(sheets[:5])}。")
    return "\n".join(lines)


def _build_manual_operation_conclusion(question: str, chunks: list[dict]) -> str | None:
    manual_chunks = [chunk for chunk in chunks if chunk.get("doc_type") == "manual"]
    if not manual_chunks:
        return None

    operations: list[tuple[str, str]] = []
    for chunk in manual_chunks:
        page = f"page {chunk.get('page_number')}" if chunk.get("page_number") else _format_location(chunk)
        for label in _manual_operation_labels(chunk.get("content") or ""):
            operations.append((label, page))

    deduped = []
    seen = set()
    for operation, page in operations:
        key = (operation, page)
        if key not in seen:
            deduped.append(key)
            seen.add(key)

    if not deduped:
        summary = "；".join(_summarize_content(chunk["content"], max_length=140) for chunk in manual_chunks[:3])
        return f"根据实验手册，检索到与 TAP 培养基相关的内容，但需要人工复核具体操作边界：{summary}"

    focus = _manual_focus(question)
    if focus == "hygromycin":
        selected = [(operation, page) for operation, page in deduped if "潮霉素" in operation]
        if selected:
            lines = ["有。实验手册中提到潮霉素 TAP 平板，主要用于电击转化后的筛选："]
            for index, (operation, page) in enumerate(selected[:3], start=1):
                lines.append(f"{index}. {operation}（{page}）")
            return "\n".join(lines)

    if focus == "electroporation":
        selected = [
            (operation, page)
            for operation, page in deduped
            if any(keyword in operation for keyword in ["电击", "OD750", "蔗糖 TAP", "潮霉素", "恢复培养", "涂板"])
        ]
        if selected:
            lines = ["根据实验手册，TAP 培养基在电击转化相关操作中主要这样使用："]
            for index, (operation, page) in enumerate(selected[:6], start=1):
                lines.append(f"{index}. {operation}（{page}）")
            return "\n".join(lines)

    lines = ["根据实验手册，与 TAP 培养基相关的操作主要包括："]
    for index, (operation, page) in enumerate(deduped[:6], start=1):
        lines.append(f"{index}. {operation}（{page}）")
    return "\n".join(lines)


def _manual_focus(question: str) -> str:
    text = question or ""
    if re.search(r"潮霉素|hygromycin", text, re.I):
        return "hygromycin"
    if re.search(r"电击|转化|electroporation|OD750|蔗糖 TAP", text, re.I):
        return "electroporation"
    return "general"


def _filter_manual_chunks_by_focus(question: str, chunks: list[dict]) -> list[dict]:
    focus = _manual_focus(question)
    if focus == "hygromycin":
        return [
            chunk
            for chunk in chunks
            if "潮霉素 TAP 平板" in (chunk.get("content") or "")
            or "hygromycin tap" in (chunk.get("content") or "").lower()
        ]
    if focus == "electroporation":
        return [
            chunk
            for chunk in chunks
            if re.search(r"电击|转化|OD750|蔗糖 TAP|恢复培养|涂板", chunk.get("content") or "", re.I)
        ]
    return []


def _filter_recipe_chunks_by_focus(question: str, chunks: list[dict]) -> list[dict]:
    focus = _recipe_focus(question)
    section_map = {
        "phosphate": ["table 4", "table 2"],
        "trace": ["table 4", "table 3"],
        "salts": ["table 4", "table 1"],
        "working": ["table 4"],
    }
    wanted = section_map.get(focus)
    if not wanted:
        return []
    filtered = []
    for chunk in chunks:
        section = str(chunk.get("section") or "").lower()
        if any(item in section for item in wanted):
            filtered.append(chunk)
    return filtered


def _content_for_section(chunks_by_section: dict[str, dict], section_keyword: str) -> str:
    for section, chunk in chunks_by_section.items():
        if section_keyword in section:
            return chunk.get("content") or ""
    return ""


def _parse_pipe_pairs(content: str) -> list[tuple[str, str]]:
    rows = _table_rows(content)
    unit_suffix = ""
    if rows and len(rows[0]) >= 2 and "/g" in rows[0][1]:
        unit_suffix = " g"
    pairs = []
    for row in rows:
        if not row or row[0] == "缁勫垎":
            continue
        if len(row) >= 2 and row[0] and row[1]:
            value = _append_unit(row[1], unit_suffix)
            pairs.append((row[0], value))
    return pairs


def _parse_trace_table(content: str) -> list[tuple[str, str]]:
    triples = []
    for row in _table_rows(content):
        if not row or row[0] == "缁勫垎":
            continue
        if len(row) < 2:
            continue
        name = row[0]
        amount = row[1]
        water = row[2] if len(row) >= 3 else ""
        if name.startswith("瀹氬"):
            continue
        if name and amount:
            value = f"{amount} g"
            if water:
                value += f"; H2O {water} ml"
            triples.append((name, value))
    return triples


def _table_rows(content: str) -> list[list[str]]:
    rows = []
    for line in (content or "").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def _format_pairs(pairs: list[tuple[str, str]]) -> str:
    return "；".join(f"{_prettify_chemical_name(name)}: {_prettify_chemical_name(value)}" for name, value in pairs)


def _format_pair_bullets(pairs: list[tuple[str, str]], suffix: str = "") -> list[str]:
    return [
        f"- {_prettify_chemical_name(name)}：{_prettify_chemical_name(value)}{suffix}"
        for name, value in pairs
    ]


def _summarize_recipe_chunk(chunk: dict) -> str:
    section = str(chunk.get("section") or "")
    if "table 4" in section:
        pairs = _parse_pipe_pairs(chunk.get("content") or "")
        return "宸ヤ綔娑查厤鍒讹細" + _format_pairs(pairs) if pairs else _summarize_content(chunk["content"])
    if "table 1" in section:
        pairs = _parse_pipe_pairs(chunk.get("content") or "")
        return "姣嶆恫1锛堢洂婧舵恫锛夛細" + _format_pairs(pairs) if pairs else _summarize_content(chunk["content"])
    if "table 2" in section:
        pairs = _parse_pipe_pairs(chunk.get("content") or "")
        return "姣嶆恫2锛堢７閰哥洂婧舵恫锛夛細" + _format_pairs(pairs) if pairs else _summarize_content(chunk["content"])
    if "table 3" in section:
        pairs = _parse_trace_table(chunk.get("content") or "")
        return "姣嶆恫3锛圚utner's 寰噺鍏冪礌锛夛細" + _format_pairs(pairs) if pairs else _summarize_content(chunk["content"])
    return _summarize_content(chunk["content"])


def _summarize_manual_operation_chunk(chunk: dict) -> str:
    content = chunk.get("content") or ""
    if chunk.get("doc_type") != "manual":
        return _summarize_content(content)
    summary_items = _manual_operation_labels(content, evidence_style=True)
    if summary_items:
        return "；".join(summary_items)
    return _summarize_content(content)


def _summarize_paper_chunk(chunk: dict) -> str:
    title = _paper_display_title(chunk)
    tags = _paper_topic_tags(chunk)
    if tags:
        return f"{title}锛涘懡涓富棰橈細{', '.join(tags)}"
    return title


def _summarize_experiment_chunk(chunk: dict) -> str:
    content = chunk.get("content") or ""
    fields = []
    for field in ["时间 h", "Iμmol·m-2·s-1", "N mg/L", "P mg/L", "tf", "biomass g/L", "OD750"]:
        if field.lower() in content.lower():
            fields.append(field)
    location = _format_location(chunk)
    if fields:
        return f"{location}；字段包含：{', '.join(fields)}"
    return f"{location}；{_summarize_content(content, max_length=120)}"


def _paper_display_title(chunk: dict) -> str:
    topic = (chunk.get("topic") or (chunk.get("metadata") or {}).get("topic") or "").strip()
    file_name = chunk.get("file_name") or topic or "paper"
    stem = file_name.rsplit(".", 1)[0]
    if stem.startswith("paper__"):
        parts = stem.split("__")
        if len(parts) >= 2:
            topic = parts[1]
            year = parts[2] if len(parts) >= 3 and parts[2].isdigit() else None
            title = topic.replace("_", " ")
            return f"{title} ({year})" if year else title
    return (topic or stem).replace("_", " ")


def _paper_topic_tags(chunk: dict) -> list[str]:
    text = " ".join(
        str(item or "")
        for item in [
            chunk.get("file_name"),
            chunk.get("title"),
            chunk.get("topic"),
            (chunk.get("metadata") or {}).get("topic"),
            chunk.get("content"),
        ]
    ).lower()
    tags = []
    tag_patterns = [
        ("machine learning", r"machine learning|\bml\b"),
        ("deep learning", r"deep learning|\bdl\b|neural network|ann"),
        ("data-driven modeling", r"data-driven|data driven|black box"),
        ("growth prediction", r"growth prediction|growth curve|forecast"),
        ("cultivation optimization", r"optimization|environmental factors|cultivation"),
        ("photobioreactor / biorefinery", r"photobioreactor|biorefinery|biofuel"),
    ]
    for label, pattern in tag_patterns:
        if re.search(pattern, text, re.I):
            tags.append(label)
    return tags[:4]


def _select_diverse_paper_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    selected = []
    seen_files = set()
    remaining = []
    for chunk in chunks:
        file_name = chunk.get("file_name")
        if file_name and file_name not in seen_files:
            selected.append(chunk)
            seen_files.add(file_name)
        else:
            remaining.append(chunk)
        if len(selected) >= max(int(top_k or 5), 1):
            return selected
    for chunk in remaining:
        if len(selected) >= max(int(top_k or 5), 1):
            break
        selected.append(chunk)
    return selected


def _manual_operation_labels(content: str, evidence_style: bool = False) -> list[str]:
    lowered = content.lower()
    labels = []
    if "1.2.4 tap" in lowered or "tap 液体培养基" in content:
        labels.append("TAP 液体培养基配制")
    if "潮霉素 tap 平板" in lowered or "潮霉素 TAP 平板" in content:
        labels.append(
            "潮霉素 TAP 平板配制" if evidence_style else "潮霉素 TAP 平板配制，用于转化后筛选"
        )
    if "蔗糖 tap 培养基" in lowered or "蔗糖 TAP 培养基" in content:
        labels.append(
            "40 mM 蔗糖 TAP 培养基配制"
            if evidence_style
            else "40 mM 蔗糖 TAP 培养基配制，用于电击转化相关操作"
        )
    if "OD750=0.3-0.5" in content:
        labels.append(
            "莱茵衣藻培养至 OD750=0.3-0.5"
            if evidence_style
            else "将莱茵衣藻培养至 OD750=0.3-0.5 后用于电击转化材料准备"
        )
    if "蔗糖 TAP 溶液重悬" in content or "40 mM 蔗糖 TAP 溶液" in content:
        labels.append(
            "蔗糖 TAP 溶液用于重悬、恢复培养或涂板"
            if evidence_style
            else "使用蔗糖 TAP 溶液重悬、恢复培养和涂板筛选"
        )
    return labels


def _recipe_chunk_order(chunk: dict) -> tuple[int, int]:
    section = str(chunk.get("section") or "")
    if "table 4" in section:
        return (0, int(chunk.get("chunk_index") or 0))
    if "table 1" in section:
        return (1, int(chunk.get("chunk_index") or 0))
    if "table 2" in section:
        return (2, int(chunk.get("chunk_index") or 0))
    if "table 3" in section:
        return (3, int(chunk.get("chunk_index") or 0))
    return (4, int(chunk.get("chunk_index") or 0))


def _manual_chunk_order(chunk: dict) -> tuple[int, int]:
    return (int(chunk.get("page_number") or 999999), int(chunk.get("chunk_index") or 0))


def _append_unit(value: str, unit_suffix: str) -> str:
    if not unit_suffix:
        return value
    if re.search(r"[a-zA-Zμµ]", value):
        return value
    return f"{value}{unit_suffix}"


def _prettify_chemical_name(text: str) -> str:
    replacements = {
        "(NH4)6Mo7O24·4H2O": "(NH₄)₆Mo₇O₂₄·4H₂O",
        "(NH4)6Mo7O24": "(NH₄)₆Mo₇O₂₄",
        "Na2-EDTA2H2O": "Na₂-EDTA·2H₂O",
        "MgSO4 7H2O": "MgSO₄·7H₂O",
        "MgSO4· 7H2O": "MgSO₄·7H₂O",
        "MgSO4·7H2O": "MgSO₄·7H₂O",
        "CaCl2 2H2O": "CaCl₂·2H₂O",
        "CaCl2· 2H2O": "CaCl₂·2H₂O",
        "CaCl2·2H2O": "CaCl₂·2H₂O",
        "ZnSO4 7H2O": "ZnSO₄·7H₂O",
        "FeSO4 7H2O": "FeSO₄·7H₂O",
        "MnCl2 4H2O": "MnCl₂·4H₂O",
        "CoCl2 6H2O": "CoCl₂·6H₂O",
        "CuSO4 5H2O": "CuSO₄·5H₂O",
        "K2HPO4": "K₂HPO₄",
        "KH2PO4": "KH₂PO₄",
        "NH4Cl": "NH₄Cl",
        "H3BO3": "H₃BO₃",
        "H2O": "H₂O",
    }
    formatted = str(text)
    for raw, pretty in replacements.items():
        formatted = formatted.replace(raw, pretty)
    return formatted


def _build_uncertainty(question: str, chunks: list[dict]) -> list[str]:
    uncertainty = []
    if not chunks:
        return [
            "当前知识库没有检索到可引用的来源。",
            "该问题不应凭空回答；请补充 SOP、手册、配方或论文后重新索引。",
        ]
    if _mentions_fact_layer(question):
        uncertainty.append("该回答仅来自知识层；当前状态、代数、pending 和提醒状态必须以 SQLite 事实层为准。")
    doc_types = {chunk.get("doc_type") for chunk in chunks}
    if "paper" in doc_types:
        uncertainty.append("论文来源仅作为参考，不能覆盖实验室内部 SOP 或配方文档。")
    if "media_recipe" not in doc_types and _is_recipe_query(question):
        uncertainty.append("当前回答未命中内部配方文档，正式实验应以实验室 SOP 或配方文件为准。")
    if re.search(r"tap|培养基|medium", question, re.I):
        uncertainty.append("当前知识库不一定说明该配方适用于所有藻种或所有 Chlorella 品系。")
    uncertainty.append("如果用于正式实验操作，应由实验员复核来源文件并按审批流程执行。")
    return uncertainty


def _mentions_fact_layer(question: str) -> bool:
    normalized = question.lower()
    return any(keyword.lower() in normalized for keyword in FACT_LAYER_KEYWORDS)


def _format_location(chunk: dict) -> str:
    if chunk.get("page_number"):
        return f"page {chunk['page_number']}"
    if chunk.get("sheet_name"):
        row_start = chunk.get("row_start")
        row_end = chunk.get("row_end")
        if row_start and row_end:
            return f"sheet {chunk['sheet_name']}, rows {row_start}-{row_end}"
        return f"sheet {chunk['sheet_name']}"
    if chunk.get("section"):
        return str(chunk["section"])
    return f"chunk {chunk.get('chunk_index')}"


def _summarize_content(content: str, max_length: int = 180) -> str:
    text = re.sub(r"\s+", " ", content).strip()
    text = _prettify_chemical_name(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."
