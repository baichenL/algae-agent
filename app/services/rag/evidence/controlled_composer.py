from app.models.rag_schema import RagAnswer, RagAnswerSegment, RagCitation


def enrich_controlled_answer(
    question: str,
    answer: RagAnswer,
    citations: list[RagCitation],
) -> RagAnswer:
    if answer.facts and answer.explanations and answer.suggestions:
        return answer

    citation_ids = {item.source_id for item in citations}
    if not answer.facts:
        answer.facts = _build_facts(answer, citations, citation_ids)
    if not answer.explanations:
        answer.explanations = _build_explanations(question, answer, citation_ids)
    if not answer.suggestions:
        answer.suggestions = _build_suggestions(answer, citation_ids)
    answer.uncertainty = _normalize_uncertainty(answer, citation_ids)
    return answer


def _build_facts(
    answer: RagAnswer,
    citations: list[RagCitation],
    citation_ids: set[int],
) -> list[RagAnswerSegment]:
    sufficiency = (answer.debug or {}).get("evidence_sufficiency") or {}
    direct_source_ids = _source_ids_for_evidence_ids(citations, sufficiency.get("direct_evidence_ids") or [])
    background_source_ids = _source_ids_for_evidence_ids(citations, sufficiency.get("background_evidence_ids") or [])
    if sufficiency and sufficiency.get("status") != "sufficient":
        return _build_background_facts(answer, citation_ids, background_source_ids)
    allowed_ids = direct_source_ids or citation_ids
    facts = []
    for item in answer.evidence:
        ids = [item.source_id] if item.source_id in citation_ids else []
        ids = [source_id for source_id in ids if source_id in allowed_ids]
        if not ids:
            continue
        facts.append(
            RagAnswerSegment(
                text=f"{item.quote_summary}（{item.file_name}，{item.location}）",
                citation_ids=ids,
            )
        )
    return facts


def _build_background_facts(
    answer: RagAnswer,
    citation_ids: set[int],
    background_source_ids: set[int],
) -> list[RagAnswerSegment]:
    facts = []
    allowed_ids = background_source_ids or citation_ids
    for item in answer.evidence:
        ids = [item.source_id] if item.source_id in allowed_ids else []
        if not ids:
            continue
        text = item.quote_summary
        if not text.startswith("背景资料"):
            text = f"背景资料：{text}"
        facts.append(
            RagAnswerSegment(
                text=f"{text}（{item.file_name}，{item.location}）",
                citation_ids=ids,
            )
        )
    return facts


def _source_ids_for_evidence_ids(citations: list[RagCitation], evidence_ids: list[str]) -> set[int]:
    wanted = set(evidence_ids or [])
    wanted.update(item.replace("chunk:", "", 1) for item in list(wanted) if item.startswith("chunk:"))
    wanted.update(f"chunk:{item}" for item in list(wanted) if not item.startswith("chunk:"))
    source_ids = set()
    for citation in citations:
        if citation.chunk_id in wanted:
            source_ids.add(citation.source_id)
    return source_ids


def _build_explanations(
    question: str,
    answer: RagAnswer,
    citation_ids: set[int],
) -> list[RagAnswerSegment]:
    status = ((answer.debug or {}).get("answerability") or {}).get("status")
    reason = ((answer.debug or {}).get("answerability") or {}).get("reason")
    all_ids = sorted(citation_ids)
    frame = (answer.debug or {}).get("query_frame") or {}
    if frame.get("question_type") == "overview":
        return [
            RagAnswerSegment(
                text="概览按结构化配方分组整理；表格中的母液用量没有被改写为最终工作液浓度。",
                citation_ids=all_ids,
            )
        ]
    if status == "answered":
        return [
            RagAnswerSegment(
                text="上述结论只基于已选中的本地知识库证据；结构化证据优先于普通文本片段。",
                citation_ids=all_ids,
            )
        ]
    if status in {"partial", "not_found"}:
        return [
            RagAnswerSegment(
                text=f"当前证据不足以给出完整结论；answerability={status or 'unknown'}，原因是 {reason or 'no_explicit_reason'}。",
                citation_ids=all_ids,
            )
        ]
    if not answer.evidence:
        return [
            RagAnswerSegment(
                text="当前没有可引用证据支撑事实性回答。",
                citation_ids=[],
            )
        ]
    return [
        RagAnswerSegment(
            text="回答依据来自检索到的本地知识库证据，未自动引入外部知识。",
            citation_ids=all_ids,
        )
    ]


def _build_suggestions(answer: RagAnswer, citation_ids: set[int]) -> list[RagAnswerSegment]:
    frame = (answer.debug or {}).get("query_frame") or {}
    if frame.get("question_type") == "overview":
        return []
    doc_types = {item.doc_type for item in answer.evidence}
    suggestions = [
        RagAnswerSegment(
            text="如果该回答将用于正式实验操作，应由实验员复核原始来源，并通过既有审批/工作流执行。",
            citation_ids=[],
        )
    ]
    if "paper" in doc_types:
        suggestions.append(
            RagAnswerSegment(
                text="论文证据适合作为背景或方法参考，不能直接替代本实验室 SOP、配方或审批后的培养方案。",
                citation_ids=sorted(citation_ids),
            )
        )
    if "experiment_data" in doc_types:
        suggestions.append(
            RagAnswerSegment(
                text="实验数据结论应区分已观测统计结果和因果/优化判断；没有建模证据时不应推断因果。",
                citation_ids=sorted(citation_ids),
            )
        )
    return suggestions


def _normalize_uncertainty(answer: RagAnswer, citation_ids: set[int]) -> list[str]:
    uncertainty = list(answer.uncertainty or [])
    if answer.conclusion and not citation_ids and "当前回答没有绑定 citation，不能作为事实性依据。" not in uncertainty:
        uncertainty.append("当前回答没有绑定 citation，不能作为事实性依据。")
    frame = (answer.debug or {}).get("query_frame") or {}
    needs_action_boundary = bool(answer.suggestions) or frame.get("question_type") in {"procedure", "optimization"}
    if needs_action_boundary and "建议不会自动转化为执行动作；执行、审批、写库或发邮件必须走受控工具/工作流。" not in uncertainty:
        uncertainty.append("建议不会自动转化为执行动作；执行、审批、写库或发邮件必须走受控工具/工作流。")
    return uncertainty
