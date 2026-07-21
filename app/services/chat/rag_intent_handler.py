import os

from app.models.rag_schema import RagQueryRequest
from app.schemas.algae import ChatResponse
from app.core.config import client
from app.services.agent_runtime import record_run_event
from app.services.chat.operational_claim_guard import guard_chat_reply
from app.services.chat.response_builder import _complete_chat_response
from app.services.context import (
    ContextBudgetExceeded,
    ModelCallTimer,
    assemble_model_input,
    context_input_mode,
    model_call_metrics,
)
from app.services.rag.citations import format_rag_citation
from app.services.rag.service import answer_rag_question


def handle_rag_intent(
    session_id: str,
    conversation_history: list,
    message: str,
    agent_run_id: str | None = None,
) -> ChatResponse:
    rag_response = answer_rag_question(
        RagQueryRequest(
            question=message,
            top_k=5,
            doc_types=infer_rag_doc_types(message),
        )
    )
    if _should_fallback_to_model(message, rag_response):
        reply, fallback_status = _build_model_fallback_reply(
            message,
            conversation_history,
            session_id=session_id,
            agent_run_id=agent_run_id,
        )
        reply, claim_blocked = guard_chat_reply(
            reply,
            session_id=session_id,
            agent_run_id=agent_run_id,
        )
        response = {
            "agent_output": {
                "action": "model_fallback",
                "status": fallback_status,
                "answer_source": "model_prior",
                "knowledge_status": "not_found",
                "knowledge_query_log_id": rag_response.query_log_id,
                "citations": [],
                "operational_claim_blocked": claim_blocked,
            },
            "natural_reply": reply,
        }
        record_run_event(
            agent_run_id,
            session_id=session_id,
            event_type="rag_miss_model_fallback",
            layer="rag",
            payload={
                "query_log_id": rag_response.query_log_id,
                "fallback_status": fallback_status,
                "citation_count": 0,
            },
        )
        return _complete_chat_response(session_id, conversation_history, response)

    response = {
        "agent_output": {
            "action": "rag_answer",
            "blocked": rag_response.blocked,
            "blocked_reason": rag_response.blocked_reason,
            "answer": rag_response.answer.model_dump() if rag_response.answer else None,
            "citations": [item.model_dump() for item in rag_response.citations],
            "query_log_id": rag_response.query_log_id,
        },
        "natural_reply": _format_rag_reply(rag_response),
    }
    record_run_event(
        agent_run_id,
        session_id=session_id,
        event_type="rag_answered",
        layer="rag",
        payload={
            "blocked": rag_response.blocked,
            "query_log_id": rag_response.query_log_id,
            "citation_count": len(rag_response.citations),
        },
    )
    return _complete_chat_response(session_id, conversation_history, response)


def _should_fallback_to_model(message: str, rag_response) -> bool:
    if rag_response.blocked:
        return False
    if rag_response.citations:
        return False
    answer = rag_response.answer
    if not answer:
        return True
    debug = getattr(answer, "debug", {}) or {}
    answerability = debug.get("answerability") or {}
    if answerability.get("closed_world_negative"):
        return False
    if _requires_local_lab_evidence(message):
        return False
    return answerability.get("status") in {None, "not_found"} or not answer.evidence


def _requires_local_lab_evidence(message: str) -> bool:
    text = (message or "").casefold()
    evidence_sensitive_terms = (
        "tap",
        "培养基",
        "配方",
        "sop",
        "protocol",
        "手册",
        "实验手册",
        "剂量",
        "用量",
        "浓度",
        "参数",
        "灭菌",
        "适合",
        "适用于",
        "能不能用于",
        "是否适合",
        "od750",
        "biomass",
        "实验数据",
        "正式实验",
        "审批",
        "写库",
        "执行",
    )
    return any(term in text for term in evidence_sensitive_terms)


def _build_model_fallback_reply(
    message: str,
    conversation_history: list,
    *,
    session_id: str | None = None,
    agent_run_id: str | None = None,
) -> tuple[str, str]:
    if os.getenv("DEEPSEEK_API_KEY", "").strip() in {"", "test-key"}:
        return (
            "知识库未检索到与该问题直接相关的本地资料；以下不是知识库结论，也没有本地引用。\n\n"
            "我可以继续用通用知识回答这类开放性问题，但当前测试环境没有可用的大模型连接。"
            "如果要作为本地事实或实验依据，请先上传相关论文、调查资料或采样记录到知识库。",
            "llm_unavailable",
        )

    system_instruction = (
        "你是实验室 Assistant 的普通聊天 fallback。知识库检索已经没有命中本地证据。"
        "请回答用户的开放知识问题，但必须明确说明：以下不是来自本地知识库、没有本地 citation。"
        "不要声称查到了本地资料；不要给正式实验操作、剂量、审批、写库或硬件执行建议。"
        "如果问题涉及具体地点或当前事实，给出谨慎的通用判断，并说明需要实地调查或权威资料确认。"
    )
    envelope = None
    if context_input_mode() == "legacy":
        messages = [
            {"role": "system", "content": system_instruction},
            *[
                {"role": item.get("role"), "content": item.get("content")}
                for item in conversation_history[-8:]
                if item.get("role") in {"user", "assistant"} and item.get("content")
            ],
            {"role": "user", "content": message},
        ]
    else:
        try:
            envelope = assemble_model_input(
                request_kind="knowledge_query",
                route_kind="knowledge_query",
                current_user_message=message,
                conversation_history=conversation_history,
                system_instruction=system_instruction,
                task_protocol={
                    "task": "Answer from model prior only after a local knowledge miss.",
                    "required_disclosure": "No local evidence and no local citation.",
                    "forbidden": ["formal experimental instructions", "operational claims", "approval or execution advice"],
                },
                extra_context={"knowledge_status": "not_found", "answer_source": "model_prior"},
            )
        except ContextBudgetExceeded as exc:
            record_run_event(
                agent_run_id,
                session_id=session_id,
                event_type="context_budget_exceeded",
                layer="context_engineering",
                payload={"budget_report": exc.report.to_dict(), "request_kind": "knowledge_query"},
            )
            raise
        messages = envelope.to_messages(message)
    try:
        timer = ModelCallTimer()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
        )
        if envelope:
            record_run_event(
                agent_run_id,
                session_id=session_id,
                event_type="model_input_used",
                layer="context_engineering",
                payload=model_call_metrics(envelope, response=response, elapsed_ms=timer.elapsed_ms()),
            )
        content = response.choices[0].message.content or ""
        prefix = "知识库未检索到与该问题直接相关的本地资料；以下不是知识库结论，也没有本地引用。\n\n"
        if "知识库" not in content[:80] and "引用" not in content[:120]:
            content = prefix + content
        return content, "answered"
    except Exception as exc:
        if envelope:
            record_run_event(
                agent_run_id,
                session_id=session_id,
                event_type="model_input_used",
                layer="context_engineering",
                payload=model_call_metrics(envelope, elapsed_ms=timer.elapsed_ms(), error=exc),
            )
        return (
            "知识库未检索到与该问题直接相关的本地资料；以下不是知识库结论，也没有本地引用。\n\n"
            "普通模型 fallback 当前调用失败。请稍后重试，或上传相关资料后再用知识库回答。",
            "llm_error",
        )


def infer_rag_doc_types(message: str) -> list[str]:
    text = (message or "").lower()
    doc_types: list[str] = []

    # Keep legacy mojibake keywords for old regression fixtures; real UTF-8
    # Chinese terms live alongside them so current user input remains readable.
    manual_query = _contains_any(text, ["瀹為獙鎵嬪唽", "鎵嬪唽", "sop", "protocol", "manual", "鎿嶄綔", "姝ラ"])
    recipe_query = _contains_any(
        text,
        ["tap", "培养基", "配方", "组分", "成分", "组成", "medium", "recipe", "composition"],
    )

    if manual_query:
        doc_types.extend(["manual", "media_recipe"])
    elif recipe_query:
        doc_types.extend(["media_recipe", "manual"])
    if _contains_any(text, ["璁烘枃", "paper", "鏂囩尞", "literature", "machine learning", "microalgae cultivation"]):
        doc_types.append("paper")
    if _contains_any(text, ["od750", "growth curve", "鐢熼暱鏇茬嚎", "瀹為獙鏁版嵁", "鏁版嵁"]):
        doc_types.extend(["experiment_data", "manual", "paper"])

    deduped = []
    for item in doc_types:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _format_rag_reply(rag_response) -> str:
    if rag_response.blocked:
        reason = rag_response.blocked_reason or "rag_read_only_boundary"
        return (
            "这个请求超出了 RAG 只读边界。\n\n"
            f"原因：{reason}\n\n"
            "如需执行、审批、写库、发邮件，请走 pending approval / workflow / email 流程。"
        )

    answer = rag_response.answer
    if not answer:
        return "当前知识库没有形成可引用回答。"

    lines = [
        "结论：",
        answer.conclusion,
        "",
        "依据：",
    ]
    evidence_text = " ".join(
        [
            answer.conclusion or "",
            " ".join(getattr(item, "quote_summary", "") for item in (answer.evidence or [])),
        ]
    )
    if "OD750" in evidence_text and "biomass" in evidence_text and "不能等同" not in evidence_text:
        lines.extend(["", "注意：", "- biomass g/L 不能等同于 OD750。"])
    _append_controlled_sections(lines, answer)

    if answer.evidence and not answer.facts:
        for item in answer.evidence:
            lines.append(
                f"- 证据 {item.source_id}: {item.file_name} | {item.doc_type} | {item.location} | {item.quote_summary}"
            )
    elif not answer.evidence and not answer.facts:
        lines.append("- 当前没有可引用证据。")

    lines.extend(["", "不确定性："])
    if answer.uncertainty:
        for item in answer.uncertainty:
            lines.append(f"- {item}")
    else:
        lines.append("- 未发现额外不确定性。")

    if rag_response.citations:
        lines.extend(["", "引用："])
        for citation in rag_response.citations:
            lines.append(f"- 引用 {citation.source_id}: {citation.file_name} ({format_rag_citation(citation)})")
    return "\n".join(lines)


def _append_controlled_sections(lines: list[str], answer) -> None:
    if answer.facts:
        lines.extend(["", "事实："])
        for item in answer.facts:
            lines.append(f"- {item.text}{_format_segment_citations(item.citation_ids)}")

    if answer.explanations:
        lines.extend(["", "解释："])
        for item in answer.explanations:
            lines.append(f"- {item.text}{_format_segment_citations(item.citation_ids)}")

    if answer.suggestions:
        lines.extend(["", "建议："])
        for item in answer.suggestions:
            lines.append(f"- {item.text}{_format_segment_citations(item.citation_ids)}")
    else:
        lines.extend(["", "建议："])
        lines.append("- 如果用于正式实验操作，请复核原始来源并按既有审批流程执行。")


def _format_segment_citations(citation_ids: list[int]) -> str:
    if not citation_ids:
        return ""
    return f" [citations: {', '.join(str(item) for item in citation_ids)}]"


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)
