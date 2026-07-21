import asyncio
import json

from app.core.db.connection import connect
from app.schemas.algae import ChatRequest
from app.services.agent_runtime.decision import decision_from_routing_decision
from app.services.agent_runtime.state import AgentRunState
from app.services.chat import chat_service, response_builder
from app.services.intent.routing_models import (
    ChatDecision,
    ClarificationDecision,
    CompositeDecision,
    EmailDecision,
    EntityRef,
    KnowledgeDecision,
    PendingFormDecision,
    QueryDecision,
    ReasonCode,
    RiskLevel,
    RouteCandidate,
    RouteKind,
    ToolInfoDecision,
    WorkflowDecision,
    WriteDecision,
)
from tests.conftest import context


def _state() -> AgentRunState:
    return AgentRunState(
        agent_run_id="decision-run",
        session_id="decision-session",
        user_message="test",
        conversation_history=[],
    )


def _candidate(kind: RouteKind) -> RouteCandidate:
    return RouteCandidate(
        kind=kind,
        reason_code=ReasonCode.DEFAULT_CHAT,
        risk_level=RiskLevel.NONE,
        evidence="test",
    )


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with connect(row_factory=True) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def test_decision_contract_for_read_routes():
    cases = [
        (
            ToolInfoDecision(
                reason_code=ReasonCode.TOOL_INFO_MATCHED,
                risk_level=RiskLevel.NONE,
                explanation="tool info",
                candidates=(_candidate(RouteKind.TOOL_INFO),),
            ),
            "tool_info",
            "tool_info",
        ),
        (
            QueryDecision(
                kind=RouteKind.LAB_QUERY,
                reason_code=ReasonCode.LAB_QUERY_MATCHED,
                risk_level=RiskLevel.LOW,
                explanation="lab query",
                query_type="generation_number",
                candidates=(_candidate(RouteKind.LAB_QUERY),),
            ),
            "lab_query",
            "generation_number",
        ),
        (
            QueryDecision(
                kind=RouteKind.PENDING_QUERY,
                reason_code=ReasonCode.PENDING_QUERY_MATCHED,
                risk_level=RiskLevel.LOW,
                explanation="pending query",
                query_type="pending",
                candidates=(_candidate(RouteKind.PENDING_QUERY),),
            ),
            "pending_query",
            "list_pending_actions",
        ),
        (
            KnowledgeDecision(
                reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
                risk_level=RiskLevel.LOW,
                explanation="rag",
                candidates=(_candidate(RouteKind.KNOWLEDGE_QUERY),),
            ),
            "knowledge_query",
            "retrieve_rag",
        ),
    ]

    for raw_decision, route_kind, action_name in cases:
        decision = decision_from_routing_decision(raw_decision, _state())
        assert decision.route_kind == route_kind
        assert decision.action_type == "read"
        assert decision.action_name == action_name
        assert decision.requires_approval is False
        assert decision.can_continue is True
        assert decision.confidence == 1.0
        assert decision.decision_source == "intent_router"


def test_decision_contract_for_approval_routes():
    target = EntityRef("strain", "Chlorella_01", "Chlorella_01", "database_id", True)
    cases = [
        (
            WriteDecision(
                reason_code=ReasonCode.WRITE_ACTION_MATCHED,
                risk_level=RiskLevel.MEDIUM,
                explanation="write",
                operation="add",
                tool_name="add_algae_strain",
                arguments={"strain_id": "Chlorella_01"},
                target=target,
                candidates=(_candidate(RouteKind.WRITE_ACTION),),
            ),
            "write_action",
            "approval_request",
            "add_algae_strain",
        ),
        (
            WorkflowDecision(
                reason_code=ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
                risk_level=RiskLevel.HIGH,
                explanation="workflow",
                target=target,
                candidates=(_candidate(RouteKind.WORKFLOW),),
            ),
            "workflow",
            "approval_request",
            "trigger_subculture_workflow",
        ),
        (
            EmailDecision(
                reason_code=ReasonCode.EMAIL_MATCHED,
                risk_level=RiskLevel.MEDIUM,
                explanation="email",
                candidates=(_candidate(RouteKind.EMAIL),),
            ),
            "email",
            "draft_or_approval_request",
            "email_draft",
        ),
    ]

    for raw_decision, route_kind, action_type, action_name in cases:
        decision = decision_from_routing_decision(raw_decision, _state())
        assert decision.route_kind == route_kind
        assert decision.action_type == action_type
        assert decision.action_name == action_name
        assert decision.requires_approval is True
        assert decision.can_continue is True
        assert decision.confidence == 1.0
        if raw_decision.target:
            assert decision.target["canonical_id"] == "Chlorella_01"


def test_decision_contract_for_terminal_and_internal_routes():
    clarification = decision_from_routing_decision(
        ClarificationDecision(
            reason_code=ReasonCode.AMBIGUOUS_SUBCULTURE,
            risk_level=RiskLevel.HIGH,
            explanation="ambiguous",
            questions=("which one?",),
            candidates=(_candidate(RouteKind.CLARIFICATION),),
        ),
        _state(),
    )
    assert clarification.action_type == "clarification"
    assert clarification.can_continue is False
    assert clarification.confidence == 0.5
    assert clarification.terminal_hint == "needs_clarification"

    chat = decision_from_routing_decision(
        ChatDecision(
            reason_code=ReasonCode.DEFAULT_CHAT,
            risk_level=RiskLevel.NONE,
            explanation="chat",
            candidates=(_candidate(RouteKind.CHAT),),
        ),
        _state(),
    )
    assert chat.action_type == "chat"
    assert chat.action_name == "chat"
    assert chat.confidence == 0.6

    composite = decision_from_routing_decision(
        CompositeDecision(
            reason_code=ReasonCode.COMPOSITE_PLAN,
            risk_level=RiskLevel.LOW,
            explanation="composite",
            steps=(chat.raw_decision,),
            candidates=(_candidate(RouteKind.COMPOSITE),),
        ),
        _state(),
    )
    assert composite.action_type == "composite"
    assert composite.action_name == "composite"
    assert composite.can_continue is True


def test_missing_fields_make_decision_non_continuable():
    raw_decision = PendingFormDecision(
        reason_code=ReasonCode.REQUIRED_FIELDS_MISSING,
        risk_level=RiskLevel.MEDIUM,
        explanation="missing",
        form_action="start",
        operation="add",
        tool_name="add_algae_strain",
        missing_fields=("name_en",),
        candidates=(_candidate(RouteKind.WRITE_ACTION),),
    )

    decision = decision_from_routing_decision(raw_decision, _state())

    assert decision.missing_fields == ["name_en"]
    assert decision.can_continue is False
    assert decision.terminal_hint == "needs_more_info"
    assert decision.requires_approval is True


def test_agent_decision_event_contains_expanded_contract(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: context())
    monkeypatch.setattr(chat_service.uuid, "uuid4", lambda: "run-decision-event-1")

    response = asyncio.run(
        chat_service.handle_chat(ChatRequest(message="你有哪些工具？", session_id="decision-event-s1"))
    )

    assert response.agent_output["action"] == "tool_info"
    rows = _rows(
        "SELECT payload_json FROM agent_run_events WHERE agent_run_id = ? AND event_type = ?",
        ("run-decision-event-1", "agent_decision_made"),
    )
    payload = json.loads(rows[0]["payload_json"])
    decision = payload["decision"]
    assert decision["target"] is None
    assert decision["candidate_routes"] == ["tool_info"]
    assert decision["decision_source"] == "intent_router"
    assert decision["terminal_hint"] is None
    assert decision["confidence"] == 1.0
