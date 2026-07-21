import asyncio
from types import SimpleNamespace

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.chat import chat_service, response_builder
from app.services.intent import dispatcher
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import RouteKind, RoutingInput
from app.services.intent.write_action_parser import parse_update_fields


def _context():
    return SimpleNamespace(
        session_id="rollout-test",
        strains=[
            {
                "strain_id": "Chlorella_01",
                "name_cn": "小球藻",
                "name_en": "Chlorella vulgaris",
                "generation_number": 14,
                "days_since_last_subculture": 6,
            }
        ],
        pending_actions=[],
        target_pending_actions=[],
        to_prompt_facts=lambda: "rollout context",
    )


def _patch_chat(monkeypatch, events=None):
    monkeypatch.setattr(
        chat_service,
        "get_session_memory",
        lambda session_id: [{"role": "system", "content": "test"}],
    )
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: _context())
    event_sink = events if events is not None else []
    monkeypatch.setattr(chat_service, "append_decision_event", event_sink.append)
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)


def test_routing_audit_logs_structured_decision(monkeypatch, isolated_sqlite_db):
    events = []
    _patch_chat(monkeypatch, events)

    response = asyncio.run(
        chat_service.handle_chat(
            ChatRequest(message="你有哪些工具？", session_id="audit-s1")
        )
    )

    routed = next(item for item in events if item["event"] == "chat_intent_routed")
    assert response.agent_output["action"] == "tool_info"
    assert routed["route_kind"] == "tool_info"
    assert routed["reason_code"] == "tool_info_matched"
    assert "router_version" not in routed


def test_router_executes_compatible_compound_request(monkeypatch, isolated_sqlite_db):
    _patch_chat(monkeypatch)

    def fake_rag(session_id, conversation_history, message):
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "rag_answer"},
            natural_reply=f"rag:{message}",
        )

    monkeypatch.setattr(dispatcher, "handle_rag_intent", fake_rag)

    response = asyncio.run(
        chat_service.handle_chat(
            ChatRequest(
                message="查询 Chlorella_01 当前状态，然后介绍 TAP 培养基",
                session_id="compound-s1",
            )
        )
    )

    assert response.agent_output["action"] == "composite_result"
    assert [item["action"] for item in response.agent_output["steps"]] == [
        "query_strain",
        "rag_answer",
    ]


def test_fullwidth_email_keyword_is_detected():
    message = normalize_input("ＰＬＥＡＳＥ　ＥＭＡＩＬ　ＴＨＥ　ＬＡＢ")
    decision = build_routing_decision(RoutingInput(message, "email-s1", _context()))

    assert decision.kind == RouteKind.EMAIL


def test_direct_write_parser_uses_the_shared_normalizer():
    fields = parse_update_fields("代数：１５")

    assert fields["generation_number"] == 15


def test_direct_router_uses_the_shared_normalizer():
    message = normalize_input("　ＴＡＰ　培养基是什么？　")
    decision = build_routing_decision(RoutingInput(message, "rag-s1", _context()))

    assert decision.kind == RouteKind.KNOWLEDGE_QUERY


def test_email_dispatch_receives_original_not_normalized_text(monkeypatch, isolated_sqlite_db):
    _patch_chat(monkeypatch)
    original = "  ＰＬＥＡＳＥ　ＥＭＡＩＬ　ＴＨＥ　ＬＡＢ  "
    captured = {}

    def fake_email_handler(session_id, conversation_history, message, context_snapshot):
        captured["message"] = message
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "email_draft", "requires_confirmation": True},
            natural_reply="draft",
        )

    monkeypatch.setattr(dispatcher, "handle_email_intent", fake_email_handler)

    response = asyncio.run(
        chat_service.handle_chat(
            ChatRequest(message=original, session_id="email-original-s1")
        )
    )

    assert response.agent_output["action"] == "email_draft"
    assert captured["message"] == original
