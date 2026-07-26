from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.db import agent_artifacts_v2, assistant_conversations, conversation_tasks
from app.core.model_registry import classify_provider_error, model_name
from app.schemas.algae import ChatResponse
from app.services.agent_runtime.state import RuntimeRequestContext
from app.services.chat.answer_envelope import attach_answer_envelope
from app.services.chat.task_coordinator import (
    ResolvedConversationTask,
    synchronize_task_from_response,
)
from app.services.chat.task_spec import build_task_spec
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import (
    CompositeDecision,
    KnowledgeDecision,
    ReasonCode,
    RiskLevel,
    RouteKind,
    RoutingInput,
)
from app.services.scientific.service import run_scientific_task
from app.tools.reasoning_tool_handlers import read_agent_artifact
from app.tools.email_tool import (
    create_email_pending_from_draft,
    handle_email_draft_tool,
)


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        strains=[
            {
                "strain_id": "Chlamydomonas_01",
                "name_cn": "莱茵衣藻",
                "name_en": "Chlamydomonas reinhardtii",
            },
            {
                "strain_id": "Chlorella_01",
                "name_cn": "小球藻",
                "name_en": "Chlorella vulgaris",
            },
        ]
    )


def test_scientific_capability_parser_routes_agent_007_and_binds_target(monkeypatch):
    from app.core.db import scientific

    monkeypatch.setattr(
        scientific,
        "list_datasets",
        lambda: [
            {
                "id": "ds_chlorella0001",
                "strain_id": "Chlorella_01",
                "status": "ready",
                "created_at": "2026-07-25 10:00:00",
            }
        ],
    )
    decision = build_routing_decision(
        RoutingInput(
            normalize_input("对 Chlamydomonas_01 做多源调查、构造三个候选并逐一仿真。"),
            "uat-agent-007",
            _context(),
        )
    )

    assert decision.kind == RouteKind.SCIENTIFIC_TASK
    assert decision.arguments["target_strain_id"] == "Chlamydomonas_01"
    assert decision.arguments["dataset_id"] is None
    assert decision.arguments["candidate_count"] == 3
    assert set(decision.arguments["requested_capabilities"]) >= {
        "investigate",
        "generate_candidates",
        "simulate",
        "multi_source",
    }


def test_scientific_service_rejects_cross_strain_dataset(isolated_sqlite_db):
    from app.core.db import scientific

    scientific.insert_dataset(
        {
            "id": "ds_cross_strain_guard",
            "name": "cross strain fixture",
            "strain_id": "Chlorella_01",
            "source_file": "fixture.csv",
            "content_hash": "cross-strain-hash",
            "mapping": {"metric_name": "biomass"},
            "row_count": 0,
            "status": "ready",
        },
        [],
    )

    with pytest.raises(ValueError, match="scientific_target_dataset_mismatch"):
        run_scientific_task(
            dataset_id="ds_cross_strain_guard",
            target_strain_id="Chlamydomonas_01",
            mode="diagnose",
            create_pending=False,
        )


def test_task_spec_cancels_old_form_and_processes_current_read_question():
    decision = KnowledgeDecision(
        reason_code=ReasonCode.KNOWLEDGE_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="Answer a knowledge question.",
    )
    spec = build_task_spec(
        message="顺便告诉我 TAP 培养基是什么，不要继续刚才的更新。",
        decision=decision,
        active_task={"task_type": "strain_mutation"},
    )

    assert spec.task_relation == "cancel_previous_and_start"
    assert spec.task_domain == "knowledge"
    assert spec.side_effect_policy == "read_only"


def test_explicit_new_scientific_task_suspends_active_update_form(monkeypatch):
    from app.core.db import scientific

    monkeypatch.setattr(scientific, "list_datasets", lambda: [])
    decision = build_routing_decision(
        RoutingInput(
            normalize_input(
                "开始一个新任务：调查 Chlorella_01 最近的生长趋势，只读，不要创建 proposal。"
            ),
            "uat-state-002-new-task",
            _context(),
            active_pending_form={
                "active": True,
                "operation": "update",
                "tool_name": "update_algae_strain",
                "collected_fields": {"strain_id": "Chlamydomonas_01"},
                "missing_fields": ["fields_to_update"],
            },
        )
    )

    assert isinstance(decision, CompositeDecision)
    assert decision.steps[0].kind == RouteKind.PENDING_FORM
    assert decision.steps[0].form_action == "suspend"
    assert decision.steps[-1].kind == RouteKind.SCIENTIFIC_TASK
    assert decision.steps[-1].arguments["target_strain_id"] == "Chlorella_01"


def test_paused_response_persists_paused_task_not_completed(isolated_sqlite_db):
    conversation = assistant_conversations.create_conversation("tester")
    task = conversation_tasks.create_task(
        conversation_id=conversation["id"],
        owner="tester",
        workspace_id="shared",
        goal_text="investigate",
        task_type="scientific",
    )
    resolved = ResolvedConversationTask(
        task=task,
        request_context=RuntimeRequestContext(
            owner="tester",
            role="scientist",
            workspace_id="shared",
            conversation_id=conversation["id"],
            task_id=task["id"],
        ),
        reused_active=False,
    )
    response = ChatResponse(
        session_id=conversation["id"],
        natural_reply="阶段性结果",
        agent_output={
            "action": "agent_task",
            "status": "paused",
            "outcome_status": "paused",
            "budget_exhausted": "tool_calls",
            "checkpoint_id": f"agent-task:{task['id']}",
            "remaining_work": ["验证反证"],
        },
    )

    updated = synchronize_task_from_response(resolved, response)

    assert updated["status"] == "paused"
    assert updated["pause_reason"] == "tool_calls"
    assert updated["proposed_action"]["checkpoint_id"] == f"agent-task:{task['id']}"


def test_truncated_artifact_can_be_read_by_same_run(isolated_sqlite_db):
    artifact = agent_artifacts_v2.save_agent_artifact(
        agent_run_id="run-artifact-owner",
        artifact_type="tool_observation_payload",
        payload={"data": {"rows": [{"value": index} for index in range(20)]}},
    )

    result = read_agent_artifact(
        {
            "artifact_ref": artifact["artifact_id"],
            "agent_run_id": "run-artifact-owner",
        }
    )
    denied = read_agent_artifact(
        {
            "artifact_ref": artifact["artifact_id"],
            "agent_run_id": "different-run",
        }
    )

    assert result["status"] == "success"
    assert result["summary"]["data"]["rows"]["count"] == 20
    assert denied["status"] == "not_found"


def test_answer_envelope_keeps_direct_answer_and_references():
    response = ChatResponse(
        session_id="answer-envelope",
        natural_reply="当前会话中的直接结论。",
        agent_output={
            "status": "partial",
            "trace_id": "trace-1",
            "scientific_run_id": "sci-1",
            "source_statuses": [
                {"source": "rag", "status": "failed", "impact": "缺少 SOP 支持"}
            ],
        },
    )

    attach_answer_envelope(response)
    envelope = response.agent_output["answer_envelope"]

    assert envelope["outcome_status"] == "partial"
    assert envelope["direct_answer"] == "当前会话中的直接结论。"
    assert {"type": "agent_trace", "id": "trace-1"} in envelope["references"]
    assert {"type": "scientific_run", "id": "sci-1"} in envelope["references"]
    assert envelope["unknowns"] == ["缺少 SOP 支持"]


def test_model_registry_roles_and_configuration_errors(monkeypatch):
    monkeypatch.delenv("ALGAE_MODEL_CHAT", raising=False)
    monkeypatch.delenv("ALGAE_MODEL_AGENT", raising=False)

    assert model_name("chat") == "deepseek-v4-flash"
    assert model_name("agent") == "deepseek-v4-pro"
    error = classify_provider_error(
        RuntimeError("supported API model names are deepseek-v4-pro or deepseek-v4-flash")
    )
    assert error["code"] == "model_configuration_error"
    assert error["retryable"] is False
    assert "deepseek" not in error["public_message"].casefold()


def test_email_review_then_send_creates_one_approval_for_exact_draft(
    isolated_sqlite_db,
    monkeypatch,
):
    monkeypatch.setattr(
        "app.tools.email_tool.get_default_recipients",
        lambda: ["test-lab@example.invalid"],
    )
    draft_result = handle_email_draft_tool(
        {
            "message": "提醒明天复查 Chlamydomonas_01，先给我审阅。",
            "session_id": "email-review",
            "agent_run_id": "email-run-draft",
            "created_by_tool_call_id": "draft-call",
            "request_approval": False,
        }
    )
    draft = draft_result["response_payload"]["draft"]

    assert draft_result["response_payload"]["status"] == "success"
    assert draft_result["response_payload"]["pending_id"] is None

    pending = create_email_pending_from_draft(
        draft,
        agent_run_id="email-run-send",
        created_by_tool_call_id="send-call",
        requester="tester",
    )

    assert pending["pending_id"]
    from app.core.db import pending_actions

    stored = pending_actions.get_pending_action(int(pending["pending_id"]))
    assert stored["status"] == "pending"
    assert stored["payload"]["data"]["draft"] == draft
