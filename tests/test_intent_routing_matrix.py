import asyncio
from types import SimpleNamespace

import pytest

from app.schemas.algae import ChatRequest
from app.services.agent_runtime.decision import decision_from_routing_decision
from app.services.agent_runtime.executor import build_agent_action
from app.services.agent_runtime.policy import evaluate_policy
from app.services.agent_runtime.state import AgentRunState
from app.services.chat import chat_service, pending_form_state as pfs
from app.services.intent import intent_router
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_handlers import handle_query_status_intent
from app.services.intent.llm_candidate_provider import LlmRouteCandidate
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import (
    ClarificationDecision,
    CompositeDecision,
    QueryDecision,
    ReasonCode,
    RiskLevel,
    RouteKind,
    RoutingInput,
)


def _context(strains=None, pending=None):
    return SimpleNamespace(
        session_id="routing-test",
        strains=strains
        if strains is not None
        else [
            {
                "strain_id": "Chlorella_01",
                "name_cn": "小球藻",
                "name_en": "Chlorella vulgaris",
                "generation_number": 14,
                "days_since_last_subculture": 6,
            },
            {
                "strain_id": "Chlamydomonas_01",
                "name_cn": "莱茵衣藻",
                "name_en": "Chlamydomonas reinhardtii",
                "generation_number": 2,
                "days_since_last_subculture": 1,
            },
        ],
        pending_actions=pending or [],
        target_pending_actions=[],
        to_prompt_facts=lambda: "test context",
    )


def _decision(message, *, context=None, active_form=None):
    ctx = context or _context()
    normalized = normalize_input(message)
    return build_routing_decision(
        RoutingInput(
            message=normalized,
            session_id="routing-test",
            context_snapshot=ctx,
            active_pending_form=active_form,
        )
    )


@pytest.mark.parametrize(
    "raw,normalized,match_text",
    [
        ("  hello  ", "hello", "hello"),
        ("hello\n\tworld", "hello world", "hello world"),
        ("ＥＭＡＩＬ", "EMAIL", "email"),
        ("ＡＢＣ＿０１", "ABC_01", "abc_01"),
        ("查询：小球藻", "查询:小球藻", "查询:小球藻"),
        ("查询，小球藻", "查询,小球藻", "查询,小球藻"),
        ("查询；小球藻", "查询;小球藻", "查询;小球藻"),
        ("查询。小球藻", "查询.小球藻", "查询.小球藻"),
        ("“小球藻”", '"小球藻"', '"小球藻"'),
        ("‘mail’", "'mail'", "'mail'"),
        ("Cafe\u0301", "Café", "café"),
        ("  多个　空格  ", "多个 空格", "多个 空格"),
    ],
)
def test_input_normalization_is_conservative(raw, normalized, match_text):
    result = normalize_input(raw)

    assert result.original_text == raw
    assert result.normalized_text == normalized
    assert result.match_text == match_text


@pytest.mark.parametrize(
    "message,expected_kind",
    [
        ("你有哪些工具？", RouteKind.TOOL_INFO),
        ("当前可用工具列表", RouteKind.TOOL_INFO),
        ("请发邮件提醒实验员", RouteKind.EMAIL),
        ("PLEASE EMAIL THE LAB", RouteKind.EMAIL),
        ("mail reminder", RouteKind.EMAIL),
        ("当前有哪些藻种？", RouteKind.LAB_QUERY),
        ("列出所有品系", RouteKind.LAB_QUERY),
        ("Chlorella_01 当前第几代？", RouteKind.LAB_QUERY),
        ("当前 Chlorella_01 的 generation_number 是多少？", RouteKind.LAB_QUERY),
        ("哪些品系需要传代？", RouteKind.LAB_QUERY),
        ("有哪些待确认请求？", RouteKind.PENDING_QUERY),
        ("list pending", RouteKind.PENDING_QUERY),
        ("TAP 培养基是什么？", RouteKind.KNOWLEDGE_QUERY),
        ("介绍一下 SOP", RouteKind.KNOWLEDGE_QUERY),
        ("解释微藻光合作用原理", RouteKind.KNOWLEDGE_QUERY),
        ("更新 Chlorella_01 的代数为 15", RouteKind.WRITE_ACTION),
        ("删除 Chlorella_01", RouteKind.WRITE_ACTION),
        ("新增 Spirulina_02 | 螺旋藻 (Spirulina platensis)", RouteKind.WRITE_ACTION),
        ("执行 Chlorella_01 的传代", RouteKind.WORKFLOW),
        ("现在启动 Chlorella_01 传代流程", RouteKind.WORKFLOW),
        ("传代", RouteKind.CLARIFICATION),
        ("你好", RouteKind.CHAT),
        ("谢谢", RouteKind.CHAT),
        ("帮我总结一下", RouteKind.CHAT),
        ("你好，简单介绍一下你自己", RouteKind.CHAT),
        ("　ＰＬＥＡＳＥ　ＥＭＡＩＬ　ＴＨＥ　ＬＡＢ　", RouteKind.EMAIL),
    ],
)
def test_non_conflicting_routing_matrix(message, expected_kind):
    assert _decision(message).kind == expected_kind


@pytest.mark.parametrize(
    "message,expected_kind,expected_query_type",
    [
        ("你有哪些工具？", RouteKind.TOOL_INFO, None),
        ("当前有哪些藻种？", RouteKind.LAB_QUERY, "list_strains"),
        ("Chlorella_01 当前代数是多少？", RouteKind.LAB_QUERY, "generation_number"),
        ("哪些藻种需要传代？", RouteKind.LAB_QUERY, "due_subculture"),
        ("帮我处理一下快到期的藻株", RouteKind.LAB_QUERY, "due_subculture"),
        ("小球藻现在怎么样？", RouteKind.LAB_QUERY, "strain_status"),
        ("帮我总结一下 Chlorella_01 的状态", RouteKind.LAB_QUERY, "strain_status"),
        ("小球藻怎么传代？", RouteKind.KNOWLEDGE_QUERY, None),
        ("传代周期多久？", RouteKind.KNOWLEDGE_QUERY, None),
        ("小球藻是什么？", RouteKind.KNOWLEDGE_QUERY, None),
        ("帮我把小球藻过一代", RouteKind.WORKFLOW, None),
        ("发送邮件提醒检查 Chlorella_01 状态", RouteKind.EMAIL, None),
    ],
)
def test_real_utf8_chinese_routing_matrix(message, expected_kind, expected_query_type):
    decision = _decision(message)

    assert decision.kind == expected_kind
    if expected_query_type:
        assert decision.query_type == expected_query_type


def test_colloquial_subculture_phrase_with_verified_target_routes_to_workflow(monkeypatch):
    monkeypatch.setattr(intent_router, "collect_llm_route_candidates", lambda **kwargs: [])

    decision = _decision("帮 Chlorella_01 过一代")

    assert decision.kind == RouteKind.WORKFLOW


def test_llm_open_high_risk_expression_requires_deterministic_clarification(monkeypatch):
    def fake_llm_candidates(**kwargs):
        return [
            LlmRouteCandidate(
                route_kind=RouteKind.WORKFLOW,
                confidence=0.86,
                speech_act=None,
                risk_level=RiskLevel.HIGH,
                evidence=("llm mapped open cultivation phrasing to subculture workflow",),
                mentioned_targets=("Chlorella_01",),
            )
        ]

    monkeypatch.setattr(intent_router, "collect_llm_route_candidates", fake_llm_candidates)

    decision = _decision("让 Chlorella_01 进入下一轮培养")

    assert decision.kind == RouteKind.CLARIFICATION
    assert decision.selection_trace["selector"] == "hybrid_deterministic_v1"
    assert any(item["target"] == "Chlorella_01" for item in decision.selection_trace["candidates"])
    assert any(item["source"] == "llm" for item in decision.selection_trace["candidates"])
    assert any(
        "llm mapped open cultivation" in evidence
        for item in decision.selection_trace["candidates"]
        for evidence in item["evidence_items"]
    )
    assert any(
        "high_risk_deterministic_routing_required" in item["negative_signals"]
        for item in decision.selection_trace["candidates"]
    )


def test_rule_audit_outranks_llm_workflow_candidate(monkeypatch):
    def fake_llm_candidates(**kwargs):
        return [
            LlmRouteCandidate(
                route_kind=RouteKind.WORKFLOW,
                confidence=0.89,
                risk_level=RiskLevel.HIGH,
                evidence=("llm saw a possible workflow request",),
                mentioned_targets=("Chlorella_01",),
            )
        ]

    monkeypatch.setattr(intent_router, "collect_llm_route_candidates", fake_llm_candidates)

    decision = _decision("为什么 Chlorella_01 没有执行传代？")

    assert decision.kind == RouteKind.WORKFLOW_AUDIT
    assert decision.selection_trace["selection_reason"] == "audit_or_confirmation_outranks_execution"
    candidate_sources = {(item["route_kind"], item["source"]) for item in decision.selection_trace["candidates"]}
    assert ("workflow_audit", "rule") in candidate_sources
    assert ("workflow", "llm") in candidate_sources


def test_llm_workflow_candidate_with_missing_target_is_clarification(monkeypatch):
    def fake_llm_candidates(**kwargs):
        return [
            LlmRouteCandidate(
                route_kind=RouteKind.WORKFLOW,
                confidence=0.86,
                risk_level=RiskLevel.HIGH,
                evidence=("llm mapped open cultivation phrasing to workflow",),
                mentioned_targets=("Chlorella_99",),
            )
        ]

    monkeypatch.setattr(intent_router, "collect_llm_route_candidates", fake_llm_candidates)

    decision = _decision("让 Chlorella_99 进入下一轮培养")

    assert isinstance(decision, ClarificationDecision)
    assert decision.reason_code == ReasonCode.TARGET_NOT_FOUND


def test_llm_workflow_candidate_with_multiple_targets_is_clarification(monkeypatch):
    def fake_llm_candidates(**kwargs):
        return [
            LlmRouteCandidate(
                route_kind=RouteKind.WORKFLOW,
                confidence=0.86,
                risk_level=RiskLevel.HIGH,
                evidence=("llm mapped open cultivation phrasing to workflow",),
                mentioned_targets=("Chlorella_01", "Spirulina_01"),
            )
        ]

    monkeypatch.setattr(intent_router, "collect_llm_route_candidates", fake_llm_candidates)
    context = _context(
        strains=[
            {"strain_id": "Chlorella_01", "name_cn": "小球藻", "name_en": "Chlorella vulgaris"},
            {"strain_id": "Spirulina_01", "name_cn": "螺旋藻", "name_en": "Spirulina platensis"},
        ]
    )

    decision = _decision("让 Chlorella_01 和 Spirulina_01 进入下一轮培养", context=context)

    assert isinstance(decision, ClarificationDecision)
    assert decision.reason_code == ReasonCode.MULTIPLE_TARGETS_CONFLICT


@pytest.mark.parametrize(
    "message",
    [
        "查询 Chlorella_01 当前状态，然后删除 Chlorella_01",
        "请发邮件提醒实验员，然后删除 Chlorella_01",
        "查询当前藻种；然后介绍 TAP 培养基",
        "有哪些 pending；然后查询当前藻种",
        "执行 Chlorella_01 的传代 and 发邮件提醒实验员",
        "TAP 培养基是什么 then 删除 Chlorella_01",
    ],
)
def test_compatible_intent_frames_form_composite_plan(message):
    decision = _decision(message)

    assert isinstance(decision, CompositeDecision)
    assert decision.reason_code.value == "composite_plan"
    assert len(decision.steps) >= 2


@pytest.mark.parametrize(
    "message",
    [
        "删除 Chlorella_01，然后执行 Chlorella_01 的传代",
        "更新 Chlorella_01 的代数为 15；然后删除 Chlorella_01",
    ],
)
def test_multiple_state_proposals_still_require_clarification(message):
    decision = _decision(message)

    assert isinstance(decision, ClarificationDecision)
    assert decision.reason_code.value == "multiple_intents_conflict"


def test_unknown_explicit_target_is_not_treated_as_confirmed():
    decision = _decision("删除 Missing_99")

    assert isinstance(decision, ClarificationDecision)
    assert decision.reason_code.value == "target_not_found"


def test_database_field_name_is_not_misclassified_as_strain_id():
    decision = _decision("当前 Chlorella_01 的 generation_number 是多少？")

    assert decision.kind == RouteKind.LAB_QUERY


def test_multiple_read_targets_are_preserved_in_query_frame():
    decision = _decision("查询小球藻和莱茵衣藻当前状态")

    assert isinstance(decision, QueryDecision)
    assert [item.canonical_id for item in decision.targets] == [
        "Chlorella_01",
        "Chlamydomonas_01",
    ]


def test_multiple_query_types_form_read_only_plan():
    decision = _decision("查询 Chlorella_01 的当前状态和代数")

    assert isinstance(decision, CompositeDecision)
    assert {step.query_type for step in decision.steps} == {
        "strain_status",
        "generation_number",
    }


def test_active_form_cannot_bypass_multi_target_query_validation():
    decision = _decision(
        "查询 Chlorella_01 和 Chlamydomonas_01 当前状态",
        active_form={"active": True, "operation": "add"},
    )

    assert isinstance(decision, QueryDecision)
    assert len(decision.targets) == 2

    response = handle_query_status_intent(decision, _context())
    assert response["agent_output"]["action"] == "query_strains"
    assert response["agent_output"]["strain_ids"] == [
        "Chlorella_01",
        "Chlamydomonas_01",
    ]


def test_direct_external_effect_is_not_composed_with_another_operation():
    decision = _decision("删除 Chlorella_01，然后立即发送邮件提醒实验员")

    assert isinstance(decision, ClarificationDecision)
    assert decision.reason_code == ReasonCode.MULTIPLE_INTENTS_CONFLICT


def test_reason_text_does_not_control_pending_query_behavior():
    decision = QueryDecision(
        kind=RouteKind.PENDING_QUERY,
        query_type="pending",
        reason_code=ReasonCode.PENDING_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="this explanation is intentionally unrelated",
    )
    ctx = _context(pending=[{"id": 1}])

    response = handle_query_status_intent(decision, ctx)

    assert response["agent_output"]["action"] == "list_pending"
    assert response["agent_output"]["pending"] == [{"id": 1}]


def test_reason_text_does_not_control_due_query_behavior():
    decision = QueryDecision(
        query_type="due_subculture",
        reason_code=ReasonCode.LAB_QUERY_MATCHED,
        risk_level=RiskLevel.LOW,
        explanation="arbitrary human explanation",
    )

    response = handle_query_status_intent(decision, _context())

    assert response["agent_output"]["action"] == "query_due_subculture"


def test_conflict_dispatch_has_no_side_effects_and_keeps_form(tmp_path, monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    pfs.save_pending_form_state(
        "conflict-s1",
        {
            "active": True,
            "operation": "add",
            "tool_name": "add_algae_strain",
            "collected_fields": {"name_cn": "螺旋藻"},
            "missing_fields": ["strain_id", "name_en"],
            "candidates": [],
            "source_message": "增加螺旋藻",
        },
    )
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: _context())
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)

    async def forbidden_tool(*args, **kwargs):
        raise AssertionError("clarification must not call tools")

    async def forbidden_llm(*args, **kwargs):
        raise AssertionError("clarification must not call the LLM")

    monkeypatch.setattr(chat_service, "execute_registered_tool", forbidden_tool)
    monkeypatch.setattr(chat_service, "_handle_llm_or_tool_path", forbidden_llm)

    response = asyncio.run(
        chat_service.handle_chat(
            ChatRequest(
                message="删除 Chlorella_01，然后执行 Chlorella_01 的传代",
                session_id="conflict-s1",
            )
        )
    )

    assert response.agent_output["action"] == "pending_form_conflict"
    assert pfs.get_pending_form_state("conflict-s1") is not None


def test_original_message_is_preserved_in_pending_form(tmp_path, monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(chat_service, "get_session_memory", lambda session_id: [{"role": "system", "content": "test"}])
    monkeypatch.setattr(chat_service, "build_context_snapshot", lambda session_id: _context())
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)
    original = "  新增　螺旋藻  "

    asyncio.run(chat_service.handle_chat(ChatRequest(message=original, session_id="original-s1")))

    assert pfs.get_pending_form_state("original-s1")["source_message"] == original
