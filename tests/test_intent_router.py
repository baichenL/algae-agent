from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import KnowledgeDecision, PendingFormDecision, RouteKind, RoutingInput, WriteDecision
from tests.conftest import context


def _decision(message: str, ctx=None):
    return build_routing_decision(RoutingInput(
        message=normalize_input(message),
        session_id="intent-router-test",
        context_snapshot=ctx or context(),
    ))


def test_router_routes_incomplete_add_to_pending_form():
    decision = _decision("增加衣藻")

    assert isinstance(decision, PendingFormDecision)
    assert decision.operation == "add"
    assert decision.tool_name == "add_algae_strain"
    assert "strain_id" in decision.missing_fields


def test_router_extracts_update_new_strain_id():
    decision = _decision(
        "把实验室里的莱茵衣藻的品系改为 Chlamydomonas_137AH",
        context(strains=[{
            "strain_id": "Chlamydomonas_01",
            "name_cn": "莱茵衣藻",
            "name_en": "Chlamydomonas reinhardtii",
        }]),
    )

    assert isinstance(decision, WriteDecision)
    assert decision.kind == RouteKind.WRITE_ACTION
    assert decision.operation == "update"
    assert decision.arguments["strain_id"] == "Chlamydomonas_01"
    assert decision.arguments["new_strain_id"] == "Chlamydomonas_137AH"


def test_knowledge_question_does_not_require_unique_database_strain_target():
    decision = _decision(
        "TAP 是否适合莱茵衣藻？",
        context(strains=[
            {"strain_id": "Chlamydomonas_01", "name_cn": "莱茵衣藻", "name_en": "Chlamydomonas reinhardtii"},
            {"strain_id": "Chlamydomonas_02", "name_cn": "莱茵衣藻", "name_en": "Chlamydomonas reinhardtii"},
        ]),
    )

    assert isinstance(decision, KnowledgeDecision)
    assert decision.kind == RouteKind.KNOWLEDGE_QUERY


def test_external_ecology_algae_question_does_not_query_lab_strain_database():
    decision = _decision("我想知道在沈阳市，野外分布有哪些常见藻种")

    assert isinstance(decision, KnowledgeDecision)
    assert decision.kind == RouteKind.KNOWLEDGE_QUERY


def test_external_waterbody_algae_question_does_not_query_lab_strain_database():
    decision = _decision("自然河流中常见微藻有哪些")

    assert isinstance(decision, KnowledgeDecision)
    assert decision.kind == RouteKind.KNOWLEDGE_QUERY


def test_internal_scope_still_queries_lab_strain_database():
    decision = _decision("当前数据库中有哪些藻种")

    assert decision.kind == RouteKind.LAB_QUERY
