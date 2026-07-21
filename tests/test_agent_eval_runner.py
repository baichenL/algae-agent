import json
from pathlib import Path

from scripts.run_agent_eval import load_cases, run_eval


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_agent_eval_runner_loads_default_cases():
    cases = load_cases()

    assert len(cases) >= 30
    assert {case["id"] for case in cases} >= {
        "workflow_missing_strain_requires_clarification",
        "workflow_with_target_requires_approval",
        "tap_recipe_routes_to_rag",
        "tool_gateway_blocks_rag_runtime_write",
        "rag_execute_workflow_instruction_is_blocked",
        "pending_form_slot_fill_creates_pending",
    }


def test_agent_eval_runner_reports_metrics_for_single_case():
    cases = load_cases()
    report = run_eval(cases, case_id="workflow_missing_strain_requires_clarification")

    assert report["case_count"] == 1
    assert report["passed_count"] == 1
    assert report["route_accuracy"] == 1.0
    assert report["unsafe_action_block_rate"] == 1.0
    result = report["results"][0]
    assert result["actual"]["route_kind"] == "clarification"
    assert result["actual"]["tool_executed"] is False


def test_agent_eval_runner_full_suite_passes():
    report = run_eval(load_cases())

    assert report["failed_count"] == 0
    assert report["policy_accuracy"] == 1.0
    assert report["approval_boundary_pass_rate"] == 1.0
    assert report["clarification_recall"] == 1.0


def test_agent_eval_runner_runtime_suite_covers_safety_boundaries():
    report = run_eval(load_cases(), mode="runtime")

    assert report["case_count"] >= 30
    assert report["failed_count"] == 0
    assert report["tool_gateway_block_rate"] == 1.0
    assert report["rag_execution_boundary_pass_rate"] == 1.0
    assert report["slot_filling_recall"] == 1.0
    assert report["error_recovery_pass_rate"] == 1.0


def test_agent_eval_runner_real_zh_runtime_suite_passes():
    cases = load_cases(FIXTURES_DIR / "agent_eval_cases_real_zh.json")
    report = run_eval(cases, mode="runtime")

    assert report["case_count"] >= 10
    assert report["failed_count"] == 0
    assert report["approval_boundary_pass_rate"] == 1.0
    assert report["slot_filling_recall"] == 1.0


def test_rag_real_retrieval_fixture_documents_source_and_authority_boundary():
    cases = json.loads((FIXTURES_DIR / "rag_real_retrieval_cases.json").read_text(encoding="utf-8"))

    assert {case["id"] for case in cases} == {
        "tap_recipe_component_amount",
        "manual_protocol_subculture_steps",
        "experiment_table_field_lookup",
        "paper_protocol_background_only",
    }
    for case in cases:
        assert case["fixture_sources"]
        assert case["expected"]["must_cite_source"] is True
        assert any(
            key in case["expected"]
            for key in (
                "must_not_trigger_tool",
                "must_not_trigger_workflow",
                "must_not_use_as_current_db_fact",
                "must_not_claim_current_lab_state",
            )
        )
