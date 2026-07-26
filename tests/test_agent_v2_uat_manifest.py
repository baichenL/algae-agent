from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_agent_v2_uat import compile_assertions

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "agent_v2_extended_manual_uat_cases.json"
)

SUPPORTED_OPERATIONS = {
    "restart_api",
    "approve_test_email_pending",
    "poll_canonical_state",
    "try_approval",
    "try_queue_claim_resume_execute",
    "submit_same_approval_twice",
    "start_two_worker_claims",
    "approve_stale_pending",
    "list_mcp_tools",
    "inspect_mcp_tool_schemas",
    "call_experiment_proposal_request_via_mcp",
    "verify_external_unknown",
    "verify_ui_artifacts",
}

DEFAULT_CASE_OPERATIONS = {
    "EFFECT-007": "verify_external_unknown",
    "UI-002": "verify_ui_artifacts",
}


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda item: item["id"])
def test_every_extended_uat_case_compiles_to_executable_steps(case):
    steps = list(case.get("steps") or [])
    if case.get("prompt"):
        steps.insert(0, {"prompt": case["prompt"]})
    if not steps and case["id"] in DEFAULT_CASE_OPERATIONS:
        steps.append({"operation": DEFAULT_CASE_OPERATIONS[case["id"]]})

    assert steps, f"{case['id']} has no executable prompt or operation"
    for step in steps:
        assert bool(step.get("prompt")) ^ bool(step.get("operation"))
        if step.get("operation"):
            assert step["operation"] in SUPPORTED_OPERATIONS
    assert case["session_scope"] in {"fresh", "same_session", "two_session", "two_sessions"}
    assert case["risk"] in {"safe", "isolated_external"}
    assert case.get("expected_reply") or case.get("expected_state") or case.get("expected_trace")
    assertions = compile_assertions(case)
    expected_count = sum(
        len(case.get(field) or [])
        for field in ("expected_reply", "expected_state", "expected_trace", "forbidden")
    )
    assert len(assertions) == expected_count
    assert all(
        item.get("source")
        and item.get("path")
        and item.get("operator")
        and item.get("description")
        for item in assertions
    )


def test_extended_uat_manifest_has_exact_release_gate_set():
    cases = _cases()

    assert len(cases) == 63
    assert len({item["id"] for item in cases}) == 63
    assert {"AGENT-005", "AGENT-006", "AGENT-007"} <= {
        item["id"] for item in cases
    }
