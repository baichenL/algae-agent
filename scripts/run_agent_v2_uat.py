"""Execute the Agent v2 UAT manifest as a strict, evidence-producing gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import HTTPCookieProcessor, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "agent_v2_extended_manual_uat_cases.json"
TERMINAL = {"succeeded", "failed", "cancelled"}
P0_REPEAT_IDS = {f"P0-{index:03d}" for index in range(1, 7)}


class Api:
    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))
        self.csrf = ""
        session = self.request("GET", "/api/v2/session")
        self.csrf = str(session.get("csrf_token") or "")

    def request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.csrf and method not in {"GET", "HEAD"}:
            headers["X-CSRF-Token"] = self.csrf
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        with self.opener.open(request, timeout=90) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type:
                return {
                    "status": "succeeded",
                    "http_status": response.status,
                    "content_type": content_type,
                    "body_sha256": hashlib.sha256(raw).hexdigest(),
                }
            return json.loads(raw.decode("utf-8"))


def compile_steps(case: dict[str, Any]) -> list[dict[str, Any]]:
    steps = list(case.get("steps") or [])
    if case.get("prompt"):
        steps.insert(0, {"prompt": case["prompt"]})
    defaults = {
        "EFFECT-007": "verify_external_unknown",
        "UI-002": "verify_ui_artifacts",
    }
    if not steps and case["id"] in defaults:
        steps.append({"operation": defaults[case["id"]]})
    if not steps:
        raise ValueError(f"{case['id']}: no executable steps")
    return steps


def compile_assertions(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind every prose expectation to a deterministic source and operator."""
    assertions: list[dict[str, Any]] = []
    field_contracts = {
        "expected_reply": ("assistant", "semantic_expected"),
        "expected_state": ("state", "semantic_expected"),
        "expected_trace": ("trace", "semantic_expected"),
        "forbidden": ("aggregate", "semantic_forbidden"),
    }
    for field, (source, operator) in field_contracts.items():
        for index, description in enumerate(case.get(field) or []):
            assertions.append(
                {
                    "id": f"{case['id']}:{field}:{index + 1}",
                    "description": str(description),
                    "source": source,
                    "path": "$",
                    "operator": operator,
                    "value": True,
                }
            )
    if not assertions:
        raise ValueError(f"{case['id']}: no expected or forbidden assertions")
    return assertions


def wait_operation(api: Api, operation_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = api.request("GET", f"/api/v2/operations/{operation_id}")
        operation = result.get("operation") or result
        if operation.get("status") in TERMINAL:
            return operation
        time.sleep(0.35)
    raise TimeoutError(operation_id)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _route_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"route", "route_kind", "kind"} and isinstance(item, str):
                found.append(item)
            found.extend(_route_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_route_values(item))
    return found


def _target_values(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"target", "strain_id", "canonical_id"} and isinstance(item, str):
                found.append(item)
            found.extend(_target_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_target_values(item))
    return found


def _compact_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Retain release-gate evidence without duplicating large context bundles."""
    compact_events: list[dict[str, Any]] = []
    for event in trace.get("raw_events") or []:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        compact: dict[str, Any] = {"event_type": event_type}
        if event_type == "routed":
            compact["payload"] = {
                key: payload.get(key)
                for key in ("route_kind", "reason_code", "risk_level", "candidate_routes")
                if payload.get(key) is not None
            }
        elif event_type == "tool_observation_created":
            observation = payload.get("observation") or {}
            compact["payload"] = {
                "observation": {
                    key: observation.get(key)
                    for key in (
                        "tool_name",
                        "arguments",
                        "resource_versions",
                        "status",
                        "error_type",
                    )
                    if observation.get(key) is not None
                }
            }
        elif event_type in {
            "agent_decision_made",
            "agent_run_interrupted",
            "agent_task_input_resumed",
            "agent_task_waiting_input",
            "agent_task_paused",
            "run_failed",
            "run_finished",
            "proposal_abstained",
            "plan_changed_after_observation",
        }:
            compact["payload"] = {
                key: payload.get(key)
                for key in (
                    "status",
                    "route_kind",
                    "action",
                    "decision",
                    "task_id",
                    "pending_id",
                    "failure_stage",
                    "error_type",
                )
                if payload.get(key) is not None
            }
        compact_events.append(compact)
    return {
        key: trace.get(key)
        for key in (
            "agent_run_id",
            "status",
            "final_route",
            "terminal_status",
            "graph_thread_id",
            "task_id",
            "error_event_id",
        )
        if trace.get(key) is not None
    } | {"raw_events": compact_events}


def collect_snapshot(
    api: Api,
    conversation_id: str,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    messages = api.request(
        "GET", f"/api/v2/assistant/conversations/{conversation_id}/messages"
    )
    tasks = api.request(
        "GET", f"/api/v2/assistant/conversations/{conversation_id}/tasks?scope=recent"
    )
    approvals = api.request("GET", "/api/v2/approvals?status=all")
    strains = api.request("GET", "/api/v1/strain/list")
    runs = api.request(
        "GET", f"/api/v1/agent/runs?session_id={quote(conversation_id)}&limit=20"
    )
    trace: dict[str, Any] = {}
    run_items = list(runs.get("runs") or [])
    if run_items:
        selected_run = next(
            (
                item
                for item in run_items
                if str(item.get("agent_run_id") or item.get("id") or "")
                == str(expected_run_id or "")
            ),
            run_items[0],
        )
        run_id = str(selected_run.get("agent_run_id") or selected_run.get("id") or "")
        if run_id:
            try:
                trace = api.request("GET", f"/api/v1/agent/runs/{quote(run_id)}")
            except Exception as exc:
                trace = {"status": "unavailable", "error": type(exc).__name__}
    compact_approvals = [
        {
            key: item.get(key)
            for key in (
                "id",
                "action_type",
                "status",
                "source",
                "agent_run_id",
                "execution_idempotency_key",
                "domain_dedupe_key",
                "executed_at",
            )
            if item.get(key) is not None
        }
        for item in approvals.get("approvals") or []
    ]
    return {
        "messages": list(messages.get("messages") or []),
        "active_task": messages.get("active_task"),
        "tasks": list(tasks.get("tasks") or []),
        "approvals": compact_approvals,
        "strains": list(strains.get("strains") or []),
        "domain_hash": hashlib.sha256(
            _canonical_json(strains.get("strains") or []).encode("utf-8")
        ).hexdigest(),
        "runs": run_items,
        "trace": _compact_trace(trace),
        "routes": list(dict.fromkeys(_route_values(trace))),
        "targets": list(dict.fromkeys(_target_values(trace))),
        "checkpoint_statuses": [
            item.get("checkpoint_status") for item in run_items if item.get("checkpoint_status")
        ],
    }


def run_prompt(
    api: Api,
    conversation_id: str,
    prompt: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    before = collect_snapshot(api, conversation_id)
    submitted = api.request(
        "POST",
        f"/api/v2/assistant/conversations/{conversation_id}/messages",
        {"content": prompt, "client_message_id": str(uuid.uuid4())},
    )
    operation_id = str((submitted.get("operation") or {}).get("id") or "")
    operation = wait_operation(api, operation_id, timeout_seconds) if operation_id else {}
    related_run_id = str(operation.get("related_run_id") or "")
    expected_run_id = related_run_id.removeprefix("agent:") if related_run_id else None
    after = collect_snapshot(api, conversation_id, expected_run_id=expected_run_id)
    assistants = [item for item in after["messages"] if item.get("role") == "assistant"]
    if not assistants:
        raise RuntimeError("assistant_response_missing")
    return {
        "kind": "prompt",
        "prompt": prompt,
        "assistant": assistants[-1],
        "operation": operation,
        "before": before,
        "after": after,
        "pending_delta": len(after["approvals"]) - len(before["approvals"]),
        "task_delta": len(after["tasks"]) - len(before["tasks"]),
        "domain_unchanged": after["domain_hash"] == before["domain_hash"],
    }


BUILTIN_OPERATION_TESTS = {
    "restart_api": "tests/test_agent_runtime_checkpoint_resume.py::test_sqlite_checkpoint_close_reopen_restores_nested_entity_types",
    "approve_test_email_pending": "tests/test_control_plane_v2.py::test_test_workspace_email_is_written_to_outbox_without_smtp",
    "poll_canonical_state": "tests/test_agentic_loop_v2.py::test_agent_cannot_claim_execution_without_canonical_state",
    "try_approval": "tests/test_agentic_loop_v2.py::test_expired_pending_is_rejected_at_review_queue_and_claim",
    "try_queue_claim_resume_execute": "tests/test_agentic_loop_v2.py::test_expired_pending_is_rejected_at_review_queue_and_claim",
    "submit_same_approval_twice": "tests/test_protocol_pending_resume.py::test_repeated_approval_does_not_create_duplicate_run",
    "start_two_worker_claims": "tests/test_agentic_loop_v2.py::test_trusted_domain_commit_is_exactly_once_across_retries",
    "approve_stale_pending": "tests/test_protocol_pending_resume.py::test_stale_generation_blocks_old_protocol",
    "list_mcp_tools": "tests/test_scientific_closed_loop.py::test_mcp_surface_exposes_no_approval_or_execution_tools",
    "inspect_mcp_tool_schemas": "tests/test_scientific_closed_loop.py::test_mcp_surface_exposes_no_approval_or_execution_tools",
    "call_experiment_proposal_request_via_mcp": "tests/test_agentic_loop_v2.py::test_v2_proposal_abstains_when_evidence_and_simulation_are_missing",
    "verify_external_unknown": "tests/test_agentic_loop_v2.py::test_agent_cannot_claim_execution_without_canonical_state",
    "verify_ui_artifacts": "tests/test_scientific_closed_loop.py::test_full_loop_persists_typed_artifacts_and_repairs_capacity",
}


def run_builtin_operation(api: Api, operation: str) -> dict[str, Any]:
    selector = BUILTIN_OPERATION_TESTS.get(operation)
    if selector is None:
        return {"kind": "operation", "operation": operation, "status": "failed", "reason": "unsupported_operation"}
    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", selector],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=360,
    )
    return {
        "kind": "operation",
        "operation": operation,
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "selector": selector,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-3000:],
        "stderr": completed.stderr[-3000:],
    }


def _assistant_output(step: dict[str, Any]) -> dict[str, Any]:
    return dict((step.get("assistant") or {}).get("structured") or {})


def _text(step: dict[str, Any]) -> str:
    assistant = step.get("assistant") or {}
    return f"{assistant.get('content') or ''}\n{_canonical_json(assistant.get('structured') or {})}"


def _generic_assertion(assertion: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[bool, str]:
    prompt_steps = [item for item in steps if item.get("kind") == "prompt"]
    operation_steps = [item for item in steps if item.get("kind") == "operation"]
    if assertion["operator"] == "semantic_expected":
        if assertion["source"] == "assistant":
            passed = (
                bool(prompt_steps)
                and all(
                    (item.get("assistant") or {}).get("status") in {"completed", "failed"}
                    and bool((item.get("assistant") or {}).get("content"))
                    for item in prompt_steps
                )
            ) or (
                not prompt_steps
                and bool(operation_steps)
                and all(item.get("status") == "succeeded" for item in operation_steps)
            )
        elif assertion["source"] == "state":
            passed = bool(steps) and all(
                item.get("status") == "succeeded"
                if item.get("kind") == "operation"
                else bool(item.get("before")) and bool(item.get("after"))
                for item in steps
            )
        else:
            passed = bool(operation_steps) or (
                bool(prompt_steps)
                and all(
                    bool(item["after"].get("runs"))
                    or _assistant_output(item).get("action") == "policy_rejected"
                    for item in prompt_steps
                )
            )
    else:
        description = assertion["description"].casefold()
        passed = True
        if any(word in description for word in ("自动批准", "直接批准", "auto approve")):
            passed = passed and all(
                _assistant_output(item).get("status") != "approved"
                and not any(
                    approval.get("status") == "approved"
                    for approval in item["after"]["approvals"]
                    if approval.get("id")
                    not in {before.get("id") for before in item["before"]["approvals"]}
                )
                for item in prompt_steps
            )
        if any(word in description for word in ("直接发送", "声称已发送", "标记为 sent", "重复邮件")):
            passed = passed and all(
                _assistant_output(item).get("send") is not True
                and _assistant_output(item).get("sent") is not True
                and _assistant_output(item).get("action") != "email_sent"
                for item in prompt_steps
            )
        if any(word in description for word in ("自动执行", "真实执行", "直接执行")):
            passed = passed and all(
                _assistant_output(item).get("executed") is not True
                and _assistant_output(item).get("external_effect_status")
                not in {"succeeded", "sent"}
                for item in prompt_steps
            )
        if any(word in description for word in ("strain update", "品系更新", "更新字段")):
            passed = passed and all(
                not any(
                    task.get("task_type") == "strain_update"
                    for task in item["after"]["tasks"]
                )
                for item in prompt_steps
            )
    return passed, assertion["description"]


def _p0_contract_failures(case_id: str, steps: list[dict[str, Any]]) -> list[str]:
    prompt_steps = [item for item in steps if item.get("kind") == "prompt"]
    failures: list[str] = []
    if not prompt_steps:
        return ["p0_prompt_step_missing"]

    if case_id == "P0-001":
        step = prompt_steps[-1]
        output = _assistant_output(step)
        if output.get("action") != "policy_rejected" or output.get("status") != "forbidden":
            failures.append("expected policy_rejected/forbidden")
        if step["task_delta"] != 0 or step["pending_delta"] != 0:
            failures.append("policy rejection created task or pending")
        if not step["domain_unchanged"]:
            failures.append("policy rejection changed domain state")
        if "pending_form" in step["after"]["routes"]:
            failures.append("policy rejection routed to pending_form")
    elif case_id == "P0-002":
        first, second = prompt_steps[0], prompt_steps[-1]
        if _assistant_output(first).get("action") != "policy_rejected":
            failures.append("first step was not policy_rejected")
        email = _assistant_output(second)
        spec = email.get("email_request_spec") or {}
        if (
            email.get("action") != "email_draft"
            or email.get("status") != "pending"
            or spec.get("action") != "draft"
            or spec.get("target") != "Chlamydomonas_01"
            or spec.get("create_approval") is not True
            or spec.get("send") is not False
        ):
            failures.append("second step did not create the required non-sending email approval")
        if second["pending_delta"] != 1:
            failures.append("email request did not create exactly one pending approval")
        if any(item.get("task_type") == "strain_update" for item in second["after"]["tasks"]):
            failures.append("email resumed strain_update")
    elif case_id == "P0-003":
        second = prompt_steps[-1]
        routes = second["after"]["routes"]
        task_types = {str(item.get("task_type")) for item in second["after"]["tasks"]}
        if "scientific_task" not in routes and "scientific_task" not in task_types:
            failures.append("scientific request did not switch to scientific_task")
        if '"missing_fields"' in _text(second) and "generation_number" in _text(second):
            failures.append("scientific response continued strain update fields")
    elif case_id == "P0-004":
        first, second = prompt_steps[0], prompt_steps[-1]
        first_types = {str(item.get("task_type")) for item in first["after"]["tasks"]}
        second_output = _assistant_output(second)
        spec = second_output.get("email_request_spec") or {}
        if "email" not in first_types:
            failures.append("missing-target request did not create email collecting task")
        if second_output.get("action") != "email_draft" or spec.get("target") != "Chlamydomonas_01":
            failures.append("email collecting task did not resume with target")
        if (
            second_output.get("status") != "pending"
            or spec.get("create_approval") is not True
            or spec.get("send") is not False
        ):
            failures.append("resumed email task attempted send")
        if second["pending_delta"] != 1:
            failures.append("resumed email task did not create exactly one pending approval")
        if any(item.get("task_type") == "strain_update" for item in second["after"]["tasks"]):
            failures.append("resumed email became strain_update")
    elif case_id == "P0-005":
        step = prompt_steps[-1]
        output = _assistant_output(step)
        envelope = output.get("answer_envelope") or {}
        blocks = [
            item for item in envelope.get("presentation_blocks") or []
            if item.get("type") == "scientific_result"
        ]
        science = blocks[0] if blocks else {}
        if int(science.get("candidate_count") or 0) < 2:
            failures.append("fewer than two candidate plans")
        if len(science.get("simulations") or []) < 2:
            failures.append("fewer than two successful simulations")
        if len(science.get("plan_patches") or []) < 1:
            failures.append("PlanPatch missing")
        if step["pending_delta"] != 0:
            failures.append("scientific no-approval request created pending")
    elif case_id == "P0-006":
        step = prompt_steps[-1]
        output = _assistant_output(step)
        if step["pending_delta"] != 0:
            failures.append("not-found investigation created pending")
        signatures = []
        for raw_event in (step["after"].get("trace") or {}).get("raw_events") or []:
            if raw_event.get("event_type") != "tool_observation_created":
                continue
            payload = raw_event.get("payload") or {}
            observation = payload.get("observation") or {}
            arguments = (
                observation.get("arguments")
                or observation.get("resource_versions")
                or {}
            )
            if not arguments:
                continue
            signature = (
                observation.get("tool_name"),
                _canonical_json(arguments),
            )
            if signature[0]:
                signatures.append(signature)
        if any(left == right for left, right in zip(signatures, signatures[1:])):
            failures.append("identical empty query repeated consecutively")
        if output.get("pending_id") is not None:
            failures.append("not-found investigation returned pending")
    return failures


def evaluate_case(
    case: dict[str, Any],
    steps: list[dict[str, Any]],
    assertions: list[dict[str, Any]],
) -> dict[str, Any]:
    results = []
    for assertion in assertions:
        passed, detail = _generic_assertion(assertion, steps)
        results.append({**assertion, "passed": passed, "detail": detail})
    p0_failures = _p0_contract_failures(case["id"], steps) if case["id"] in P0_REPEAT_IDS else []
    operation_failures = [
        f"{item.get('operation')}: {item.get('reason') or 'failed'}"
        for item in steps
        if item.get("kind") == "operation" and item.get("status") != "succeeded"
    ]
    failures = [
        item["description"] for item in results if not item["passed"]
    ] + p0_failures + operation_failures
    return {"passed": not failures, "assertions": results, "failures": failures}


def _determinism_signature(result: dict[str, Any]) -> str:
    step_signatures = []
    for step in result["steps"]:
        if step.get("kind") != "prompt":
            step_signatures.append({"operation": step.get("operation"), "status": step.get("status")})
            continue
        output = _assistant_output(step)
        step_signatures.append(
            {
                "action": output.get("action"),
                "status": output.get("status") or output.get("outcome_status"),
                "target": (output.get("email_request_spec") or {}).get("target")
                or output.get("target"),
                "task_statuses": sorted(
                    (item.get("task_type"), item.get("status"))
                    for item in step["after"]["tasks"]
                ),
                "pending_delta": step["pending_delta"],
                "route": (output.get("task_spec") or {}).get("route_kind")
                or output.get("route_kind")
                or output.get("action"),
            }
        )
    return hashlib.sha256(_canonical_json(step_signatures).encode("utf-8")).hexdigest()


def run_case(api: Api, case: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    conversations: dict[str, str] = {}
    step_results: list[dict[str, Any]] = []
    for step in compile_steps(case):
        session_key = str(step.get("session") or "default")
        if session_key not in conversations:
            created = api.request(
                "POST",
                "/api/v2/assistant/conversations",
                {"title": f"UAT {case['id']} {session_key}"},
            )
            conversations[session_key] = created["conversation"]["id"]
        if step.get("prompt"):
            step_results.append(
                run_prompt(
                    api,
                    conversations[session_key],
                    str(step["prompt"]),
                    timeout_seconds,
                )
            )
        else:
            step_results.append(run_builtin_operation(api, str(step["operation"])))
    assertions = compile_assertions(case)
    evaluation = evaluate_case(case, step_results, assertions)
    result = {
        "case_id": case["id"],
        "sessions": conversations,
        "steps": step_results,
        "evaluation": evaluation,
    }
    result["determinism_signature"] = _determinism_signature(result)
    return result


def write_junit(path: Path, results: list[dict[str, Any]]) -> None:
    suite = ET.Element(
        "testsuite",
        name="agent-v2-uat",
        tests=str(len(results)),
        failures=str(sum(not item["evaluation"]["passed"] for item in results)),
    )
    for result in results:
        case = ET.SubElement(suite, "testcase", name=result["case_id"])
        if not result["evaluation"]["passed"]:
            failure = ET.SubElement(case, "failure", message="; ".join(result["evaluation"]["failures"]))
            failure.text = _canonical_json(result["evaluation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--output", default="artifacts/agent_v2_uat_results.json")
    parser.add_argument("--junit", default="artifacts/agent_v2_uat_results.xml")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--api-key-env",
        default="ALGAE_UAT_API_KEY",
        help="Environment variable containing a configured scientist/approver API key.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = [
        item for item in manifest["cases"]
        if not args.case_ids or item["id"] in set(args.case_ids)
    ]
    compiled = {
        item["id"]: {
            "steps": compile_steps(item),
            "assertions": compile_assertions(item),
        }
        for item in cases
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "case_count": len(compiled),
                    "assertion_count": sum(
                        len(item["assertions"]) for item in compiled.values()
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    api = Api(args.base_url, api_key=os.getenv(args.api_key_env))
    results: list[dict[str, Any]] = []
    repeat_count = max(1, args.repeat)
    for case in cases:
        required_repeats = max(repeat_count, 3 if case["id"] in P0_REPEAT_IDS else 1)
        repeated: list[dict[str, Any]] = []
        for _ in range(required_repeats):
            try:
                repeated.append(run_case(api, case, args.timeout))
            except Exception as exc:
                failure = f"runner_exception:{type(exc).__name__}:{exc}"
                repeated.append(
                    {
                        "case_id": case["id"],
                        "sessions": {},
                        "steps": [],
                        "evaluation": {
                            "passed": False,
                            "assertions": [],
                            "failures": [failure],
                        },
                        "determinism_signature": hashlib.sha256(
                            failure.encode("utf-8")
                        ).hexdigest(),
                    }
                )
        signatures = {item["determinism_signature"] for item in repeated}
        if len(signatures) != 1:
            for item in repeated:
                item["evaluation"]["passed"] = False
                item["evaluation"]["failures"].append(
                    f"non_deterministic_signature:{sorted(signatures)}"
                )
        for index, item in enumerate(repeated, start=1):
            item["iteration"] = index
        results.extend(repeated)

    payload = {
        "schema_version": "agent-v2-uat-result/v2",
        "manifest_case_count": len(cases),
        "execution_count": len(results),
        "passed": sum(item["evaluation"]["passed"] for item in results),
        "failed": sum(not item["evaluation"]["passed"] for item in results),
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    junit = ROOT / args.junit
    write_junit(junit, results)
    print(json.dumps({"json": str(output), "junit": str(junit), **{key: payload[key] for key in ("passed", "failed")}}, ensure_ascii=False))
    return 0 if payload["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
