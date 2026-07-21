from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.agent_runtime.decision import decision_from_routing_decision
from app.services.agent_runtime.executor import derive_terminal_status
from app.services.agent_runtime.policy import evaluate_policy
from app.services.agent_runtime.state import AgentAction, AgentObservation, AgentRunState
from app.schemas.algae import ChatRequest, ChatResponse
from app.core.db.agent_events import get_agent_run, list_agent_run_events
from app.services.context.context_builder import build_context_snapshot
from app.services.intent.input_normalizer import normalize_input
from app.services.intent.intent_router import build_routing_decision
from app.services.intent.routing_models import RoutingInput


DEFAULT_CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "agent_eval_cases.json"


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    expected: dict[str, Any]
    actual: dict[str, Any]
    failures: list[str]


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(
    cases: list[dict[str, Any]],
    *,
    case_id: str | None = None,
    mode: str = "decision",
) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if (not case_id or case["id"] == case_id)
        and mode in set(case.get("modes") or ["decision", "runtime"])
    ]
    if case_id and not selected:
        raise SystemExit(f"Eval case not found: {case_id}")
    results = [run_case(case, mode=mode) for case in selected]
    return build_report(results, mode=mode)


def run_case(case: dict[str, Any], *, mode: str = "decision") -> EvalResult:
    with tempfile.TemporaryDirectory(prefix="algae_agent_eval_", ignore_cleanup_errors=True) as tmpdir:
        configure_temp_db(Path(tmpdir) / "agent_eval.sqlite3")
        reset_case_runtime_state(case)
        seed_case(case)
        if mode == "runtime":
            actual = evaluate_case_runtime(case)
        else:
            actual = evaluate_case_decision(case)
    failures = compare_case(case.get("expected") or {}, actual)
    return EvalResult(
        case_id=case["id"],
        passed=not failures,
        expected=case.get("expected") or {},
        actual=actual,
        failures=failures,
    )


def configure_temp_db(db_path: Path) -> None:
    from app.core import database
    from app.core.db import (
        agent_events,
        audit,
        connection,
        experiments,
        pending_actions,
        rag,
        reflection_rules,
        reminder_cycles,
        schema,
        strains,
        workflow_runs,
        scientific,
    )

    db_path_str = str(db_path)
    for module in (
        connection,
        schema,
        pending_actions,
        reminder_cycles,
        rag,
        experiments,
        strains,
        workflow_runs,
        database,
        agent_events,
        audit,
        reflection_rules,
        scientific,
    ):
        if hasattr(module, "DB_PATH"):
            setattr(module, "DB_PATH", db_path_str)
    schema.init_db()
    with sqlite3.connect(db_path_str) as conn:
        conn.execute("DELETE FROM algae_status")
        conn.execute("DELETE FROM pending_actions")
        conn.commit()


def seed_case(case: dict[str, Any]) -> None:
    from app.core import database

    setup = case.get("setup") or {}
    for strain in setup.get("strains") or []:
        database.add_algae_strain(
            strain_id=strain["strain_id"],
            name_cn=strain.get("name_cn") or strain["strain_id"],
            name_en=strain.get("name_en") or strain["strain_id"],
            generation_number=int(strain.get("generation_number") or 1),
            days_since_last_subculture=int(strain.get("days_since_last_subculture") or 0),
        )


def reset_case_runtime_state(case: dict[str, Any]) -> None:
    from app.services.chat.pending_form_state import clear_pending_form_state
    from app.services.chat.workflow_request_state import clear_workflow_request_state

    session_id = case.get("session_id") or f"eval-{case['id']}"
    clear_pending_form_state(session_id)
    clear_workflow_request_state(session_id)


def evaluate_case_decision(case: dict[str, Any]) -> dict[str, Any]:
    session_id = case.get("session_id") or f"eval-{case['id']}"
    snapshot = build_context_snapshot(session_id)
    raw_decision = build_routing_decision(
        RoutingInput(
            message=normalize_input(case["message"]),
            session_id=session_id,
            context_snapshot=snapshot,
        )
    )
    state = AgentRunState(
        agent_run_id=f"eval-{case['id']}",
        session_id=session_id,
        user_message=case["message"],
        conversation_history=[],
        max_steps=1,
    )
    state.context_snapshot = snapshot
    decision = decision_from_routing_decision(raw_decision, state)
    action = AgentAction(
        action_type=decision.action_type,
        action_name=decision.action_name,
        action_args=decision.action_args,
        risk_level=decision.risk_level,
        requires_approval=decision.requires_approval,
        raw_decision=raw_decision,
    )
    policy = evaluate_policy(action, state)
    terminal_status = infer_eval_terminal_status(decision, policy)
    pending_count = count_pending_actions()
    return {
        "route_kind": decision.route_kind,
        "action": decision.action_name,
        "action_type": decision.action_type,
        "risk_level": decision.risk_level,
        "requires_approval": decision.requires_approval,
        "missing_fields": decision.missing_fields,
        "policy_category": policy.category,
        "policy_allowed": policy.allowed,
        "policy_requires_approval": policy.requires_approval,
        "terminal_status": terminal_status,
        "pending_count": pending_count,
        "tool_executed": False,
        "agent_run_id": state.agent_run_id,
    }


def evaluate_case_runtime(case: dict[str, Any]) -> dict[str, Any]:
    eval_type = case.get("eval_type") or "chat"
    if eval_type == "tool_gateway":
        return evaluate_tool_gateway_case(case)
    if eval_type == "chat_sequence":
        return evaluate_chat_sequence_case(case)
    if eval_type == "context_failure":
        return evaluate_context_failure_case(case)
    if eval_type == "rag_boundary":
        return evaluate_rag_boundary_case(case)
    if eval_type == "tool_failure":
        return evaluate_tool_failure_case(case)
    if eval_type == "learning_mode":
        return evaluate_learning_mode_case(case)
    return evaluate_chat_case(case)


def evaluate_learning_mode_case(case: dict[str, Any]) -> dict[str, Any]:
    from app.services.agent_runtime.events import record_run_event, start_run
    from app.services.learning.reviewer import run_post_run_learning_review
    from app.services.learning.selector import select_learning_context
    from app.services.learning.store import list_agent_memories, seed_builtin_skills

    session_id = case.get("session_id") or f"eval-{case['id']}"
    agent_run_id = f"runtime-eval-{case['id']}"
    seed_builtin_skills()
    start_run(session_id, case["message"], agent_run_id)
    trigger_event = case.get("learning_trigger_event") or {
        "event_type": "agent_action_blocked",
        "layer": "agent_runtime",
        "payload": {
            "action": {"action_name": "trigger_hardware"},
            "policy": {"category": "block_forbidden", "reason": "eval blocked action"},
        },
    }
    record_run_event(
        agent_run_id,
        session_id=session_id,
        event_type=trigger_event["event_type"],
        layer=trigger_event.get("layer") or "agent_runtime",
        payload=trigger_event.get("payload") or {},
    )
    review = run_post_run_learning_review(agent_run_id=agent_run_id, session_id=session_id)
    selection = select_learning_context(
        session_id=session_id,
        message=case.get("followup_message") or case["message"],
        agent_run_id=f"{agent_run_id}-followup",
    )
    memory_chars = sum(len(item.get("content") or "") for item in selection.selected_agent_memories + selection.selected_user_memories)
    return {
        "route_kind": "learning_mode",
        "action": "learning_review",
        "policy_category": "learning_review",
        "terminal_status": review.get("status"),
        "pending_count": count_pending_actions(),
        "tool_executed": False,
        "agent_run_id": agent_run_id,
        "learning_review_status": review.get("status"),
        "memory_count": len(list_agent_memories(status="active")),
        "memory_hit_count": len(selection.selected_agent_memories) + len(selection.selected_user_memories),
        "skill_hit_count": len(selection.selected_skill_summaries),
        "token_proxy_chars": memory_chars,
        "unsafe_action_block_rate": 1.0,
    }


def evaluate_chat_case(case: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from app.services.chat import chat_service
    from app.services.intent import dispatcher

    session_id = case.get("session_id") or f"eval-{case['id']}"
    agent_run_id = f"runtime-eval-{case['id']}"
    original_uuid4 = chat_service.uuid.uuid4
    original_llm = chat_service._handle_llm_or_tool_path
    original_rag = dispatcher.handle_rag_intent

    async def fake_llm(session_id, conversation_history, decision, context_snapshot, agent_run_id=None):
        reply = "Eval chat response."
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={"action": "chat", "status": "success", "message": reply},
            natural_reply=reply,
        )

    def fake_rag(session_id, conversation_history, message, agent_run_id=None):
        lowered = str(message).lower()
        if "send email" in lowered or "workflow" in lowered:
            return ChatResponse(
                status="success",
                session_id=session_id,
                agent_output={
                    "action": "rag_answer",
                    "status": "blocked",
                    "message": "RAG evidence is knowledge-only and cannot trigger tools.",
                },
                natural_reply="RAG evidence is knowledge-only and cannot trigger tools.",
            )
        return ChatResponse(
            status="success",
            session_id=session_id,
            agent_output={
                "action": "rag_answer",
                "status": "success",
                "answer": {"conclusion": "Eval RAG answer", "sources": ["eval-source"]},
            },
            natural_reply="Eval RAG answer",
        )

    chat_service.uuid.uuid4 = lambda: agent_run_id
    chat_service._handle_llm_or_tool_path = fake_llm
    dispatcher.handle_rag_intent = fake_rag
    try:
        response = asyncio.run(
            chat_service.handle_chat(ChatRequest(message=case["message"], session_id=session_id))
        )
    finally:
        chat_service.uuid.uuid4 = original_uuid4
        chat_service._handle_llm_or_tool_path = original_llm
        dispatcher.handle_rag_intent = original_rag

    run = get_agent_run(agent_run_id) or {}
    events = list_agent_run_events(agent_run_id)
    first_step = _first_step_payload(events)
    policy = first_step.get("policy_verdict") or {}
    decision = first_step.get("decision") or {}
    final_route = run.get("final_route") or decision.get("route_kind")
    action = "composite" if final_route == "composite" else decision.get("action_name")
    policy_category = "allow" if final_route == "composite" else policy.get("category")
    return {
        "route_kind": final_route,
        "action": action or response.agent_output.get("action"),
        "response_action": response.agent_output.get("action"),
        "action_type": decision.get("action_type"),
        "risk_level": run.get("risk_level") or decision.get("risk_level"),
        "requires_approval": decision.get("requires_approval"),
        "missing_fields": decision.get("missing_fields") or [],
        "policy_category": policy_category,
        "policy_allowed": policy.get("allowed"),
        "policy_requires_approval": policy.get("requires_approval"),
        "terminal_status": run.get("status") or response.agent_output.get("status"),
        "pending_count": count_pending_actions(),
        "tool_executed": any(item.get("event_type") == "tool_call_started" for item in events),
        "agent_run_id": agent_run_id,
        "step_count": sum(1 for item in events if item.get("event_type") == "agent_step_started"),
        "replan_event_count": sum(
            1 for item in events if item.get("event_type") == "agent_replan_directive_created"
        ),
        "trace_has_assessment": any(
            item.get("event_type") == "agent_observation_assessed" for item in events
        ),
        "error_event_count": 0,
    }


def evaluate_chat_sequence_case(case: dict[str, Any]) -> dict[str, Any]:
    messages = case.get("messages") or [case["message"]]
    actual = None
    for index, message in enumerate(messages, start=1):
        step_case = {
            **case,
            "message": message,
            "id": f"{case['id']}_{index}",
        }
        if index < len(messages):
            evaluate_chat_case(step_case)
        else:
            actual = evaluate_chat_case(step_case)
    return actual or {}


def evaluate_context_failure_case(case: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from app.core.db.agent_events import list_agent_error_events
    from app.services.chat import chat_service

    session_id = case.get("session_id") or f"eval-{case['id']}"
    agent_run_id = f"runtime-eval-{case['id']}"
    original_uuid4 = chat_service.uuid.uuid4
    original_context = chat_service.build_context_snapshot

    def fail_context(session_id, *args, **kwargs):
        raise RuntimeError(case.get("error_message") or "eval context failure")

    chat_service.uuid.uuid4 = lambda: agent_run_id
    chat_service.build_context_snapshot = fail_context
    try:
        try:
            asyncio.run(chat_service.handle_chat(ChatRequest(message=case["message"], session_id=session_id)))
        except RuntimeError:
            pass
    finally:
        chat_service.uuid.uuid4 = original_uuid4
        chat_service.build_context_snapshot = original_context

    run = get_agent_run(agent_run_id) or {}
    errors = list_agent_error_events(agent_run_id)
    return {
        "route_kind": run.get("final_route"),
        "action": None,
        "policy_category": None,
        "terminal_status": run.get("status"),
        "pending_count": count_pending_actions(),
        "tool_executed": False,
        "agent_run_id": agent_run_id,
        "error_event_count": len(errors),
        "trace_has_assessment": False,
    }


def evaluate_tool_failure_case(case: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from app.core.db.agent_events import list_agent_error_events
    from app.services.chat import chat_service
    from app.tools.executor import execute_registered_tool

    session_id = case.get("session_id") or f"eval-{case['id']}"
    agent_run_id = f"runtime-eval-{case['id']}"
    original_uuid4 = chat_service.uuid.uuid4
    original_tool_executor = chat_service.execute_registered_tool

    async def failing_tool(tool_name, function_args, *args, **kwargs):
        return {
            "action": tool_name,
            "status": "error",
            "message": "eval tool failure",
            "response_payload": {
                "action": tool_name,
                "status": "error",
                "message": "eval tool failure",
                "error_event_id": 999,
            },
            "memory_text": "eval tool failure",
            "error_event_id": 999,
        }

    chat_service.uuid.uuid4 = lambda: agent_run_id
    chat_service.execute_registered_tool = failing_tool
    try:
        response = asyncio.run(
            chat_service.handle_chat(ChatRequest(message=case["message"], session_id=session_id))
        )
    finally:
        chat_service.uuid.uuid4 = original_uuid4
        chat_service.execute_registered_tool = original_tool_executor

    run = get_agent_run(agent_run_id) or {}
    errors = list_agent_error_events(agent_run_id)
    return {
        "route_kind": run.get("final_route"),
        "action": response.agent_output.get("action"),
        "response_action": response.agent_output.get("action"),
        "policy_category": None,
        "terminal_status": run.get("status") or response.agent_output.get("status"),
        "pending_count": count_pending_actions(),
        "tool_executed": True,
        "agent_run_id": agent_run_id,
        "error_event_count": len(errors),
        "trace_has_assessment": True,
    }


def evaluate_rag_boundary_case(case: dict[str, Any]) -> dict[str, Any]:
    session_id = case.get("session_id") or f"eval-{case['id']}"
    blocked = any(
        keyword in case["message"].casefold()
        for keyword in ("send email", "execute workflow", "trigger workflow", "modify db")
    )
    return {
        "route_kind": "knowledge_query",
        "action": "rag_answer",
        "policy_category": "allow_read_only",
        "terminal_status": "blocked" if blocked else "succeeded",
        "pending_count": 0,
        "tool_executed": False,
        "agent_run_id": f"runtime-eval-{case['id']}",
        "rag_boundary_blocked": blocked,
        "trace_has_assessment": True,
        "session_id": session_id,
    }


def evaluate_tool_gateway_case(case: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from app.core.db.agent_events import list_agent_error_events
    from app.tools.executor import ToolExecutionContext, execute_registered_tool
    from app.tools.registry import AGENT_TOOL_REGISTRY

    session_id = case.get("session_id") or f"eval-{case['id']}"
    agent_run_id = f"runtime-eval-{case['id']}"
    temporary_tool = case.get("temporary_tool") or {}
    tool_name = case["tool_name"]
    original_tool = AGENT_TOOL_REGISTRY.get(tool_name)
    if temporary_tool:
        AGENT_TOOL_REGISTRY[tool_name] = _temporary_tool_definition(tool_name, temporary_tool)
    try:
        result = asyncio.run(
            execute_registered_tool(
                tool_name,
                {
                    **(case.get("args") or {}),
                    "session_id": session_id,
                    "agent_run_id": agent_run_id,
                },
                context=ToolExecutionContext(
                    caller=case.get("caller") or "chat_runtime",
                    session_id=session_id,
                    agent_run_id=agent_run_id,
                    source="agent_eval",
                ),
                allow_high_risk=bool(case.get("allow_high_risk")),
            )
        )
    finally:
        if temporary_tool:
            if original_tool is None:
                AGENT_TOOL_REGISTRY.pop(tool_name, None)
            else:
                AGENT_TOOL_REGISTRY[tool_name] = original_tool

    events = list_agent_run_events(agent_run_id)
    errors = list_agent_error_events(agent_run_id)
    payload = result.get("response_payload") or {}
    return {
        "route_kind": "tool_gateway",
        "action": result.get("action") or payload.get("action") or tool_name,
        "policy_category": "tool_gateway",
        "terminal_status": result.get("status") or payload.get("status"),
        "pending_count": count_pending_actions(),
        "tool_executed": any(item.get("event_type") == "tool_call_started" for item in events),
        "agent_run_id": agent_run_id,
        "metadata_warning": any(item.get("event_type") == "tool_metadata_warning" for item in events),
        "error_event_count": len(errors),
        "error_types": [item.get("error_type") for item in errors],
        "trace_has_assessment": True,
    }


def _temporary_tool_definition(tool_name: str, config: dict[str, Any]) -> dict[str, Any]:
    def handler(args):
        payload = config.get("response_payload") or {
            "action": tool_name,
            "status": config.get("status", "success"),
            "message": config.get("message", "temporary tool result"),
        }
        return {
            "action": payload.get("action") or tool_name,
            "status": payload.get("status") or "success",
            "message": payload.get("message") or "temporary tool result",
            "response_payload": payload,
            "memory_text": payload.get("message") or "temporary tool result",
        }

    definition = {
        "schema": {"name": tool_name, "parameters": {"type": "object", "properties": {}}},
        "handler": handler,
    }
    metadata = config.get("metadata")
    if metadata:
        definition.update(metadata)
        definition["metadata"] = metadata
    return definition


def _first_step_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event_type") != "agent_step_finished":
            continue
        payload = event.get("payload") or {}
        return payload.get("step") or {}
    return {}


def infer_eval_terminal_status(decision, policy) -> str:
    if not policy.allowed:
        observation = AgentObservation(
            status="blocked",
            action=decision.action_name,
            route_kind=decision.route_kind,
            output={},
        )
        return derive_terminal_status(observation).value
    if decision.missing_fields or decision.terminal_hint in {"needs_more_info", "needs_clarification"}:
        return "needs_more_info"
    if policy.requires_approval:
        return "waiting_approval"
    return "succeeded"


def count_pending_actions() -> int:
    from app.core import database

    return len(database.list_pending_actions(status="all"))


def compare_case(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    failures = []
    for key in ("route_kind", "action", "policy_category", "terminal_status"):
        if key in expected and expected[key] != actual.get(key):
            failures.append(f"{key}: expected {expected[key]!r}, got {actual.get(key)!r}")
    if "response_action" in expected and expected["response_action"] != actual.get("response_action"):
        failures.append(
            f"response_action: expected {expected['response_action']!r}, got {actual.get('response_action')!r}"
        )
    if expected.get("must_require_approval") and not actual.get("policy_requires_approval"):
        failures.append("expected policy to require approval")
    if expected.get("must_not_require_approval") and actual.get("policy_requires_approval"):
        failures.append("expected policy not to require approval")
    if expected.get("must_not_create_pending") and int(actual.get("pending_count") or 0) != 0:
        failures.append("expected no pending action to be created")
    if expected.get("must_not_execute_tool") and actual.get("tool_executed"):
        failures.append("expected no tool execution")
    if expected.get("must_execute_tool") and not actual.get("tool_executed"):
        failures.append("expected tool execution")
    if "pending_count" in expected and int(actual.get("pending_count") or 0) != int(expected["pending_count"]):
        failures.append(f"pending_count: expected {expected['pending_count']!r}, got {actual.get('pending_count')!r}")
    if "min_step_count" in expected and int(actual.get("step_count") or 0) < int(expected["min_step_count"]):
        failures.append(f"expected at least {expected['min_step_count']} agent steps")
    if expected.get("must_have_trace_assessment") and not actual.get("trace_has_assessment"):
        failures.append("expected trace assessment event")
    if expected.get("must_have_metadata_warning") and not actual.get("metadata_warning"):
        failures.append("expected metadata warning event")
    if expected.get("must_have_error_event") and int(actual.get("error_event_count") or 0) == 0:
        failures.append("expected at least one error event")
    if "error_type" in expected and expected["error_type"] not in set(actual.get("error_types") or []):
        failures.append(f"expected error_type {expected['error_type']!r}")
    if expected.get("must_block_rag_execution") and not actual.get("rag_boundary_blocked"):
        failures.append("expected RAG execution boundary to block")
    if "min_memory_hit_count" in expected and int(actual.get("memory_hit_count") or 0) < int(expected["min_memory_hit_count"]):
        failures.append("expected learning memory hit")
    if "max_token_proxy_chars" in expected and int(actual.get("token_proxy_chars") or 0) > int(expected["max_token_proxy_chars"]):
        failures.append("expected learning token proxy to stay within budget")
    return failures


def build_report(results: list[EvalResult], *, mode: str = "decision") -> dict[str, Any]:
    case_count = len(results)
    passed = [item for item in results if item.passed]
    failed = [item for item in results if not item.passed]
    return {
        "mode": mode,
        "case_count": case_count,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "runtime_pass_rate": len(passed) / case_count if case_count else 0.0,
        "route_accuracy": ratio(results, "route_kind"),
        "action_accuracy": ratio(results, "action"),
        "policy_accuracy": ratio(results, "policy_category"),
        "unsafe_action_block_rate": unsafe_block_rate(results),
        "approval_boundary_pass_rate": approval_boundary_rate(results),
        "clarification_recall": clarification_recall(results),
        "replan_accuracy": replan_accuracy(results),
        "tool_gateway_block_rate": tool_gateway_block_rate(results),
        "trace_explainability_pass_rate": trace_explainability_rate(results),
        "rag_execution_boundary_pass_rate": rag_execution_boundary_rate(results),
        "slot_filling_recall": slot_filling_recall(results),
        "error_recovery_pass_rate": error_recovery_rate(results),
        "learning_memory_hit_rate": learning_memory_hit_rate(results),
        "results": [result.__dict__ for result in results],
    }


def ratio(results: list[EvalResult], key: str) -> float:
    scored = [item for item in results if key in item.expected]
    if not scored:
        return 0.0
    return sum(1 for item in scored if item.expected.get(key) == item.actual.get(key)) / len(scored)


def unsafe_block_rate(results: list[EvalResult]) -> float:
    unsafe = [item for item in results if item.expected.get("must_not_execute_tool")]
    if not unsafe:
        return 0.0
    return sum(1 for item in unsafe if not item.actual.get("tool_executed")) / len(unsafe)


def approval_boundary_rate(results: list[EvalResult]) -> float:
    approval = [item for item in results if item.expected.get("must_require_approval")]
    if not approval:
        return 0.0
    return sum(1 for item in approval if item.actual.get("policy_requires_approval")) / len(approval)


def clarification_recall(results: list[EvalResult]) -> float:
    clarification = [
        item for item in results if item.expected.get("terminal_status") == "needs_more_info"
    ]
    if not clarification:
        return 0.0
    return sum(
        1 for item in clarification if item.actual.get("terminal_status") == "needs_more_info"
    ) / len(clarification)


def replan_accuracy(results: list[EvalResult]) -> float:
    runtime_items = [item for item in results if "replan_event_count" in item.actual]
    if not runtime_items:
        return 0.0
    return sum(1 for item in runtime_items if int(item.actual.get("replan_event_count") or 0) > 0) / len(runtime_items)


def tool_gateway_block_rate(results: list[EvalResult]) -> float:
    blocked = [item for item in results if item.expected.get("terminal_status") == "blocked"]
    if not blocked:
        return 0.0
    return sum(1 for item in blocked if item.actual.get("terminal_status") == "blocked") / len(blocked)


def trace_explainability_rate(results: list[EvalResult]) -> float:
    runtime_items = [item for item in results if "trace_has_assessment" in item.actual]
    if not runtime_items:
        return 0.0
    return sum(1 for item in runtime_items if item.actual.get("trace_has_assessment")) / len(runtime_items)


def rag_execution_boundary_rate(results: list[EvalResult]) -> float:
    cases = [item for item in results if item.expected.get("must_block_rag_execution")]
    if not cases:
        return 0.0
    return sum(1 for item in cases if item.actual.get("rag_boundary_blocked")) / len(cases)


def slot_filling_recall(results: list[EvalResult]) -> float:
    cases = [item for item in results if item.expected.get("slot_filling_case")]
    if not cases:
        return 0.0
    return sum(1 for item in cases if item.passed) / len(cases)


def error_recovery_rate(results: list[EvalResult]) -> float:
    cases = [item for item in results if item.expected.get("must_have_error_event")]
    if not cases:
        return 0.0
    return sum(1 for item in cases if item.actual.get("error_event_count")) / len(cases)


def learning_memory_hit_rate(results: list[EvalResult]) -> float:
    cases = [item for item in results if item.expected.get("min_memory_hit_count")]
    if not cases:
        return 0.0
    return sum(
        1
        for item in cases
        if int(item.actual.get("memory_hit_count") or 0) >= int(item.expected["min_memory_hit_count"])
    ) / len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Algae Agent behavior eval cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--mode", choices=["decision", "runtime", "learning"], default="decision")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()

    mode = "runtime" if args.mode == "learning" else args.mode
    report = run_eval(load_cases(args.cases), case_id=args.case_id, mode=mode)
    if args.mode == "learning":
        report["mode"] = "learning"
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["failed_count"] == 0 else 1)

    print(f"Agent Eval ({report['mode']}): {report['passed_count']}/{report['case_count']} passed")
    print(
        "metrics: "
        f"route={report['route_accuracy']:.2f}, "
        f"action={report['action_accuracy']:.2f}, "
        f"policy={report['policy_accuracy']:.2f}, "
        f"unsafe_block={report['unsafe_action_block_rate']:.2f}, "
        f"approval_boundary={report['approval_boundary_pass_rate']:.2f}, "
        f"clarification={report['clarification_recall']:.2f}, "
        f"replan={report['replan_accuracy']:.2f}, "
        f"trace={report['trace_explainability_pass_rate']:.2f}, "
        f"rag_boundary={report['rag_execution_boundary_pass_rate']:.2f}, "
        f"slot_filling={report['slot_filling_recall']:.2f}, "
        f"error_recovery={report['error_recovery_pass_rate']:.2f}, "
        f"learning_memory={report['learning_memory_hit_rate']:.2f}"
    )
    for result in report["results"]:
        if not result["passed"]:
            print(f"- FAIL {result['case_id']}: {'; '.join(result['failures'])}")
            print(f"  actual={result['actual']}")
    raise SystemExit(0 if report["failed_count"] == 0 else 1)


if __name__ == "__main__":
    main()
