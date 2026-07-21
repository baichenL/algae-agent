from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.algae import ChatRequest, ChatResponse
from app.services.chat import chat_service
from app.services.intent import dispatcher
from app.services.observability.agent_trace import build_agent_run_trace
from scripts.run_agent_eval import configure_temp_db, load_cases, run_eval, seed_case


DEMO_SESSION_ID = "project-demo-session"
DEMO_RUN_IDS = [
    "project-demo-list-strains",
    "project-demo-rag-tap",
    "project-demo-workflow",
]


async def _fake_llm(session_id, conversation_history, decision, context_snapshot, agent_run_id=None):
    return ChatResponse(
        status="success",
        session_id=session_id,
        agent_output={"action": "chat", "status": "success", "message": "Demo chat response."},
        natural_reply="Demo chat response.",
    )


def _fake_rag(session_id, conversation_history, message, agent_run_id=None):
    return ChatResponse(
        status="success",
        session_id=session_id,
        agent_output={
            "action": "rag_answer",
            "status": "success",
            "answer": {
                "conclusion": "TAP medium evidence: K2HPO4 amount should be read from the cited SOP.",
                "sources": ["demo_sop_tap_medium"],
            },
        },
        natural_reply="TAP medium evidence: K2HPO4 amount should be read from the cited SOP.",
    )


async def _run_demo_chats() -> list[dict[str, Any]]:
    run_ids = iter(DEMO_RUN_IDS)
    original_uuid4 = chat_service.uuid.uuid4
    original_llm = chat_service._handle_llm_or_tool_path
    original_rag = dispatcher.handle_rag_intent

    def next_demo_uuid():
        try:
            return next(run_ids)
        except StopIteration:
            return original_uuid4()

    chat_service.uuid.uuid4 = next_demo_uuid
    chat_service._handle_llm_or_tool_path = _fake_llm
    dispatcher.handle_rag_intent = _fake_rag
    try:
        steps = []
        for label, message in [
            ("DB-first lab query", "当前有哪些藻种？"),
            ("RAG knowledge query", "TAP 培养基里 K2HPO4 用量是多少？"),
            ("High-risk workflow request", "给 Chlorella_01 执行传代"),
        ]:
            response = await chat_service.handle_chat(
                ChatRequest(message=message, session_id=DEMO_SESSION_ID)
            )
            agent_output = response.agent_output or {}
            steps.append(
                {
                    "label": label,
                    "message": message,
                    "agent_run_id": DEMO_RUN_IDS[len(steps)],
                    "action": agent_output.get("action"),
                    "status": agent_output.get("status", response.status),
                    "pending_id": agent_output.get("pending_id"),
                    "requires_approval": bool(
                        agent_output.get("require_confirmation")
                        or agent_output.get("requires_confirmation")
                    ),
                }
            )
        return steps
    finally:
        chat_service.uuid.uuid4 = original_uuid4
        chat_service._handle_llm_or_tool_path = original_llm
        dispatcher.handle_rag_intent = original_rag


def _seed_demo_data() -> None:
    seed_case(
        {
            "id": "project-demo",
            "setup": {
                "strains": [
                    {
                        "strain_id": "Chlorella_01",
                        "name_cn": "小球藻",
                        "name_en": "Chlorella vulgaris",
                        "generation_number": 14,
                        "days_since_last_subculture": 7,
                    }
                ]
            },
        }
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="algae_agent_project_demo_", ignore_cleanup_errors=True) as tmpdir:
        configure_temp_db(Path(tmpdir) / "project_demo.sqlite3")
        _seed_demo_data()
        steps = asyncio.run(_run_demo_chats())
        workflow_trace = build_agent_run_trace("project-demo-workflow") or {}
        eval_report = run_eval(load_cases(), mode="runtime")

    output = {
        "status": "success" if _demo_succeeded(steps, eval_report, workflow_trace) else "failed",
        "demo_steps": steps,
        "workflow_trace_summary": {
            "agent_run_id": "project-demo-workflow",
            "stop_reason": (workflow_trace.get("trace") or {}).get("stop_reason"),
            "final_status": (workflow_trace.get("trace") or {}).get("final_status"),
            "step_count": len((workflow_trace.get("trace") or {}).get("steps") or []),
            "requires_approval": _workflow_requires_approval(workflow_trace),
        },
        "runtime_eval": {
            "case_count": eval_report["case_count"],
            "passed_count": eval_report["passed_count"],
            "failed_count": eval_report["failed_count"],
            "runtime_pass_rate": eval_report["runtime_pass_rate"],
            "approval_boundary_pass_rate": eval_report["approval_boundary_pass_rate"],
            "trace_explainability_pass_rate": eval_report["trace_explainability_pass_rate"],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    raise SystemExit(0 if output["status"] == "success" else 1)


def _demo_succeeded(
    steps: list[dict[str, Any]],
    eval_report: dict[str, Any],
    workflow_trace: dict[str, Any],
) -> bool:
    workflow_step = next(
        (step for step in steps if step.get("label") == "High-risk workflow request"),
        {},
    )
    return (
        eval_report["failed_count"] == 0
        and workflow_step.get("status") == "pending"
        and bool(workflow_step.get("pending_id"))
        and _workflow_requires_approval(workflow_trace)
    )


def _workflow_requires_approval(trace: dict[str, Any]) -> bool:
    for step in (trace.get("trace") or {}).get("steps") or []:
        policy = step.get("policy_verdict") or {}
        if policy.get("requires_approval"):
            return True
    return False


if __name__ == "__main__":
    main()
