# app/tools/strain_crud.py
# 负责把工具参数整理成业务 payload，然后调用 service 创建 pending 请求，让 tool 层不用直接关心 service 的具体参数格式
"""Pending-action adapters for strain CRUD tools.

These helpers are called by tool handlers and never write directly to the main
strain table. They create human-in-the-loop pending actions via the service
layer so approval remains mandatory.
"""

from typing import Any, Dict

from app.services.strains import strain_service


def create_add_strain_request(args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "strain_id": args.get("strain_id"),
        "name_cn": args.get("name_cn"),
        "name_en": args.get("name_en"),
        "generation_number": int(args.get("generation_number", 1)),
        "days_since_last_subculture": int(args.get("days_since_last_subculture", 0)),
    }
    pending_id = strain_service.create_pending_add_strain(
        payload,
        requester="LLM",
        session_id=args.get("session_id"),
        agent_run_id=args.get("agent_run_id"),
        source_message=args.get("source_message"),
        graph_thread_id=args.get("graph_thread_id"),
        execution_idempotency_key=args.get("execution_idempotency_key"),
        domain_dedupe_key=args.get("domain_dedupe_key"),
    )
    return {"status": "pending", "pending_id": pending_id, "msg": f"Add strain request created, pending_id={pending_id}"}


def create_update_strain_request(args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"strain_id": args.get("strain_id")}
    for key in ["new_strain_id", "name_cn", "name_en"]:
        if args.get(key) is not None:
            payload[key] = args.get(key)
    for key in ["generation_number", "days_since_last_subculture"]:
        if args.get(key) is not None:
            payload[key] = int(args.get(key))
    pending_id = strain_service.create_pending_update_strain(
        payload,
        requester="LLM_update",
        session_id=args.get("session_id"),
        agent_run_id=args.get("agent_run_id"),
        source_message=args.get("source_message"),
        graph_thread_id=args.get("graph_thread_id"),
        execution_idempotency_key=args.get("execution_idempotency_key"),
        domain_dedupe_key=args.get("domain_dedupe_key"),
    )
    return {"status": "pending", "pending_id": pending_id, "msg": f"Update strain request created, pending_id={pending_id}"}


def create_delete_strain_request(args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"strain_id": args.get("strain_id")}
    pending_id = strain_service.create_pending_delete_strain(
        payload["strain_id"],
        requester="LLM_delete",
        session_id=args.get("session_id"),
        agent_run_id=args.get("agent_run_id"),
        source_message=args.get("source_message"),
        graph_thread_id=args.get("graph_thread_id"),
        execution_idempotency_key=args.get("execution_idempotency_key"),
        domain_dedupe_key=args.get("domain_dedupe_key"),
    )
    return {"status": "pending", "pending_id": pending_id, "msg": f"Delete strain request created, pending_id={pending_id}"}
