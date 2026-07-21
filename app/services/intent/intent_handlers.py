from typing import Any, Dict

from app.services.intent.routing_models import QueryDecision
from app.services.notifications.notification_service import get_subculture_threshold_days


def build_tool_info_response(available_tools: list[dict]) -> Dict[str, Any]:
    safe_tool_names = [
        item.get("function", {}).get("name")
        for item in available_tools
        if item.get("function", {}).get("name")
    ]
    natural_reply = (
        "我可以使用一些工具，但只有在你明确提出对应任务时才会调用。"
        "如果你只是询问工具能力，我会只说明，不会查询数据库或执行操作。\n\n"
        "当前普通聊天路径可用工具："
        f"{', '.join(safe_tool_names) if safe_tool_names else '暂无'}。\n\n"
        "边界说明：查询品系/待确认请求需要你明确说‘查询、查看、列出’；"
        "新增、更新、删除品系只会创建 pending 请求，必须人工确认；"
        "传代 workflow 属于高风险操作，必须明确品系和执行意图；"
        "邮件默认只生成草稿，确认后才发送。"
    )
    return {
        "agent_output": {
            "action": "tool_info",
            "status": "success",
            "tools": safe_tool_names,
        },
        "natural_reply": natural_reply,
    }


def handle_query_status_intent(
    decision: QueryDecision,
    context_snapshot: Any,
) -> Dict[str, Any]:
    query_type = decision.query_type
    strain_id = decision.target.canonical_id if decision.target else None

    if len(decision.targets) > 1:
        target_ids = [item.canonical_id for item in decision.targets if item.canonical_id]
        rows = [
            item for item in context_snapshot.strains
            if item.get("strain_id") in target_ids
        ]
        row_by_id = {item.get("strain_id"): item for item in rows}
        ordered_rows = [row_by_id[item] for item in target_ids if item in row_by_id]
        if query_type == "generation_number":
            lines = [
                f"{item.get('strain_id')}: generation_number={item.get('generation_number')}"
                for item in ordered_rows
            ]
        else:
            lines = [
                f"{item.get('strain_id')}: generation_number={item.get('generation_number')}，"
                f"days_since_last_subculture={item.get('days_since_last_subculture')}"
                for item in ordered_rows
            ]
        return {
            "agent_output": {
                "action": "query_strains",
                "status": "success",
                "query_type": query_type,
                "strain_ids": target_ids,
                "strains": ordered_rows,
            },
            "natural_reply": "数据库查询：\n" + "\n".join(f"- {line}" for line in lines),
        }

    if strain_id:
        status_data = next(
            (item for item in context_snapshot.strains if item.get("strain_id") == strain_id),
            None,
        )
        if not status_data:
            natural_reply = f"数据库中未找到 {strain_id}"
        elif query_type == "generation_number":
            natural_reply = (
                f"数据库查询：{strain_id} 当前 generation_number = "
                f"{status_data.get('generation_number')}。"
            )
        elif query_type == "pending":
            target_pending = getattr(context_snapshot, "target_pending_actions", []) or []
            natural_reply = f"数据库查询：{strain_id} 当前有 {len(target_pending)} 条待确认 pending 请求。"
        elif query_type == "reminder_status":
            cycle = status_data.get("reminder_cycle") or {}
            if cycle:
                natural_reply = (
                    f"数据库查询：{strain_id} 当前提醒周期状态为 {cycle.get('status')}，"
                    f"cycle_key={cycle.get('cycle_key')}，sent_count={cycle.get('total_sent_count')}。"
                )
            else:
                natural_reply = f"数据库查询：{strain_id} 当前没有提醒周期记录。"
        else:
            natural_reply = (
                f"数据库查询：{strain_id} 当前状态："
                f"generation_number={status_data.get('generation_number')}，"
                f"days_since_last_subculture={status_data.get('days_since_last_subculture')}。"
            )
        return {
            "agent_output": {
                "action": "query_strain",
                "status": "success" if status_data else "not_found",
                "strain_id": strain_id,
                "query_type": query_type,
                "data": status_data,
                "pending": getattr(context_snapshot, "target_pending_actions", []) or [],
            },
            "natural_reply": natural_reply,
        }

    if query_type == "list_strains":
        return {
            "agent_output": {
                "action": "list_strains",
                "status": "success",
                "strains": context_snapshot.strains,
            },
            "natural_reply": f"当前数据库中共有 {len(context_snapshot.strains)} 个品系。",
        }

    if query_type == "pending":
        return {
            "agent_output": {
                "action": "list_pending",
                "status": "success",
                "pending": context_snapshot.pending_actions,
            },
            "natural_reply": f"当前有 {len(context_snapshot.pending_actions)} 条待确认请求。",
        }

    threshold = get_subculture_threshold_days()
    due_strains = [
        item
        for item in context_snapshot.strains
        if int(item.get("days_since_last_subculture") or 0) >= threshold
    ]
    if query_type == "due_subculture":
        return {
            "agent_output": {
                "action": "query_due_subculture",
                "status": "success",
                "threshold_days": threshold,
                "strains": due_strains,
            },
            "natural_reply": f"当前有 {len(due_strains)} 个品系达到临近传代提醒阈值。",
        }

    return {
        "agent_output": {
            "action": "list_strains",
            "status": "success",
            "strains": context_snapshot.strains,
        },
        "natural_reply": f"当前数据库中共有 {len(context_snapshot.strains)} 个品系。",
    }
