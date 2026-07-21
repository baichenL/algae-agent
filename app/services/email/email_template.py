import datetime
from typing import Any, Dict, List

from app.models.email_schema import EmailDraft


def build_manual_check_draft(message: str, recipients: List[str], metadata: Dict[str, Any]) -> EmailDraft:
    target = metadata.get("strain_id") or "微藻样品"
    subject = f"[Algae Agent] 检查提醒：{target}"
    body = "\n".join([
        "这是来自 Algae Agent 的人工检查提醒。",
        "",
        f"提醒对象: {target}",
        f"用户指令: {message}",
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "请查看实验室培养状态、培养天数、OD750 和最近记录。",
    ])
    return EmailDraft(
        subject=subject,
        body=body,
        recipients=recipients,
        template_type="manual_check",
        metadata=metadata,
        requires_confirmation=True,
    )


def build_passage_reminder_draft(strains: List[Dict[str, Any]], recipients: List[str], threshold_days: int) -> EmailDraft:
    subject = f"[Algae Agent] {len(strains)} 个品系达到 {threshold_days} 天传代阈值"
    lines = [
        "以下微藻品系已达到传代提醒阈值。",
        "请进入 Algae Agent 系统查看状态，并在确认安全后执行传代流程。",
        "",
        f"提醒阈值: 培养天数 >= {threshold_days} 天",
        "安全边界: 本邮件只提醒，不会自动执行硬件 workflow 或写入传代结果。",
        "",
    ]
    for strain in strains:
        lines.append(
            f"- {strain.get('strain_id')} | {strain.get('name_cn')} ({strain.get('name_en')}) | "
            f"代数={strain.get('generation_number')} | "
            f"培养天数={strain.get('days_since_last_subculture')} | "
            f"最后传代={strain.get('last_subculture_time') or '未知'}"
        )
    return EmailDraft(
        subject=subject,
        body="\n".join(lines),
        recipients=recipients,
        template_type="passage_reminder",
        metadata={"threshold_days": threshold_days, "count": len(strains)},
        requires_confirmation=False,
    )


def build_od750_alert_draft(records: List[Dict[str, Any]], recipients: List[str], threshold: float) -> EmailDraft:
    return EmailDraft(
        subject=f"[Algae Agent] OD750 超阈值提醒",
        body=f"检测到 {len(records)} 条 OD750 记录超过阈值 {threshold}。请检查实验记录。",
        recipients=recipients,
        template_type="od750_alert",
        metadata={"threshold": threshold, "count": len(records)},
        requires_confirmation=False,
    )


def build_status_check_draft(records: List[Dict[str, Any]], recipients: List[str], stale_days: int) -> EmailDraft:
    return EmailDraft(
        subject="[Algae Agent] 培养记录久未更新提醒",
        body=f"检测到 {len(records)} 条记录超过 {stale_days} 天未更新。请检查实验记录。",
        recipients=recipients,
        template_type="status_check",
        metadata={"stale_days": stale_days, "count": len(records)},
        requires_confirmation=False,
    )
