import datetime
import os
from typing import Any, Dict, List

from app.core.database import list_algae_status
from app.services.email.email_notifier import send_email


REMINDER_TYPE_SUBCULTURE_DUE = "subculture_due"
REMINDER_TYPE_EXPERIMENT_FAILED = "experiment_failed"
REMINDER_TYPE_REFLECTION_ANOMALY = "reflection_anomaly"
REMINDER_TYPE_PENDING_APPROVAL_TIMEOUT = "pending_approval_timeout"


def get_reminder_threshold_days() -> int:
    return _int_env("REMINDER_THRESHOLD_DAYS", 5)


def get_reminder_check_interval_seconds() -> int:
    return _int_env("REMINDER_CHECK_INTERVAL_SECONDS", 86400)


def get_due_subculture_strains(threshold_days: int | None = None) -> List[Dict[str, Any]]:
    threshold = threshold_days if threshold_days is not None else get_reminder_threshold_days()
    due_strains = []
    for strain in list_algae_status():
        days = int(strain.get("days_since_last_subculture") or 0)
        if days >= threshold:
            due_strains.append(strain)
    return due_strains


def build_subculture_due_email(
    due_strains: List[Dict[str, Any]],
    threshold_days: int | None = None,
) -> tuple[str, str]:
    threshold = threshold_days if threshold_days is not None else get_reminder_threshold_days()
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subject = f"[Algae Agent] {len(due_strains)} 个藻种临近传代，请查看实验室状态"

    lines = [
        "Algae Agent 检测到以下藻种已达到临近传代提醒阈值。",
        "",
        f"提醒时间: {today}",
        f"提醒阈值: 距上次传代 >= {threshold} 天",
        "",
        "需要查看的藻种:",
    ]
    for item in due_strains:
        lines.append(
            "- "
            f"{item.get('strain_id')} | "
            f"{item.get('name_cn')} ({item.get('name_en')}) | "
            f"当前代数: {item.get('generation_number')} | "
            f"距上次传代: {item.get('days_since_last_subculture')} 天 | "
            f"最后传代时间: {item.get('last_subculture_time') or '未知'}"
        )

    lines.extend(
        [
            "",
            "此邮件仅用于提醒查看实验室状态，不会触发传代、审批或数据库写入。",
            "",
            "预留通知类型:",
            f"- {REMINDER_TYPE_EXPERIMENT_FAILED}: 实验失败提醒",
            f"- {REMINDER_TYPE_REFLECTION_ANOMALY}: Reflection 异常发现提醒",
            f"- {REMINDER_TYPE_PENDING_APPROVAL_TIMEOUT}: pending approval 超时提醒",
        ]
    )
    return subject, "\n".join(lines)


def send_subculture_due_reminder(threshold_days: int | None = None) -> bool:
    threshold = threshold_days if threshold_days is not None else get_reminder_threshold_days()
    due_strains = get_due_subculture_strains(threshold)
    if not due_strains:
        print(f"[邮件提醒] 无藻种达到提醒阈值 {threshold} 天。")
        return False

    subject, body = build_subculture_due_email(due_strains, threshold)
    return send_email(subject, body)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
