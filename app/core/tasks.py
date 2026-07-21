import asyncio
import os

from app.services.notifications.notification_service import send_due_subculture_notifications
from app.services.notifications.reminder_service import (
    build_subculture_due_email,
    get_due_subculture_strains,
    get_reminder_check_interval_seconds,
    get_reminder_threshold_days,
)
from app.services.email.email_notifier import send_email


async def daily_schedule_monitor():
    """Scan all strains for subculture-due reminders.

    This monitor is notification-only. It must not execute the hardware
    workflow or write strain status; formal subculture execution still needs an
    explicit user instruction or a future human-approval workflow.
    """
    while True:
        try:
            if _subculture_monitor_enabled():
                send_due_subculture_notifications(source="subculture_due_monitor")
        except Exception as monitor_err:
            print(f"[传代提醒监控] 处理失败: {str(monitor_err)}")

        await asyncio.sleep(_subculture_monitor_interval_seconds())


async def email_reminder_monitor():
    """Legacy reminder loop kept for compatibility with old imports.

    New startup code uses daily_schedule_monitor plus the centralized
    notification service. This function remains notification-only.
    """
    last_sent_key = None

    while True:
        try:
            threshold = get_reminder_threshold_days()
            due_strains = get_due_subculture_strains(threshold)

            if due_strains:
                today = __import__("datetime").datetime.date.today().isoformat()
                strain_key = ",".join(sorted(s.get("strain_id", "") for s in due_strains))
                send_key = f"{today}:{threshold}:{strain_key}"

                if send_key != last_sent_key:
                    subject, body = build_subculture_due_email(due_strains, threshold)
                    if send_email(subject, body):
                        last_sent_key = send_key
                else:
                    print("[邮件提醒] 今日相同临近传代提醒已发送，跳过重复发送。")
            else:
                print(f"[邮件提醒] 当前无品系达到 {threshold} 天提醒阈值。")
        except Exception as reminder_err:
            print(f"[邮件提醒异常] 提醒任务处理失败: {str(reminder_err)}")

        await asyncio.sleep(get_reminder_check_interval_seconds())


def _subculture_monitor_enabled() -> bool:
    return _bool_env("SUBCULTURE_DUE_MONITOR_ENABLED", True)


def _subculture_monitor_interval_seconds() -> int:
    return _int_env("SUBCULTURE_DUE_MONITOR_INTERVAL_SECONDS", 300)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
