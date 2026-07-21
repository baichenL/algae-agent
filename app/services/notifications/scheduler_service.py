import asyncio
import os

from app.core.time_utils import local_now
from app.services.notifications.notification_service import (
    get_active_reminder_window,
    send_due_subculture_notifications,
)


async def email_scheduler_loop():
    last_run_window_key = None
    while True:
        try:
            if _scheduler_enabled():
                now = local_now()
                window = get_active_reminder_window(now)
                if window and window["window_key"] != last_run_window_key:
                    send_due_subculture_notifications(source="scheduler", now=now)
                    last_run_window_key = window["window_key"]
        except Exception as exc:
            print(f"[email scheduler] Failed to run reminder scheduler: {exc}")
        await asyncio.sleep(60)


def _scheduler_enabled() -> bool:
    return _bool_env("EMAIL_SCHEDULER_ENABLED", False)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
