import datetime
import os
from typing import List

from app.core.db.reminder_cycles import (
    build_cycle_key,
    list_reminder_cycles_for_strains,
    mark_reserved_events_failed,
    mark_reserved_events_sent,
    reserve_reminder_window,
)
from app.core.time_utils import local_now
from app.models.email_schema import EmailDraft, EmailSendRequest, EmailSendResponse
from app.services.context.context_builder import build_context_snapshot
from app.services.email.email_service import get_default_recipients, send_email
from app.services.email.email_template import build_passage_reminder_draft


def collect_daily_notifications() -> List[EmailDraft]:
    recipients = get_default_recipients()
    if not recipients:
        return []
    snapshot = build_context_snapshot("scheduler")
    threshold = get_subculture_threshold_days()
    due_strains = get_due_subculture_strains(snapshot.strains, threshold)
    drafts = []
    if due_strains:
        drafts.append(build_passage_reminder_draft(due_strains, recipients, threshold))
    return drafts


def send_due_subculture_notifications(
    source: str = "subculture_due_monitor",
    now: datetime.datetime | None = None,
) -> List[EmailSendResponse]:
    recipients = get_default_recipients()
    if not recipients:
        print("[subculture reminder] Default recipients are not configured; skipping email.")
        return []

    current_time = now or local_now()
    window = get_active_reminder_window(current_time)
    if not window:
        print("[subculture reminder] Current time is outside reminder windows; skipping scan send.")
        return []

    threshold = get_subculture_threshold_days()
    snapshot = build_context_snapshot(source)
    due_strains = get_due_subculture_strains(snapshot.strains, threshold)
    if not due_strains:
        print(f"[subculture reminder] No strains reached the {threshold}-day threshold.")
        return []

    reserved_strains: list[dict] = []
    reserved_cycle_keys: list[str] = []
    for strain in due_strains:
        strain_id = str(strain.get("strain_id") or "")
        generation_number = int(strain.get("generation_number") or 0)
        if not strain_id or generation_number <= 0:
            continue

        cycle_key = build_cycle_key(strain_id, generation_number)
        reserved, reason, cycle = reserve_reminder_window(
            strain_id=strain_id,
            generation_number=generation_number,
            window_key=window["window_key"],
            sent_date=window["sent_date"],
            source=source,
            now=current_time,
            daily_limit=get_daily_reminder_limit(),
            cycle_limit=get_cycle_reminder_limit(),
            max_reminder_days=get_max_reminder_days(),
            metadata={"threshold_days": threshold, "source": source},
        )
        if reserved:
            enriched = {
                **strain,
                "reminder_cycle_key": cycle_key,
                "reminder_cycle_status": (cycle or {}).get("status", "active"),
                "reminder_window_key": window["window_key"],
            }
            reserved_strains.append(enriched)
            reserved_cycle_keys.append(cycle_key)
        else:
            print(f"[subculture reminder] Skipped {cycle_key}: {reason}")

    if not reserved_strains:
        print("[subculture reminder] All due strains were outside reminder limits or already reserved.")
        return []

    draft = build_passage_reminder_draft(reserved_strains, recipients, threshold)
    response = send_email(EmailSendRequest(
        subject=draft.subject,
        body=draft.body,
        recipients=draft.recipients,
        source=source,
        metadata={
            **draft.metadata,
            "template_type": draft.template_type,
            "cycle_keys": reserved_cycle_keys,
            "window_key": window["window_key"],
        },
    ))
    if response.sent:
        mark_reserved_events_sent(
            reserved_cycle_keys,
            window["window_key"],
            response.log_id,
            now=current_time,
            cycle_limit=get_cycle_reminder_limit(),
        )
        print(f"[subculture reminder] Sent reminder for {len(reserved_strains)} due strains.")
    else:
        mark_reserved_events_failed(
            reserved_cycle_keys,
            window["window_key"],
            response.message,
            now=current_time,
        )
        print(f"[subculture reminder] Reminder email failed: {response.message}")
    return [response]


def get_subculture_threshold_days() -> int:
    return _int_env(
        "MICROALGAE_SUBCULTURE_THRESHOLD_DAYS",
        _int_env(
            "MICROALGAE_PASSAGE_DAY_THRESHOLD",
            _int_env("REMINDER_THRESHOLD_DAYS", 6),
        ),
    )


def get_due_subculture_strains(strains: list[dict], threshold: int | None = None) -> list[dict]:
    resolved_threshold = threshold if threshold is not None else get_subculture_threshold_days()
    return [
        strain
        for strain in strains
        if int(strain.get("days_since_last_subculture") or 0) >= resolved_threshold
    ]


def attach_reminder_status_to_strains(strains: list[dict]) -> list[dict]:
    cycles = list_reminder_cycles_for_strains(strains)
    enriched = []
    threshold = get_subculture_threshold_days()
    for strain in strains:
        generation = int(strain.get("generation_number") or 0)
        strain_id = strain.get("strain_id")
        cycle_key = build_cycle_key(strain_id, generation) if strain_id and generation else None
        cycle = cycles.get(cycle_key) if cycle_key else None
        is_due = int(strain.get("days_since_last_subculture") or 0) >= threshold
        enriched.append({
            **strain,
            "subculture_due": is_due,
            "reminder_cycle": cycle,
        })
    return enriched


def get_active_reminder_window(now: datetime.datetime | None = None) -> dict | None:
    current_time = now or local_now()
    windows = get_reminder_windows()
    window_minutes = get_reminder_window_minutes()
    current_date = current_time.date()

    for hour, minute in windows:
        start = datetime.datetime.combine(
            current_date,
            datetime.time(hour=hour, minute=minute),
            tzinfo=current_time.tzinfo,
        )
        end = start + datetime.timedelta(minutes=window_minutes)
        if start <= current_time < end:
            label = f"{hour:02d}:{minute:02d}"
            return {
                "window_key": f"{current_date.isoformat()}T{label}",
                "sent_date": current_date.isoformat(),
                "label": label,
                "start": start,
                "end": end,
            }
    return None


def get_reminder_windows() -> list[tuple[int, int]]:
    value = os.getenv("SUBCULTURE_REMINDER_WINDOWS", "09:00,14:00,18:00")
    windows = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            hour_text, minute_text = text.split(":", 1)
            hour = int(hour_text)
            minute = int(minute_text)
        except ValueError:
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            windows.append((hour, minute))
    return windows or [(9, 0), (14, 0), (18, 0)]


def get_reminder_window_minutes() -> int:
    return max(_int_env("SUBCULTURE_REMINDER_WINDOW_MINUTES", 30), 1)


def get_daily_reminder_limit() -> int:
    return max(_int_env("SUBCULTURE_REMINDER_DAILY_LIMIT", 3), 1)


def get_cycle_reminder_limit() -> int:
    return max(_int_env("SUBCULTURE_REMINDER_CYCLE_LIMIT", 6), 1)


def get_max_reminder_days() -> int:
    return max(_int_env("SUBCULTURE_REMINDER_MAX_DAYS", 2), 1)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
