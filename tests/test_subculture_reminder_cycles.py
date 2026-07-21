import datetime
from types import SimpleNamespace

from app.core import database
from app.core.db.reminder_cycles import build_cycle_key, get_reminder_cycle
from app.models.email_schema import EmailSendResponse
from app.services.notifications import notification_service


def _due_strain(generation=17):
    return {
        "strain_id": "Chlorella_01",
        "name_cn": "Chlorella",
        "name_en": "Chlorella vulgaris",
        "generation_number": generation,
        "days_since_last_subculture": 6,
        "last_subculture_time": "2026-06-11 09:00:00",
    }


def _patch_successful_reminder(monkeypatch, strain, sent_requests):
    monkeypatch.setattr(notification_service, "get_default_recipients", lambda: ["lab@example.com"])
    monkeypatch.setattr(
        notification_service,
        "build_context_snapshot",
        lambda source: SimpleNamespace(strains=[strain]),
    )

    def fake_send_email(request):
        sent_requests.append(request)
        return EmailSendResponse(
            status="success",
            sent=True,
            message="sent",
            recipients=request.recipients,
            log_id=f"log-{len(sent_requests)}",
        )

    monkeypatch.setattr(notification_service, "send_email", fake_send_email)


def test_same_cycle_window_sends_only_once(isolated_sqlite_db, monkeypatch):
    strain = _due_strain()
    sent_requests = []
    _patch_successful_reminder(monkeypatch, strain, sent_requests)

    now = datetime.datetime(2026, 6, 17, 9, 5, 0)
    first = notification_service.send_due_subculture_notifications(now=now)
    duplicate = notification_service.send_due_subculture_notifications(
        now=datetime.datetime(2026, 6, 17, 9, 20, 0)
    )

    assert len(first) == 1
    assert duplicate == []
    assert len(sent_requests) == 1

    cycle = get_reminder_cycle(build_cycle_key("Chlorella_01", 17))
    assert cycle["status"] == "active"
    assert cycle["total_sent_count"] == 1


def test_daily_limit_blocks_fourth_window(isolated_sqlite_db, monkeypatch):
    strain = _due_strain()
    sent_requests = []
    _patch_successful_reminder(monkeypatch, strain, sent_requests)
    monkeypatch.setenv("SUBCULTURE_REMINDER_WINDOWS", "09:00,14:00,18:00,20:00")

    for hour in [9, 14, 18]:
        responses = notification_service.send_due_subculture_notifications(
            now=datetime.datetime(2026, 6, 17, hour, 5, 0)
        )
        assert len(responses) == 1

    blocked = notification_service.send_due_subculture_notifications(
        now=datetime.datetime(2026, 6, 17, 20, 5, 0)
    )

    assert blocked == []
    assert len(sent_requests) == 3
    cycle = get_reminder_cycle(build_cycle_key("Chlorella_01", 17))
    assert cycle["total_sent_count"] == 3


def test_cycle_limit_silences_cycle(isolated_sqlite_db, monkeypatch):
    strain = _due_strain()
    sent_requests = []
    _patch_successful_reminder(monkeypatch, strain, sent_requests)
    monkeypatch.setenv("SUBCULTURE_REMINDER_CYCLE_LIMIT", "2")

    for when in [
        datetime.datetime(2026, 6, 17, 9, 5, 0),
        datetime.datetime(2026, 6, 17, 14, 5, 0),
    ]:
        responses = notification_service.send_due_subculture_notifications(now=when)
        assert len(responses) == 1

    blocked = notification_service.send_due_subculture_notifications(
        now=datetime.datetime(2026, 6, 17, 18, 5, 0)
    )

    assert blocked == []
    assert len(sent_requests) == 2
    cycle = get_reminder_cycle(build_cycle_key("Chlorella_01", 17))
    assert cycle["status"] == "silenced"
    assert cycle["total_sent_count"] == 2
    assert cycle["silenced_reason"] == "cycle_email_cap"


def test_subculture_completion_resolves_previous_cycle(isolated_sqlite_db, monkeypatch):
    strain = _due_strain()
    sent_requests = []
    _patch_successful_reminder(monkeypatch, strain, sent_requests)

    notification_service.send_due_subculture_notifications(
        now=datetime.datetime(2026, 6, 17, 9, 5, 0)
    )
    database.update_algae_status(
        strain_id="Chlorella_01",
        generation_number=18,
        days_since_last_subculture=0,
        last_time="2026-06-17 10:00:00",
    )

    old_cycle = get_reminder_cycle(build_cycle_key("Chlorella_01", 17))
    new_cycle = get_reminder_cycle(build_cycle_key("Chlorella_01", 18))

    assert old_cycle["status"] == "resolved"
    assert new_cycle is None


def test_status_attachment_keeps_due_visible_when_silenced(isolated_sqlite_db, monkeypatch):
    strain = _due_strain()
    sent_requests = []
    _patch_successful_reminder(monkeypatch, strain, sent_requests)
    monkeypatch.setenv("SUBCULTURE_REMINDER_CYCLE_LIMIT", "1")

    notification_service.send_due_subculture_notifications(
        now=datetime.datetime(2026, 6, 17, 9, 5, 0)
    )
    enriched = notification_service.attach_reminder_status_to_strains([strain])

    assert enriched[0]["subculture_due"] is True
    assert enriched[0]["reminder_cycle"]["status"] == "silenced"
