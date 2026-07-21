from types import SimpleNamespace

from app.services.chat import email_intent_handler, response_builder
from app.services.chat.email_intent_handler import handle_email_intent


def _history():
    return [{"role": "system", "content": "test"}]


def test_email_handler_returns_draft_by_default(monkeypatch):
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    draft = SimpleNamespace(
        model_dump=lambda: {
            "subject": "Manual check",
            "body": "Please check culture status",
            "recipients": ["lab@example.com"],
            "metadata": {"strain_id": "Chlorella_01"},
            "template_type": "manual_check",
        }
    )
    monkeypatch.setattr(
        email_intent_handler,
        "maybe_send_email_from_frontend_intent",
        lambda message, context_snapshot: draft,
    )

    response = handle_email_intent(
        "email-s2",
        _history(),
        "please email the lab",
        SimpleNamespace(),
    )

    assert response.agent_output["action"] == "email_draft"
    assert response.agent_output["requires_confirmation"] is True
    assert response.agent_output["draft"]["subject"] == "Manual check"


def test_email_handler_maps_send_result(monkeypatch):
    monkeypatch.setattr(response_builder, "save_session_memory", lambda session_id, history: None)
    sent_result = SimpleNamespace(
        sent=True,
        message="sent ok",
        model_dump=lambda: {
            "sent": True,
            "status": "success",
            "message": "sent ok",
        },
    )
    monkeypatch.setattr(
        email_intent_handler,
        "maybe_send_email_from_frontend_intent",
        lambda message, context_snapshot: sent_result,
    )

    response = handle_email_intent(
        "email-s3",
        _history(),
        "send email now",
        SimpleNamespace(),
    )

    assert response.agent_output["action"] == "email_sent"
    assert response.agent_output["sent"] is True
    assert response.natural_reply == "sent ok"
