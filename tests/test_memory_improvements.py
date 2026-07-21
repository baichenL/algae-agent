import json

from app.core import config, database
from app.services.context import context_builder
from app.services.memory import memory_service
from app.services.strains import strain_service


def test_session_memory_is_trimmed_and_session_id_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_DIR", str(tmp_path))
    monkeypatch.setattr(config, "MAX_SESSION_MESSAGES", 5)

    history = [{"role": "system", "content": "system"}]
    history.extend({"role": "user", "content": f"message-{index}"} for index in range(10))

    config.save_memory("../unsafe/session?", history)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("history_")
    assert "/" not in files[0].name
    assert "\\" not in files[0].name

    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(saved) == 5
    assert saved[0] == {"role": "system", "content": "system"}
    assert saved[-1] == {"role": "user", "content": "message-9"}


def test_context_snapshot_has_timestamp_dict_and_recent_experiments(monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(context_builder, "local_time_string", lambda: "2026-06-17 10:00:00")
    monkeypatch.setattr(
        context_builder,
        "get_status_memory",
        lambda: [{"strain_id": "Chlorella_01", "name_cn": "chlorella"}],
    )
    monkeypatch.setattr(context_builder, "get_pending_memory", lambda: [])
    monkeypatch.setattr(
        context_builder,
        "get_recent_experiments",
        lambda strain_id=None, limit=5: [
            {
                "id": 1,
                "strain": strain_id,
                "generation": 3,
                "status": "SUCCESS",
                "created_at": "2026-06-17T01:00:00",
                "hardware_log_count": 2,
            }
        ],
    )

    snapshot = context_builder.build_context_snapshot("s1", strain_id="Chlorella_01")

    assert snapshot.created_at == "2026-06-17 10:00:00"
    assert snapshot.to_dict()["recent_experiments"][0]["strain"] == "Chlorella_01"
    prompt = snapshot.to_prompt_facts()
    assert "created_at: 2026-06-17 10:00:00" in prompt
    assert "recent_experiment_count: 1" in prompt


def test_pending_creation_is_idempotent_until_reviewed(isolated_sqlite_db, monkeypatch):
    monkeypatch.setattr(strain_service, "append_decision_event", lambda event: None)
    payload = {
        "strain_id": "Spirulina_01",
        "name_cn": "Spirulina",
        "name_en": "Spirulina platensis",
        "generation_number": 1,
        "days_since_last_subculture": 0,
    }

    first_id = strain_service.create_pending_add_strain(payload)
    duplicate_id = strain_service.create_pending_add_strain(payload)

    assert duplicate_id == first_id
    active_adds = memory_service.get_active_pending_actions("Spirulina_01", "add_strain")
    assert [item["id"] for item in active_adds] == [first_id]

    update_id = strain_service.create_pending_update_strain({
        "strain_id": "Spirulina_01",
        "generation_number": 2,
    })
    assert update_id != first_id

    assert strain_service.deny_pending_action(first_id)["status"] == "success"
    new_id = strain_service.create_pending_add_strain(payload)
    assert new_id != first_id


def test_recent_experiment_memory_is_limited_and_compact(isolated_sqlite_db):
    database.insert_experiment({
        "strain": "Chlorella_01",
        "generation": 1,
        "status": "SUCCESS",
        "media_components": {"nitrate": "1x"},
        "temperature": 25.0,
        "light_intensity": 100.0,
        "od_readings": [0.2, 0.4],
        "hardware_logs": ["step-a", "step-b"],
        "created_at": "2026-06-17T01:00:00",
    })
    database.insert_experiment({
        "strain": "Chlorella_01",
        "generation": 2,
        "status": "SUCCESS",
        "media_components": {},
        "temperature": 25.0,
        "light_intensity": 100.0,
        "od_readings": [],
        "hardware_logs": ["step-c"],
        "created_at": "2026-06-17T02:00:00",
    })

    recent = memory_service.get_recent_experiments("Chlorella_01", limit=1)

    assert len(recent) == 1
    assert recent[0]["generation"] == 2
    assert recent[0]["hardware_log_count"] == 1
    assert "hardware_logs" not in recent[0]
