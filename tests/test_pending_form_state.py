from app.services.chat import pending_form_state as pfs

from tests.conftest import context


def test_pending_form_state_save_merge_and_clear(tmp_path, monkeypatch, isolated_sqlite_db):
    monkeypatch.setattr(pfs, "STATE_DIR", str(tmp_path))
    session_id = "state-test"
    state = {
        "active": True,
        "operation": "add",
        "tool_name": "add_algae_strain",
        "collected_fields": {"name_cn": "螺旋藻"},
        "missing_fields": ["strain_id", "name_en"],
        "candidates": [],
        "source_message": "增加螺旋藻",
    }

    pfs.save_pending_form_state(session_id, state)
    loaded = pfs.get_pending_form_state(session_id)
    merged = pfs.merge_pending_form_fields(
        loaded,
        "Spirulina_02 | 螺旋藻 (Spirulina platensis)",
        context(),
    )

    assert merged["missing_fields"] == []
    assert merged["collected_fields"]["strain_id"] == "Spirulina_02"
    assert merged["collected_fields"]["name_en"] == "Spirulina platensis"

    pfs.clear_pending_form_state(session_id)
    assert pfs.get_pending_form_state(session_id) is None
