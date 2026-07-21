import json

from app.core import database
from app.services.protocols import (
    ExperimentProtocolSpec,
    build_subculture_protocol,
    compute_protocol_hash,
    verify_protocol_hash,
)


def _protocol(created_at="2026-01-01T00:00:00+00:00"):
    strain = database.get_algae_status("Chlorella_01")
    return build_subculture_protocol(
        strain_id="Chlorella_01",
        strain_snapshot=strain,
        session_id="session-protocol-hash",
        agent_run_id="run-protocol-hash",
        source_message="run subculture",
        created_at=created_at,
    )


def test_protocol_is_json_serializable_and_hash_valid(isolated_sqlite_db):
    protocol = _protocol()

    encoded = json.dumps(protocol.to_dict(), ensure_ascii=False, sort_keys=True)
    decoded = ExperimentProtocolSpec.from_dict(json.loads(encoded))

    assert decoded.protocol_id == protocol.protocol_id
    assert decoded.protocol_hash == protocol.protocol_hash
    assert verify_protocol_hash(decoded)


def test_same_protocol_content_has_stable_hash(isolated_sqlite_db):
    first = _protocol("2026-01-01T00:00:00+00:00")
    second = _protocol("2026-01-02T00:00:00+00:00")

    assert first.protocol_hash == second.protocol_hash
    assert first.protocol_id == second.protocol_id


def test_parameter_change_changes_hash(isolated_sqlite_db):
    protocol = _protocol()
    modified = ExperimentProtocolSpec.from_dict(protocol.to_dict())
    modified.parameters["media_target_volume"] = 120.0

    assert compute_protocol_hash(modified) != protocol.protocol_hash
    assert not verify_protocol_hash(modified)


def test_tampered_protocol_hash_is_rejected(isolated_sqlite_db):
    protocol = _protocol()
    tampered = ExperimentProtocolSpec.from_dict(protocol.to_dict())
    tampered.steps[3].parameters["volume_ml"] = 120.0

    assert not verify_protocol_hash(tampered, protocol.protocol_hash)
