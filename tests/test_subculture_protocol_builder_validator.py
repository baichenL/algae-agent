import pytest

from app.core import database
from app.services.protocols import ProtocolBuildError, build_subculture_protocol, validate_protocol
from app.services.protocols.models import ExperimentProtocolSpec


def _valid_protocol():
    strain = database.get_algae_status("Chlorella_01")
    return build_subculture_protocol(
        strain_id="Chlorella_01",
        strain_snapshot=strain,
        session_id="validator-session",
        agent_run_id="validator-run",
        source_message="run subculture",
    )


def _validate(protocol, *, strain_snapshot=None, pending_actions=None, hardware_controller=None):
    strain = strain_snapshot
    if strain is None:
        strain = database.get_algae_status(protocol.target_strain_id)
    return validate_protocol(
        protocol,
        strain_snapshot=strain,
        all_strains=database.list_algae_status(),
        pending_actions=pending_actions or [],
        hardware_controller=hardware_controller,
    )


def test_valid_strain_builds_and_validates_protocol(isolated_sqlite_db):
    protocol = _valid_protocol()
    report = _validate(protocol)

    assert protocol.target_strain_id == "Chlorella_01"
    assert protocol.template_id == "subculture_standard_v1"
    assert report.valid is True


def test_missing_strain_cannot_build_protocol(isolated_sqlite_db):
    with pytest.raises(ProtocolBuildError) as exc:
        build_subculture_protocol(
            strain_id="Missing_01",
            strain_snapshot=None,
            source_message="run subculture",
        )

    assert exc.value.code == "TARGET_STRAIN_NOT_FOUND"


def test_missing_required_parameter_is_structured_error(isolated_sqlite_db):
    with pytest.raises(ProtocolBuildError) as exc:
        build_subculture_protocol(
            strain_id="Chlorella_01",
            strain_snapshot=database.get_algae_status("Chlorella_01"),
            parameters={"source_reactor_id": ""},
        )

    assert exc.value.code == "PROTOCOL_MISSING_REQUIRED_FIELD"
    assert "source_reactor_id" in exc.value.missing_fields


def test_step_missing_or_reordered_fails_validation(isolated_sqlite_db):
    protocol = _valid_protocol()
    missing = ExperimentProtocolSpec.from_dict(protocol.to_dict())
    missing.steps = missing.steps[:-1]
    reordered = ExperimentProtocolSpec.from_dict(protocol.to_dict())
    reordered.steps = list(reversed(reordered.steps))

    assert not _validate(missing).valid
    assert "PROTOCOL_STEP_ORDER_INVALID" in {issue.code for issue in _validate(missing).errors}
    assert not _validate(reordered).valid


def test_unknown_operation_and_parameter_range_fail_validation(isolated_sqlite_db):
    unknown = ExperimentProtocolSpec.from_dict(_valid_protocol().to_dict())
    unknown.steps[0] = type(unknown.steps[0])(
        **{**unknown.steps[0].to_dict(), "operation": "run_python"}
    )
    out_of_range = ExperimentProtocolSpec.from_dict(_valid_protocol().to_dict())
    out_of_range.parameters["media_target_volume"] = 999.0

    assert "PROTOCOL_OPERATION_NOT_ALLOWED" in {issue.code for issue in _validate(unknown).errors}
    assert "PROTOCOL_INVALID_PARAMETER" in {issue.code for issue in _validate(out_of_range).errors}


def test_hardware_capability_and_generation_stale_fail_validation(isolated_sqlite_db):
    protocol = _valid_protocol()
    stale_snapshot = dict(database.get_algae_status("Chlorella_01"))
    stale_snapshot["generation_number"] = int(stale_snapshot["generation_number"]) + 1

    class NoHardware:
        pass

    assert "HARDWARE_CAPABILITY_MISSING" in {
        issue.code for issue in _validate(protocol, hardware_controller=NoHardware()).errors
    }
    assert "TARGET_STATE_STALE" in {
        issue.code for issue in _validate(protocol, strain_snapshot=stale_snapshot).errors
    }


def test_duplicate_pending_is_reported_by_validator(isolated_sqlite_db):
    protocol = _valid_protocol()
    pending_id = database.insert_pending_action(
        "workflow_subculture",
        {
            "type": "workflow_subculture",
            "data": {"strain_id": "Chlorella_01"},
            "protocol": {"protocol_hash": protocol.protocol_hash},
        },
    )
    pending = database.get_pending_action(pending_id)

    report = _validate(protocol, pending_actions=[pending])

    assert not report.valid
    assert "DUPLICATE_PENDING" in {issue.code for issue in report.errors}
