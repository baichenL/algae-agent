from app.core import database
from app.services.protocols import build_protocol_run_preview, build_subculture_protocol, validate_protocol
from app.services.protocols.commands import CommandResult, ProtocolCommand, ProtocolRunPreview


def _protocol(parameters=None):
    return build_subculture_protocol(
        strain_id="Chlorella_01",
        strain_snapshot=database.get_algae_status("Chlorella_01"),
        session_id="preview-session",
        agent_run_id="preview-run",
        source_message="run subculture",
        parameters=parameters,
    )


def test_protocol_command_models_are_json_serializable():
    command = ProtocolCommand(
        command_id="cmd-01",
        step_id="transfer_seed_culture",
        operation="transfer_seed",
        parameters={"volume_ml": 1.0},
        device_or_location="biosafety_cabinet",
        source_container="Reactor_A",
        target_container="Reactor_B",
    )
    result = CommandResult(command_id="cmd-01", status="simulated_ok")
    preview = ProtocolRunPreview(
        protocol_id="protocol-test",
        protocol_hash="hash-test",
        commands=[command],
    )

    assert ProtocolCommand.from_dict(command.to_dict()) == command
    assert CommandResult.from_dict(result.to_dict()) == result
    assert ProtocolRunPreview.from_dict(preview.to_dict()) == preview


def test_subculture_protocol_builds_stable_run_preview(isolated_sqlite_db):
    protocol = _protocol()
    preview = build_protocol_run_preview(protocol)
    rows = [command.to_dict() for command in preview.commands]

    assert preview.protocol_id == protocol.protocol_id
    assert preview.protocol_hash == protocol.protocol_hash
    assert rows[0]["operation"] == "check_schedule"
    assert rows[3]["target_container"] == "Reactor_B"
    assert rows[4]["source_container"] == "Reactor_A"
    assert rows[4]["target_container"] == "Reactor_B"


def test_labware_capacity_overflow_is_structured_validation_error(isolated_sqlite_db):
    protocol = _protocol(parameters={"media_target_volume": 240.0, "seed_target_volume": 20.0})
    report = validate_protocol(
        protocol,
        strain_snapshot=database.get_algae_status("Chlorella_01"),
        all_strains=database.list_algae_status(),
        pending_actions=[],
    )

    assert not report.valid
    assert "LABWARE_CAPACITY_EXCEEDED" in {issue.code for issue in report.errors}


def test_unknown_container_is_structured_validation_error(isolated_sqlite_db):
    protocol = _protocol(parameters={"target_reactor_id": "Unknown_Reactor"})
    report = validate_protocol(
        protocol,
        strain_snapshot=database.get_algae_status("Chlorella_01"),
        all_strains=database.list_algae_status(),
        pending_actions=[],
    )

    assert not report.valid
    assert "LABWARE_WORKSPACE_UNKNOWN" in {issue.code for issue in report.errors}
