from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterRange:
    min_value: float | int | None = None
    max_value: float | int | None = None
    value_type: str = "number"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterRange":
        return cls(
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            value_type=data.get("value_type", "number"),
        )


@dataclass(frozen=True)
class ProtocolConstraint:
    code: str
    description: str
    level: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolConstraint":
        return cls(
            code=str(data.get("code") or ""),
            description=str(data.get("description") or ""),
            level=str(data.get("level") or "error"),
        )


@dataclass(frozen=True)
class ProtocolTemplateStep:
    step_id: str
    operation: str
    required: bool
    order: int
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    expected_observations: list[dict[str, Any]] = field(default_factory=list)
    failure_policy: str = "safe_shutdown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolTemplateStep":
        return cls(
            step_id=str(data.get("step_id") or ""),
            operation=str(data.get("operation") or ""),
            required=bool(data.get("required", True)),
            order=int(data.get("order") or 0),
            parameter_schema=dict(data.get("parameter_schema") or {}),
            preconditions=list(data.get("preconditions") or []),
            expected_observations=list(data.get("expected_observations") or []),
            failure_policy=str(data.get("failure_policy") or "safe_shutdown"),
        )


@dataclass(frozen=True)
class ProtocolTemplate:
    template_id: str
    template_version: int
    experiment_type: str
    name: str
    description: str
    required_inputs: list[str]
    required_hardware_capabilities: list[str]
    steps: list[ProtocolTemplateStep]
    safety_constraints: list[ProtocolConstraint]
    allowed_parameter_ranges: dict[str, ParameterRange]
    requires_approval: bool
    active: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        payload["safety_constraints"] = [
            constraint.to_dict() for constraint in self.safety_constraints
        ]
        payload["allowed_parameter_ranges"] = {
            key: value.to_dict()
            for key, value in self.allowed_parameter_ranges.items()
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolTemplate":
        return cls(
            template_id=str(data.get("template_id") or ""),
            template_version=int(data.get("template_version") or 0),
            experiment_type=str(data.get("experiment_type") or ""),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            required_inputs=list(data.get("required_inputs") or []),
            required_hardware_capabilities=list(
                data.get("required_hardware_capabilities") or []
            ),
            steps=[
                ProtocolTemplateStep.from_dict(item)
                for item in data.get("steps") or []
            ],
            safety_constraints=[
                ProtocolConstraint.from_dict(item)
                for item in data.get("safety_constraints") or []
            ],
            allowed_parameter_ranges={
                key: ParameterRange.from_dict(value)
                for key, value in (data.get("allowed_parameter_ranges") or {}).items()
            },
            requires_approval=bool(data.get("requires_approval", True)),
            active=bool(data.get("active", True)),
        )


@dataclass(frozen=True)
class ProtocolStep:
    step_id: str
    operation: str
    parameters: dict[str, Any]
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    expected_observation: dict[str, Any] | None = None
    failure_policy: str = "safe_shutdown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolStep":
        return cls(
            step_id=str(data.get("step_id") or ""),
            operation=str(data.get("operation") or ""),
            parameters=dict(data.get("parameters") or {}),
            preconditions=list(data.get("preconditions") or []),
            expected_observation=(
                dict(data.get("expected_observation"))
                if isinstance(data.get("expected_observation"), dict)
                else None
            ),
            failure_policy=str(data.get("failure_policy") or "safe_shutdown"),
        )


@dataclass
class ExperimentProtocolSpec:
    protocol_id: str
    protocol_version: int
    template_id: str
    template_version: int
    experiment_type: str
    objective: str
    target_strain_id: str
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    steps: list[ProtocolStep]
    hardware_requirements: list[str]
    safety_constraints: list[dict[str, Any]]
    evidence_citations: list[dict[str, Any]]
    expected_outputs: list[str]
    created_at: str
    created_by: str
    source_run_id: str | None = None
    source_session_id: str | None = None
    protocol_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentProtocolSpec":
        return cls(
            protocol_id=str(data.get("protocol_id") or ""),
            protocol_version=int(data.get("protocol_version") or 1),
            template_id=str(data.get("template_id") or ""),
            template_version=int(data.get("template_version") or 0),
            experiment_type=str(data.get("experiment_type") or ""),
            objective=str(data.get("objective") or ""),
            target_strain_id=str(data.get("target_strain_id") or ""),
            inputs=dict(data.get("inputs") or {}),
            parameters=dict(data.get("parameters") or {}),
            steps=[ProtocolStep.from_dict(item) for item in data.get("steps") or []],
            hardware_requirements=list(data.get("hardware_requirements") or []),
            safety_constraints=list(data.get("safety_constraints") or []),
            evidence_citations=list(data.get("evidence_citations") or []),
            expected_outputs=list(data.get("expected_outputs") or []),
            created_at=str(data.get("created_at") or ""),
            created_by=str(data.get("created_by") or "system"),
            source_run_id=data.get("source_run_id"),
            source_session_id=data.get("source_session_id"),
            protocol_hash=str(data.get("protocol_hash") or ""),
        )


@dataclass(frozen=True)
class ProtocolValidationCheck:
    check_code: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolValidationCheck":
        return cls(
            check_code=str(data.get("check_code") or ""),
            passed=bool(data.get("passed")),
            message=str(data.get("message") or ""),
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class ProtocolIssue:
    code: str
    level: str
    message: str
    step_id: str | None = None
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolIssue":
        return cls(
            code=str(data.get("code") or ""),
            level=str(data.get("level") or "error"),
            message=str(data.get("message") or ""),
            step_id=data.get("step_id"),
            field_path=data.get("field_path"),
            details=dict(data.get("details") or {}),
        )


@dataclass(frozen=True)
class ProtocolValidationReport:
    protocol_id: str
    valid: bool
    checks: list[ProtocolValidationCheck]
    errors: list[ProtocolIssue]
    warnings: list[ProtocolIssue]
    validated_at: str
    validator_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "valid": self.valid,
            "checks": [check.to_dict() for check in self.checks],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "validated_at": self.validated_at,
            "validator_version": self.validator_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolValidationReport":
        return cls(
            protocol_id=str(data.get("protocol_id") or ""),
            valid=bool(data.get("valid")),
            checks=[
                ProtocolValidationCheck.from_dict(item)
                for item in data.get("checks") or []
            ],
            errors=[ProtocolIssue.from_dict(item) for item in data.get("errors") or []],
            warnings=[
                ProtocolIssue.from_dict(item) for item in data.get("warnings") or []
            ],
            validated_at=str(data.get("validated_at") or ""),
            validator_version=str(data.get("validator_version") or ""),
        )


@dataclass(frozen=True)
class ProtocolSimulationStepResult:
    step_id: str
    operation: str
    success: bool
    observation: dict[str, Any]
    error_code: str | None
    error_message: str | None
    state_before: dict[str, Any]
    state_after: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolSimulationStepResult":
        return cls(
            step_id=str(data.get("step_id") or ""),
            operation=str(data.get("operation") or ""),
            success=bool(data.get("success")),
            observation=dict(data.get("observation") or {}),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            state_before=dict(data.get("state_before") or {}),
            state_after=dict(data.get("state_after") or {}),
        )


@dataclass(frozen=True)
class ProtocolSimulationReport:
    simulation_id: str
    protocol_id: str
    protocol_hash: str
    success: bool
    started_at: str
    completed_at: str
    step_results: list[ProtocolSimulationStepResult]
    final_simulated_state: dict[str, Any]
    failure_code: str | None
    failure_message: str | None
    safe_shutdown_triggered: bool
    simulator_name: str
    simulator_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "success": self.success,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "step_results": [item.to_dict() for item in self.step_results],
            "final_simulated_state": self.final_simulated_state,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "safe_shutdown_triggered": self.safe_shutdown_triggered,
            "simulator_name": self.simulator_name,
            "simulator_version": self.simulator_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolSimulationReport":
        return cls(
            simulation_id=str(data.get("simulation_id") or ""),
            protocol_id=str(data.get("protocol_id") or ""),
            protocol_hash=str(data.get("protocol_hash") or ""),
            success=bool(data.get("success")),
            started_at=str(data.get("started_at") or ""),
            completed_at=str(data.get("completed_at") or ""),
            step_results=[
                ProtocolSimulationStepResult.from_dict(item)
                for item in data.get("step_results") or []
            ],
            final_simulated_state=dict(data.get("final_simulated_state") or {}),
            failure_code=data.get("failure_code"),
            failure_message=data.get("failure_message"),
            safe_shutdown_triggered=bool(data.get("safe_shutdown_triggered")),
            simulator_name=str(data.get("simulator_name") or ""),
            simulator_version=str(data.get("simulator_version") or ""),
        )
