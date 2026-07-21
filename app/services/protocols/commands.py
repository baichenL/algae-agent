from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProtocolCommand:
    command_id: str
    step_id: str
    operation: str
    parameters: dict[str, Any]
    device_or_location: str | None = None
    source_container: str | None = None
    target_container: str | None = None
    requires_human_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolCommand":
        return cls(
            command_id=str(data.get("command_id") or ""),
            step_id=str(data.get("step_id") or ""),
            operation=str(data.get("operation") or ""),
            parameters=dict(data.get("parameters") or {}),
            device_or_location=data.get("device_or_location"),
            source_container=data.get("source_container"),
            target_container=data.get("target_container"),
            requires_human_approval=bool(data.get("requires_human_approval", True)),
        )


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    status: str
    observation: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandResult":
        return cls(
            command_id=str(data.get("command_id") or ""),
            status=str(data.get("status") or ""),
            observation=dict(data.get("observation") or {}),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
        )


@dataclass(frozen=True)
class ProtocolRunPreview:
    protocol_id: str
    protocol_hash: str
    commands: list[ProtocolCommand]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "commands": [command.to_dict() for command in self.commands],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolRunPreview":
        return cls(
            protocol_id=str(data.get("protocol_id") or ""),
            protocol_hash=str(data.get("protocol_hash") or ""),
            commands=[
                ProtocolCommand.from_dict(item)
                for item in data.get("commands") or []
            ],
            summary=dict(data.get("summary") or {}),
        )
