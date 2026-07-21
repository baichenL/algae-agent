from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabwareSpec:
    labware_id: str
    display_name: str
    category: str
    max_volume_ml: float | None = None
    compatible_operations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabwareSpec":
        return cls(
            labware_id=str(data.get("labware_id") or ""),
            display_name=str(data.get("display_name") or ""),
            category=str(data.get("category") or ""),
            max_volume_ml=(
                float(data["max_volume_ml"])
                if data.get("max_volume_ml") is not None
                else None
            ),
            compatible_operations=list(data.get("compatible_operations") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class WorkspaceLocation:
    location_id: str
    display_name: str
    zone_type: str
    supported_operations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceLocation":
        return cls(
            location_id=str(data.get("location_id") or ""),
            display_name=str(data.get("display_name") or ""),
            zone_type=str(data.get("zone_type") or ""),
            supported_operations=list(data.get("supported_operations") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ContainerRef:
    container_id: str
    labware_id: str
    location_id: str
    role: str
    current_volume_ml: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContainerRef":
        return cls(
            container_id=str(data.get("container_id") or ""),
            labware_id=str(data.get("labware_id") or ""),
            location_id=str(data.get("location_id") or ""),
            role=str(data.get("role") or ""),
            current_volume_ml=(
                float(data["current_volume_ml"])
                if data.get("current_volume_ml") is not None
                else None
            ),
            metadata=dict(data.get("metadata") or {}),
        )
