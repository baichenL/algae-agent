from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class EffectClass(str, Enum):
    READ = "read"
    COMPUTE = "compute"
    CONTROL_WRITE = "control_write"
    ARTIFACT_WRITE = "artifact_write"
    PROPOSAL_WRITE = "proposal_write"
    DOMAIN_FACT_COMMIT = "domain_fact_commit"
    EXTERNAL_ACTUATION = "external_actuation"


class CanonicalExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StateObservation:
    state_observation_id: str
    canonical_status: str
    pending_status: str | None
    workflow_status: str | None
    business_state_version: str | None
    external_effect_status: str | None
    receipt_ref: str
    updated_at: str
