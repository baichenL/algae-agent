from __future__ import annotations

from app.core.db import scientific as scientific_db
from app.core.effects import stable_hash
from app.services.agent_runtime.safety_v2 import POLICY_VERSION
from app.services.scientific.models import experiment_design_hash
from app.services.scientific.proposal_gate import (
    SIMULATOR_VERSION,
    VALIDATOR_VERSION,
)


def validate_pending_for_execution(pending: dict) -> str | None:
    """Pure, read-only validation shared by the API and trusted worker."""
    action_type = ((pending.get("payload") or {}).get("type") or pending.get("action_type"))
    if action_type == "scientific_experiment_plan":
        data = (pending.get("payload") or {}).get("data") or {}
        design = dict(data.get("design") or {})
        claimed = data.get("design_hash")
        if (
            not claimed
            or design.get("design_hash") != claimed
            or experiment_design_hash(design) != claimed
        ):
            return "design_hash_mismatch"
    if int(pending.get("proposal_version") or 1) < 2:
        return None
    envelope = pending.get("proposal_envelope") or {}
    if not envelope:
        return "proposal_envelope_missing"
    if stable_hash(envelope) != pending.get("proposal_hash"):
        return "proposal_hash_mismatch"
    if int(pending.get("approval_version") or 0) <= 0:
        return "approval_version_missing"
    if envelope.get("policy_version") != pending.get("policy_version"):
        return "proposal_policy_binding_mismatch"
    if envelope.get("policy_version") != POLICY_VERSION:
        return "policy_version_stale"
    if envelope.get("expires_at") != pending.get("expires_at"):
        return "proposal_expiry_binding_mismatch"
    if action_type == "scientific_experiment_plan":
        if envelope.get("validator_version") != VALIDATOR_VERSION:
            return "validator_version_stale"
        if envelope.get("simulator_version") != SIMULATOR_VERSION:
            return "simulator_version_stale"
        if (envelope.get("simulation_result") or {}).get("status") != "success":
            return "preproposal_simulation_not_successful"
    elif action_type == "email_send":
        if envelope.get("validator_version") != "email-draft-validator-v2":
            return "validator_version_stale"
        if envelope.get("simulator_version") != "not_applicable":
            return "simulator_version_stale"
    else:
        return "unsupported_v2_proposal_type"
    expected = envelope.get("expected_resource_versions") or {}
    if expected != (pending.get("expected_resource_versions") or {}):
        return "expected_resource_versions_binding_mismatch"
    payload_hash = ((pending.get("payload") or {}).get("data") or {}).get("proposal_hash")
    if payload_hash != pending.get("proposal_hash"):
        return "payload_proposal_hash_mismatch"
    if action_type == "scientific_experiment_plan":
        design = dict(envelope.get("design_or_protocol") or {})
        claimed_design_hash = design.get("design_hash")
        if not claimed_design_hash:
            return "design_hash_missing"
        if experiment_design_hash(design) != claimed_design_hash:
            return "design_hash_mismatch"
        data = (pending.get("payload") or {}).get("data") or {}
        run = scientific_db.get_scientific_run(str(data.get("scientific_run_id") or ""))
        if not run:
            return "scientific_run_not_found"
        dataset = scientific_db.get_dataset(
            str(run.get("dataset_id") or ""),
            include_measurements=False,
        )
        if not dataset:
            return "scientific_dataset_not_found"
        if dataset.get("content_hash") != expected.get("dataset_content_hash"):
            return "dataset_version_stale"
        if claimed_design_hash != expected.get("design_hash"):
            return "design_version_stale"
    elif action_type == "email_send":
        data = (pending.get("payload") or {}).get("data") or {}
        draft = data.get("draft") or {}
        draft_hash = stable_hash(draft)
        if draft_hash != data.get("draft_hash"):
            return "draft_hash_mismatch"
        if draft_hash != expected.get("draft_hash"):
            return "draft_version_stale"
    return None
