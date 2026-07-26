from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def experiment_design_hash(payload: dict[str, Any]) -> str:
    """Hash scientific content, excluding run-local identifiers.

    The pending record separately binds the design to a scientific run. Keeping
    run-local IDs outside this digest makes equal data/config/seed produce the
    same design hash while every content change still invalidates approval.
    """
    semantic = {
        key: value
        for key, value in payload.items()
        if key not in {"design_hash", "design_id", "scientific_run_id"}
    }
    return canonical_hash(semantic)


@dataclass(frozen=True)
class GoalContract:
    dataset_id: str
    target_strain_id: str | None = None
    dataset_version: str | None = None
    dataset_content_hash: str | None = None
    target_metric: str = "biomass"
    direction: Literal["maximize", "minimize"] = "maximize"
    target_batch_ids: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    max_cycles: int = 2
    success_threshold: float = 0.02
    run_seed: int = 2025

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScientificPlanNode:
    node_id: str
    skill: str
    tool_name: str
    purpose: str
    depends_on: tuple[str, ...] = ()
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScientificPlanGraph:
    version: int
    nodes: list[ScientificPlanNode]
    active_node_id: str | None = None
    completed_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "active_node_id": self.active_node_id,
            "completed_node_ids": list(self.completed_node_ids),
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [
                {"from": dependency, "to": item.node_id}
                for item in self.nodes
                for dependency in item.depends_on
            ],
        }


@dataclass(frozen=True)
class ScientificObservation:
    step_id: str
    tool_name: str
    status: str
    output: dict[str, Any]
    quality_flags: tuple[str, ...] = ()
    evidence_refs: tuple[dict[str, Any], ...] = ()
    provenance: str = "deterministic_analysis"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    verdict: Literal["pass", "revise", "clarify", "block"]
    checks: tuple[VerificationCheck, ...]
    missing_information: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "checks": [item.to_dict() for item in self.checks],
            "missing_information": list(self.missing_information),
            "unsupported_claims": list(self.unsupported_claims),
        }


@dataclass(frozen=True)
class PlanPatch:
    base_version: int
    new_version: int
    reason: str
    add_nodes: tuple[ScientificPlanNode, ...] = ()
    replace_nodes: tuple[ScientificPlanNode, ...] = ()
    remove_node_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "new_version": self.new_version,
            "reason": self.reason,
            "add_nodes": [item.to_dict() for item in self.add_nodes],
            "replace_nodes": [item.to_dict() for item in self.replace_nodes],
            "remove_node_ids": list(self.remove_node_ids),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class DiagnosisHypothesis:
    hypothesis_id: str
    candidate_cause: str
    score: float
    support_level: str
    data_signal: float
    evidence_support: float
    alternative_exclusion: float
    evidence_refs: tuple[dict[str, Any], ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosisReport:
    anomalies: tuple[dict[str, Any], ...]
    hypotheses: tuple[DiagnosisHypothesis, ...]
    conclusion: str
    causal_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": list(self.anomalies),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "conclusion": self.conclusion,
            "causal_claim": self.causal_claim,
        }


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str
    factors: dict[str, float]
    predicted_value: float
    uncertainty: float
    acquisition_score: float
    role: Literal["candidate", "control"] = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentDesignSpec:
    design_id: str
    scientific_run_id: str
    target_metric: str
    direction: str
    conditions: list[ExperimentCondition]
    replicates: int
    sampling_hours: list[float]
    required_capabilities: list[str]
    evidence_citations: list[dict[str, Any]] = field(default_factory=list)
    simulation_only: bool = True
    design_hash: str = ""

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "design_id": self.design_id,
            "scientific_run_id": self.scientific_run_id,
            "target_metric": self.target_metric,
            "direction": self.direction,
            "conditions": [item.to_dict() for item in self.conditions],
            "replicates": self.replicates,
            "sampling_hours": list(self.sampling_hours),
            "required_capabilities": list(self.required_capabilities),
            "evidence_citations": list(self.evidence_citations),
            "simulation_only": self.simulation_only,
        }
        if include_hash:
            payload["design_hash"] = self.design_hash or canonical_hash(payload)
        return payload

    def freeze(self) -> "ExperimentDesignSpec":
        self.design_hash = experiment_design_hash(self.to_dict(include_hash=False))
        return self


def build_default_plan() -> ScientificPlanGraph:
    nodes = [
        ScientificPlanNode("quality", "growth_anomaly_diagnosis", "scientific_data_quality", "Validate data before scientific inference."),
        ScientificPlanNode("metrics", "growth_anomaly_diagnosis", "scientific_growth_metrics", "Compute deterministic growth metrics.", ("quality",)),
        ScientificPlanNode("anomaly", "growth_anomaly_diagnosis", "scientific_anomaly_detection", "Detect anomalous batches and residuals.", ("metrics",)),
        ScientificPlanNode("evidence", "evidence_grounded_hypothesis", "scientific_evidence_search", "Ground candidate explanations in SOP and literature.", ("anomaly",)),
        ScientificPlanNode("diagnosis", "evidence_grounded_hypothesis", "scientific_hypothesis_ranking", "Rank candidate causes without asserting causality.", ("evidence",)),
        ScientificPlanNode("optimize", "constrained_experiment_optimization", "scientific_experiment_optimize", "Propose informative next conditions.", ("diagnosis",)),
        ScientificPlanNode("validate", "constrained_experiment_optimization", "scientific_design_validate", "Validate resource and capability constraints.", ("optimize",)),
        ScientificPlanNode("proposal", "constrained_experiment_optimization", "scientific_proposal_create", "Freeze a human-reviewable experiment proposal.", ("validate",)),
    ]
    return ScientificPlanGraph(version=1, nodes=nodes, active_node_id="quality")
