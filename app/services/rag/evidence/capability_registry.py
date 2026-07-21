from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceCapability:
    evidence_type: str
    can_answer: tuple[str, ...]
    closed_world: bool = False


CAPABILITIES = {
    "table_schema": EvidenceCapability(
        evidence_type="table_schema",
        can_answer=("column_existence", "column_list", "environment_variable_list", "column_value", "column_equivalence"),
        closed_world=True,
    ),
    "recipe_component": EvidenceCapability(
        evidence_type="recipe_component",
        can_answer=("component_existence", "component_amount", "component_list", "component_group"),
        closed_world=True,
    ),
    "sop_fact": EvidenceCapability(
        evidence_type="sop_fact",
        can_answer=("procedure_step", "material_used", "parameter_value", "explicit_statement"),
        closed_world=False,
    ),
    "paper_fact": EvidenceCapability(
        evidence_type="paper_fact",
        can_answer=("method_used", "topic_presence", "summary_context"),
        closed_world=False,
    ),
    "text_chunk": EvidenceCapability(
        evidence_type="text_chunk",
        can_answer=("explanation", "general_context", "explicit_statement_candidate"),
        closed_world=False,
    ),
}


def get_capability(evidence_type: str) -> EvidenceCapability | None:
    return CAPABILITIES.get(evidence_type)


def is_closed_world(evidence_type: str) -> bool:
    capability = get_capability(evidence_type)
    return bool(capability and capability.closed_world)
