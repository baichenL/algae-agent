from app.models.rag_evidence_schema import EvidencePlan, QueryFrame


def build_evidence_plan(frame: QueryFrame) -> EvidencePlan:
    if frame.target_attribute == "write_or_execution_request":
        return EvidencePlan(
            required_capabilities=[],
            preferred_evidence_types=[],
            source_constraints=frame.source_constraint,
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.evidence_requirement == "structured_schema" and frame.target_attribute in {
        "column",
        "environment_variables",
        "column_equivalence",
        "column_value",
    }:
        capability_by_attribute = {
            "column": "column_existence",
            "environment_variables": "environment_variable_list",
            "column_equivalence": "column_equivalence",
            "column_value": "column_value",
        }
        capability = capability_by_attribute[frame.target_attribute]
        return EvidencePlan(
            required_capabilities=[capability],
            preferred_evidence_types=["table_schema"],
            source_constraints=frame.source_constraint or ["experiment_data"],
            fallback_allowed=False,
            closed_world_required=True,
        )

    if frame.evidence_requirement == "structured_schema" and frame.target_attribute == "component":
        return EvidencePlan(
            required_capabilities=["component_existence"],
            preferred_evidence_types=["recipe_component"],
            source_constraints=frame.source_constraint or ["media_recipe"],
            fallback_allowed=False,
            closed_world_required=True,
        )

    if frame.evidence_requirement == "structured_overview" and frame.target_attribute == "entity_overview":
        if frame.target_entity and "medium" in frame.target_entity.lower():
            return EvidencePlan(
                required_capabilities=["entity_overview"],
                preferred_evidence_types=["recipe_component", "text_chunk"],
                source_constraints=frame.source_constraint or ["media_recipe"],
                fallback_allowed=True,
                closed_world_required=False,
            )
        return EvidencePlan(
            required_capabilities=["entity_overview"],
            preferred_evidence_types=["text_chunk"],
            source_constraints=frame.source_constraint,
            fallback_allowed=True,
            closed_world_required=False,
        )

    if frame.target_attribute == "standard_equivalence":
        return EvidencePlan(
            required_capabilities=["explicit_standard_equivalence"],
            preferred_evidence_types=["recipe_component", "sop_fact"],
            source_constraints=frame.source_constraint or ["media_recipe", "manual"],
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.target_attribute == "protocol_material_overview":
        return EvidencePlan(
            required_capabilities=["protocol_material_fact"],
            preferred_evidence_types=["sop_fact"],
            source_constraints=frame.source_constraint or ["manual"],
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.evidence_requirement == "structured_schema" and frame.target_attribute in {
        "component_amount",
        "component_group",
    }:
        capability = "component_amount" if frame.target_attribute == "component_amount" else "component_group"
        return EvidencePlan(
            required_capabilities=[capability],
            preferred_evidence_types=["recipe_component"],
            source_constraints=frame.source_constraint or ["media_recipe"],
            fallback_allowed=False,
            closed_world_required=True,
        )

    if frame.evidence_requirement == "structured_data_analysis":
        return EvidencePlan(
            required_capabilities=["structured_data_analysis"],
            preferred_evidence_types=["data_analysis"],
            source_constraints=frame.source_constraint or ["experiment_data"],
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.target_entity == "paper":
        return EvidencePlan(
            required_capabilities=[frame.target_attribute or "paper_fact"],
            preferred_evidence_types=["paper_fact"],
            source_constraints=frame.source_constraint or ["paper"],
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.question_type == "suitability":
        return EvidencePlan(
            required_capabilities=["explicit_statement"],
            preferred_evidence_types=["sop_fact", "text_chunk"],
            source_constraints=frame.source_constraint,
            fallback_allowed=True,
            closed_world_required=False,
        )

    if frame.question_type == "procedure":
        return EvidencePlan(
            required_capabilities=["procedure_step"],
            preferred_evidence_types=["sop_fact"],
            source_constraints=frame.source_constraint,
            fallback_allowed=False,
            closed_world_required=False,
        )

    if frame.target_entity == "manual" and frame.question_type == "list":
        return EvidencePlan(
            required_capabilities=["procedure_step"],
            preferred_evidence_types=["sop_fact"],
            source_constraints=frame.source_constraint or ["manual"],
            fallback_allowed=False,
            closed_world_required=False,
        )

    return EvidencePlan(
        required_capabilities=["general_context"],
        preferred_evidence_types=["text_chunk"],
        source_constraints=frame.source_constraint,
        fallback_allowed=True,
        closed_world_required=False,
    )
