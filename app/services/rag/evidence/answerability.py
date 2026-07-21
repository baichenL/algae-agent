import re

from app.models.rag_evidence_schema import AnswerabilityResult, EvidencePlan, EvidenceUnit, QueryFrame
from app.services.rag.evidence.adapters.recipe_adapter import normalize_component_name
from app.services.rag.evidence.adapters.table_schema_adapter import normalize_column_name
from app.services.rag.evidence.query_frame import is_blocked_frame


ENVIRONMENT_ROLES = {
    "light_or_irradiance",
    "nitrogen",
    "phosphorus",
    "temperature",
    "tf",
    "ph",
    "co2",
}


def check_answerability(
    frame: QueryFrame,
    plan: EvidencePlan,
    evidence: list[EvidenceUnit],
) -> AnswerabilityResult:
    if is_blocked_frame(frame):
        return AnswerabilityResult(
            status="blocked",
            reason="rag_read_only_boundary",
        )

    if frame.target_attribute == "standard_equivalence":
        usable = [
            item
            for item in evidence
            if item.fact_type in {"recipe_component", "sop_fact", "text_chunk"}
        ]
        return AnswerabilityResult(
            status="not_found",
            reason="no_explicit_standard_equivalence_statement",
            usable_evidence_ids=[item.evidence_id for item in usable],
            missing_evidence=["explicit_standard_equivalence"],
        )

    if "table_schema" in plan.preferred_evidence_types:
        return _check_table_schema_answerability(frame, evidence)

    if "recipe_component" in plan.preferred_evidence_types:
        if frame.target_attribute == "entity_overview":
            components = [item for item in evidence if item.fact_type == "recipe_component"]
            text_chunks = [item for item in evidence if item.fact_type == "text_chunk"]
            return AnswerabilityResult(
                status="answered" if components else ("partial" if text_chunks else "not_found"),
                reason=(
                    "structured_entity_overview_available"
                    if components
                    else ("text_only_entity_overview_available" if text_chunks else "no_structured_overview_evidence")
                ),
                usable_evidence_ids=[item.evidence_id for item in (components or text_chunks)],
                missing_evidence=[] if components else ["recipe_component"],
            )
        return _check_recipe_answerability(frame, evidence)

    if frame.evidence_requirement == "structured_data_analysis":
        return _check_data_analysis_answerability(frame, evidence)

    if "sop_fact" in plan.preferred_evidence_types:
        return _check_sop_answerability(frame, evidence)

    if "paper_fact" in plan.preferred_evidence_types:
        return _check_paper_answerability(frame, evidence)

    if frame.question_type == "suitability":
        explicit = _explicit_statement_evidence(frame, evidence)
        if explicit:
            return AnswerabilityResult(
                status="answered",
                reason="explicit_statement_found",
                usable_evidence_ids=[item.evidence_id for item in explicit],
            )
        return AnswerabilityResult(
            status="not_found",
            reason="no_explicit_suitability_statement",
            missing_evidence=["explicit_statement"],
        )

    if evidence:
        return AnswerabilityResult(
            status="partial",
            reason="only_open_world_text_evidence_available",
            usable_evidence_ids=[item.evidence_id for item in evidence],
        )

    return AnswerabilityResult(
        status="not_found",
        reason="no_evidence_found",
        missing_evidence=plan.required_capabilities,
    )


def select_usable_evidence(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> list[EvidenceUnit]:
    if answerability.usable_evidence_ids:
        wanted = set(answerability.usable_evidence_ids)
        return [item for item in evidence if item.evidence_id in wanted]

    if frame.target_attribute == "environment_variables":
        return [
            item
            for item in evidence
            if item.fact_type == "table_column"
            and (item.metadata.get("inferred_role") in ENVIRONMENT_ROLES)
        ]

    if frame.target_attribute == "column":
        return [item for item in evidence if item.fact_type == "table_column"]

    if frame.target_attribute == "column_value":
        return [item for item in evidence if item.fact_type == "table_column"]

    if frame.target_attribute == "column_equivalence":
        return [item for item in evidence if item.fact_type == "table_column"]

    if frame.evidence_requirement == "structured_data_analysis":
        return [item for item in evidence if item.fact_type == "data_value"]

    return evidence


def _check_table_schema_answerability(frame: QueryFrame, evidence: list[EvidenceUnit]) -> AnswerabilityResult:
    table_columns = [item for item in evidence if item.fact_type == "table_column"]
    if not table_columns:
        return AnswerabilityResult(
            status="not_found",
            reason="no_table_schema_evidence",
            missing_evidence=["table_schema"],
        )

    if frame.question_type == "list":
        usable = table_columns
        if frame.target_attribute == "environment_variables":
            usable = [
                item
                for item in table_columns
                if item.metadata.get("inferred_role") in ENVIRONMENT_ROLES
            ]
        return AnswerabilityResult(
            status="answered" if usable else "not_found",
            reason="table_schema_list_answered" if usable else "no_matching_schema_columns",
            usable_evidence_ids=[item.evidence_id for item in usable],
            missing_evidence=[] if usable else ["matching_schema_columns"],
            closed_world_negative=not bool(usable),
        )

    if frame.question_type == "comparison" and frame.target_attribute == "column_equivalence":
        if not frame.target_value or "|" not in frame.target_value:
            return AnswerabilityResult(
                status="partial",
                reason="column_equivalence_question_without_two_targets",
                usable_evidence_ids=[item.evidence_id for item in table_columns],
                missing_evidence=["two_target_columns"],
            )
        return AnswerabilityResult(
            status="answered",
            reason="table_schema_column_equivalence_answered",
            usable_evidence_ids=[item.evidence_id for item in table_columns],
        )

    if frame.question_type == "value" and frame.target_attribute == "column_value":
        target = frame.target_value
        if not target:
            return AnswerabilityResult(
                status="partial",
                reason="column_value_question_without_target_value",
                usable_evidence_ids=[item.evidence_id for item in table_columns],
                missing_evidence=["target_value"],
            )
        matching = _matching_table_columns(table_columns, target)
        if not matching:
            return AnswerabilityResult(
                status="answered",
                reason="table_schema_complete_value_column_not_found",
                usable_evidence_ids=[item.evidence_id for item in table_columns],
                missing_evidence=[target],
                closed_world_negative=True,
            )
        return AnswerabilityResult(
            status="partial",
            reason="column_exists_but_value_query_needs_row_or_aggregation",
            usable_evidence_ids=[item.evidence_id for item in matching],
            missing_evidence=["row_selector_or_aggregation_rule"],
        )

    target = frame.target_value
    if not target:
        return AnswerabilityResult(
            status="partial",
            reason="column_existence_question_without_target_value",
            usable_evidence_ids=[item.evidence_id for item in table_columns],
            missing_evidence=["target_value"],
        )

    normalized_target = normalize_column_name(target)
    exact = _matching_table_columns(table_columns, target)
    if exact:
        return AnswerabilityResult(
            status="answered",
            reason="exact_table_column_found",
            usable_evidence_ids=[item.evidence_id for item in exact],
        )

    related = _related_table_columns(normalized_target, table_columns)
    return AnswerabilityResult(
        status="answered",
        reason="table_schema_complete_column_not_found",
        usable_evidence_ids=[item.evidence_id for item in related],
        missing_evidence=[target],
        closed_world_negative=True,
    )


def _related_table_columns(normalized_target: str, table_columns: list[EvidenceUnit]) -> list[EvidenceUnit]:
    related_roles = set()
    if normalized_target in {"od750", "od"}:
        related_roles.add("biomass")
    if normalized_target in {"ph"}:
        related_roles.update({"temperature", "co2"})
    return [
        item
        for item in table_columns
        if item.metadata.get("inferred_role") in related_roles
    ]


def _check_recipe_answerability(frame: QueryFrame, evidence: list[EvidenceUnit]) -> AnswerabilityResult:
    components = [item for item in evidence if item.fact_type == "recipe_component"]
    if not components:
        return AnswerabilityResult(
            status="not_found",
            reason="no_structured_recipe_component_evidence",
            missing_evidence=["recipe_component"],
        )

    if frame.target_attribute == "component_group":
        if not frame.target_value:
            return AnswerabilityResult(
                status="partial",
                reason="recipe_group_question_without_target_group",
                usable_evidence_ids=[item.evidence_id for item in components],
                missing_evidence=["target_group"],
            )
        group_items = [
            item
            for item in components
            if item.metadata.get("group_name") == frame.target_value
        ]
        return AnswerabilityResult(
            status="answered" if group_items else "not_found",
            reason="recipe_group_answered" if group_items else "recipe_group_not_found",
            usable_evidence_ids=[item.evidence_id for item in (group_items or components)],
            missing_evidence=[] if group_items else [frame.target_value],
            closed_world_negative=not bool(group_items),
        )

    if frame.target_attribute in {"component", "component_amount"}:
        if not frame.target_value:
            return AnswerabilityResult(
                status="partial",
                reason="recipe_component_question_without_target_value",
                usable_evidence_ids=[item.evidence_id for item in components],
                missing_evidence=["target_component"],
            )
        matching = _matching_recipe_components(components, frame.target_value)
        return AnswerabilityResult(
            status="answered",
            reason="recipe_component_found" if matching else "recipe_component_not_found_in_complete_recipe",
            usable_evidence_ids=[item.evidence_id for item in (matching or components)],
            missing_evidence=[] if matching else [frame.target_value],
            closed_world_negative=not bool(matching),
        )

    return AnswerabilityResult(
        status="partial",
        reason="unsupported_recipe_question_frame",
        usable_evidence_ids=[item.evidence_id for item in components],
        missing_evidence=[frame.target_attribute or "recipe_question"],
    )


def _matching_recipe_components(components: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
    normalized_target = normalize_component_name(target)
    return [
        item
        for item in components
        if item.metadata.get("normalized_component_name") == normalized_target
    ]


def _check_sop_answerability(frame: QueryFrame, evidence: list[EvidenceUnit]) -> AnswerabilityResult:
    facts = [item for item in evidence if item.fact_type == "sop_fact"]
    if not facts:
        return AnswerabilityResult(status="not_found", reason="no_sop_fact_evidence", missing_evidence=["sop_fact"])
    selected = _matching_sop_facts(frame, facts)
    return AnswerabilityResult(
        status="answered" if selected else "not_found",
        reason="sop_fact_answered" if selected else "no_matching_sop_fact",
        usable_evidence_ids=[item.evidence_id for item in (selected or facts)],
        missing_evidence=[] if selected else [frame.target_attribute or "sop_fact"],
    )


def _check_paper_answerability(frame: QueryFrame, evidence: list[EvidenceUnit]) -> AnswerabilityResult:
    facts = [item for item in evidence if item.fact_type == "paper_claim"]
    if not facts:
        return AnswerabilityResult(status="not_found", reason="no_paper_fact_evidence", missing_evidence=["paper_fact"])
    if frame.target_attribute == "project_parameter":
        matching = [item for item in facts if item.attribute == "project_parameter"]
        return AnswerabilityResult(
            status="answered" if matching else "not_found",
            reason="paper_project_parameter_found" if matching else "no_project_parameter_claim",
            usable_evidence_ids=[item.evidence_id for item in (matching or facts)],
            missing_evidence=[] if matching else ["project_parameter"],
        )
    matching = [item for item in facts if item.attribute == frame.target_attribute]
    if frame.target_attribute == "method_used":
        matching = [item for item in facts if item.attribute in {"machine_learning", "deep_learning", "data_driven_modeling", "growth_prediction", "cultivation_optimization"}]
    return AnswerabilityResult(
        status="answered" if matching else "not_found",
        reason="paper_fact_answered" if matching else "no_matching_paper_fact",
        usable_evidence_ids=[item.evidence_id for item in (matching or facts)],
        missing_evidence=[] if matching else [frame.target_attribute or "paper_fact"],
    )


def _check_data_analysis_answerability(frame: QueryFrame, evidence: list[EvidenceUnit]) -> AnswerabilityResult:
    values = [item for item in evidence if item.fact_type == "data_value"]
    if not values:
        return AnswerabilityResult(status="not_found", reason="no_experiment_data_values", missing_evidence=["data_values"])
    if frame.target_attribute == "row_lookup":
        row_index, column = _parse_row_lookup_target(frame.target_value)
        if row_index is None or not column:
            return AnswerabilityResult(status="partial", reason="row_lookup_missing_row_or_column", missing_evidence=["row_index", "column"])
        matches = [item for item in _matching_data_values(values, column) if item.location.get("row_index") == row_index]
        return AnswerabilityResult(
            status="answered" if matches else "not_found",
            reason="row_lookup_answered" if matches else "row_or_column_not_found",
            usable_evidence_ids=[item.evidence_id for item in (matches or values)],
            missing_evidence=[] if matches else [frame.target_value or "row_lookup"],
        )
    if frame.target_attribute in {"max_value", "min_value", "mean_value", "best_condition"}:
        target = frame.target_value or "biomass"
        matching = _numeric_data_values(values, target)
        return AnswerabilityResult(
            status="answered" if matching else "not_found",
            reason="data_analysis_ready" if matching else "target_numeric_column_not_found",
            usable_evidence_ids=[
                item.evidence_id for item in (values if matching and frame.target_attribute == "best_condition" else (matching or values))
            ],
            missing_evidence=[] if matching else [target],
        )
    if frame.target_attribute in {"relationship", "trend", "analysis"}:
        target = frame.target_value
        if target and not _matching_data_values(values, target):
            return AnswerabilityResult(
                status="not_found",
                reason="analysis_target_column_not_found",
                usable_evidence_ids=[item.evidence_id for item in values],
                missing_evidence=[target],
            )
        return AnswerabilityResult(
            status="partial",
            reason="advanced_data_analysis_not_supported",
            usable_evidence_ids=[item.evidence_id for item in values],
            missing_evidence=["correlation_or_modeling_adapter"],
        )
    return AnswerabilityResult(status="partial", reason="unsupported_data_analysis_frame", missing_evidence=[frame.target_attribute or "analysis"])


def _matching_sop_facts(frame: QueryFrame, facts: list[EvidenceUnit]) -> list[EvidenceUnit]:
    target = frame.target_attribute or ""
    if target == "entity_procedure":
        return [item for item in facts if item.entity == frame.target_value]
    if target == "protocol_material_overview":
        return [item for item in facts if item.entity == frame.target_value]
    if target == "recovery_culture":
        return [item for item in facts if item.entity == "recovery_culture"]
    if target == "plating_or_screening":
        return [item for item in facts if item.entity in {"hygromycin_tap_plate", "recovery_culture"}]
    if target == "electroporation":
        return [item for item in facts if item.entity in {"electroporation", "sucrose_tap_medium", "recovery_culture"}]
    if target == "tap_related_operations":
        return [item for item in facts if "TAP" in (item.value or item.text_span or "")]
    return facts


def _matching_data_values(values: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
    normalized = normalize_column_name(target)
    role_aliases = {"biomass": "biomass", "ph": "ph", "od750": "od", "od": "od"}
    wanted_role = role_aliases.get(normalized)
    return [
        item
        for item in values
        if normalize_column_name(item.attribute or "") == normalized
        or normalize_column_name(item.attribute or "").startswith(f"{normalized}_")
        or (wanted_role is not None and item.metadata.get("inferred_role") == wanted_role)
    ]


def _numeric_data_values(values: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
    return [item for item in _matching_data_values(values, target) if item.metadata.get("numeric_value") is not None]


def _parse_row_lookup_target(target: str | None) -> tuple[int | None, str | None]:
    if not target:
        return None, None
    row_match = re.search(r"row:(\d+)", target)
    column_match = re.search(r"column:([^|]+)", target)
    row_index = int(row_match.group(1)) if row_match else None
    column = column_match.group(1).strip() if column_match else None
    return row_index, column


def _matching_table_columns(table_columns: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
    normalized_target = normalize_column_name(target)
    role_aliases = {
        "biomass": "biomass",
        "od750": "od",
        "od": "od",
        "ph": "ph",
        "co2": "co2",
        "temperature": "temperature",
        "tf": "tf",
    }
    wanted_role = role_aliases.get(normalized_target)
    return [
        item
        for item in table_columns
        if normalize_column_name(item.value or "") == normalized_target
        or normalize_column_name(item.value or "").startswith(f"{normalized_target}_")
        or (wanted_role is not None and item.metadata.get("inferred_role") == wanted_role)
    ]


def _explicit_statement_evidence(frame: QueryFrame, evidence: list[EvidenceUnit]) -> list[EvidenceUnit]:
    target_attribute = frame.target_attribute or ""
    matches = []
    for item in evidence:
        text = item.text_span or ""
        if target_attribute == "dark_culture" and re.search(r"黑暗|dark", text, re.I):
            matches.append(item)
        elif target_attribute == "strain_applicability" and re.search(r"all|所有|全部|Chlorella|小球�", text, re.I):
            matches.append(item)
        elif target_attribute == "suitability" and re.search(r"适合|适用于|suitable|applicable", text, re.I):
            matches.append(item)
    return matches
