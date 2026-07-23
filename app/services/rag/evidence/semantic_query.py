import json
import os
import re
from contextvars import ContextVar

from app.core.database import list_rag_recipe_components, list_rag_sop_facts
from app.models.rag_evidence_schema import QueryFrame
from app.models.rag_semantic_schema import (
    EvidenceContract,
    RequestedOutput,
    SemanticQuery,
    SemanticSubquery,
    TypedEntity,
)
from app.services.rag.evidence.entity_normalization import (
    canonicalize_chemical_formula,
    canonicalize_entity_text,
    extract_formula_candidates,
    normalize_unicode_text,
)
from app.services.context import ModelCallTimer, assemble_model_input, context_input_mode, model_call_metrics


_LAST_SEMANTIC_PARSER_DIAGNOSTICS: ContextVar[dict | None] = ContextVar(
    "last_semantic_parser_diagnostics",
    default=None,
)


def current_semantic_parser_diagnostics() -> dict | None:
    value = _LAST_SEMANTIC_PARSER_DIAGNOSTICS.get()
    return dict(value) if value else None


GROUP_ALIASES = {
    "salt_solution": ("salt_solution", "salt solution", "盐溶液", "母液1"),
    "phosphate_solution": ("phosphate_solution", "phosphate solution", "磷酸盐溶液", "磷酸盐", "母液2"),
    "trace_elements": ("trace_elements", "trace elements", "微量元素", "母液3", "hutner"),
    "working_solution": ("working_solution", "working solution", "工作液"),
}


def parse_semantic_query(
    question: str,
    source_constraints: list[str] | None = None,
) -> SemanticQuery:
    local = _parse_typed_local(question, source_constraints)
    if local.confidence >= 0.8:
        return local
    llm_result = _parse_with_structured_llm(question)
    return llm_result or local


def _parse_typed_local(
    question: str,
    source_constraints: list[str] | None = None,
) -> SemanticQuery:
    text = (question or "").strip()
    clauses = _semantic_clauses(text)
    inherited_medium = _link_medium(text)
    subqueries = []
    for index, clause in enumerate(clauses, start=1):
        entities = _link_entities(clause, inherited_medium)
        relation = _infer_relation(clause, entities)
        if relation == "unknown" and len(clauses) > 1:
            relation = _infer_relation(f"{inherited_medium.surface if inherited_medium else ''} {clause}", entities)
        output = _requested_output(clause, relation)
        contract = _evidence_contract(relation)
        subqueries.append(
            SemanticSubquery(
                subquery_id=f"sq{index}",
                original_text=clause,
                intent=relation,
                relation=relation,
                entities=entities,
                constraints=list(source_constraints or []),
                requested_output=output,
                evidence_contract=contract,
                confidence=0.95 if relation != "unknown" else 0.4,
            )
        )
    confidence = min((item.confidence for item in subqueries), default=0.0)
    return SemanticQuery(
        original_question=text,
        effect="knowledge_read",
        subqueries=subqueries,
        confidence=confidence,
    )


def _parse_with_structured_llm(question: str) -> SemanticQuery | None:
    mode = os.getenv("RAG_SEMANTIC_PARSER", "hybrid").strip().lower()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if mode == "local" or not api_key or api_key == "test-key":
        return None
    try:
        from app.core.config import client

        schema = SemanticQuery.model_json_schema()
        system_instruction = (
            "Parse the user knowledge question into the supplied JSON schema. "
            "Use only schema enum values. Never classify or authorize an external action; "
            "effect must be knowledge_read. Split genuinely independent claims into subqueries."
        )
        envelope = None
        if context_input_mode() == "legacy":
            messages = [
                {"role": "system", "content": f"{system_instruction} Schema: {json.dumps(schema, ensure_ascii=False)}"},
                {"role": "user", "content": question},
            ]
        else:
            envelope = assemble_model_input(
                request_kind="knowledge_query",
                route_kind="knowledge_query",
                current_user_message=question,
                system_instruction=system_instruction,
                task_protocol={
                    "task": "Parse a knowledge_read query into SemanticQuery JSON.",
                    "output_schema": schema,
                    "effect": "knowledge_read",
                },
            )
            messages = envelope.to_messages(question)
        timer = ModelCallTimer()
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        if envelope:
            _LAST_SEMANTIC_PARSER_DIAGNOSTICS.set(
                model_call_metrics(envelope, response=response, elapsed_ms=timer.elapsed_ms())
            )
        payload = json.loads(response.choices[0].message.content or "{}")
        payload["original_question"] = question
        payload["effect"] = "knowledge_read"
        payload["parser"] = "structured_llm"
        result = SemanticQuery.model_validate(payload)
        if not result.subqueries:
            return None
        for subquery in result.subqueries:
            if subquery.evidence_contract.relation != subquery.relation:
                return None
        return result
    except Exception as exc:
        if "envelope" in locals() and envelope:
            _LAST_SEMANTIC_PARSER_DIAGNOSTICS.set(
                model_call_metrics(envelope, elapsed_ms=timer.elapsed_ms(), error=exc)
            )
        return None


def semantic_subquery_to_frame(
    subquery: SemanticSubquery,
    source_constraints: list[str] | None = None,
) -> QueryFrame | None:
    relation = subquery.relation
    constraints = _merge(source_constraints or [], subquery.evidence_contract.preferred_sources)
    subject = _entity(subquery, "culture_medium", "subject") or _entity(subquery, "culture_medium")
    if subject and not _entity_is_grounded_in_question(subquery.original_text, subject):
        subject = None
    chemicals = [item for item in subquery.entities if item.entity_type == "chemical"]
    groups = [item for item in subquery.entities if item.entity_type == "recipe_group"]
    condition = next(
        (item for item in subquery.entities if item.role == "condition"),
        None,
    )

    if relation == "has_component_amount" and chemicals:
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="value",
            target_entity=subject.canonical_id if subject else "TAP medium",
            target_attribute="component_amount",
            target_value=chemicals[0].surface,
            answer_shape="value",
            evidence_requirement="structured_schema",
            source_constraint=_merge(constraints, ["media_recipe"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "has_component" and chemicals:
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="existence",
            target_entity=subject.canonical_id if subject else "TAP medium",
            target_attribute="component",
            target_value=chemicals[0].surface,
            answer_shape="yes_no",
            evidence_requirement="structured_schema",
            source_constraint=_merge(constraints, ["media_recipe"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "has_component_group" and groups:
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="list",
            target_entity=subject.canonical_id if subject else "TAP medium",
            target_attribute="component_group",
            target_value=groups[0].canonical_id,
            answer_shape="list",
            evidence_requirement="structured_schema",
            source_constraint=_merge(constraints, ["media_recipe"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "overview_of":
        protocol_material = _entity(subquery, "protocol_material")
        if protocol_material:
            return QueryFrame(
                original_question=subquery.original_text,
                question_type="explanation",
                target_entity="manual",
                target_attribute="protocol_material_overview",
                target_value=protocol_material.canonical_id,
                answer_shape="overview",
                evidence_requirement="explicit_statement",
                source_constraint=_merge(constraints, ["manual"]),
                raw_terms=[item.surface for item in subquery.entities],
            )
        if not subject:
            return None
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="overview",
            target_entity=subject.canonical_id,
            target_attribute="entity_overview",
            answer_shape="overview",
            evidence_requirement="structured_overview",
            source_constraint=_merge(constraints, ["media_recipe"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "used_in_step":
        procedure_entity = next(
            (
                item
                for item in subquery.entities
                if item.entity_type in {"protocol_material", "medium_variant"}
                and item.canonical_id != "tap_medium"
            ),
            None,
        )
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="procedure",
            target_entity="manual",
            target_attribute="entity_procedure" if procedure_entity else "tap_related_operations",
            target_value=procedure_entity.canonical_id if procedure_entity else None,
            answer_shape="procedure",
            evidence_requirement="explicit_statement",
            source_constraint=_merge(constraints, ["manual"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "suitable_for":
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="suitability",
            target_entity=subject.canonical_id if subject else "TAP medium",
            target_attribute="suitability",
            target_value=condition.surface if condition else None,
            answer_shape="yes_no",
            evidence_requirement="explicit_statement",
            source_constraint=constraints,
            raw_terms=[item.surface for item in subquery.entities],
        )
    if relation == "standard_equivalence":
        variant = _entity(subquery, "medium_variant")
        return QueryFrame(
            original_question=subquery.original_text,
            question_type="comparison",
            target_entity=subject.canonical_id if subject else "TAP medium",
            target_attribute="standard_equivalence",
            target_value=variant.canonical_id if variant else None,
            answer_shape="yes_no",
            evidence_requirement="explicit_statement",
            source_constraint=_merge(constraints, ["media_recipe", "manual"]),
            raw_terms=[item.surface for item in subquery.entities],
        )
    return None


def expand_group_subqueries(query: SemanticQuery) -> SemanticQuery:
    expanded = []
    for subquery in query.subqueries:
        groups = [item for item in subquery.entities if item.entity_type == "recipe_group"]
        if subquery.relation != "has_component_group" or len(groups) <= 1:
            expanded.append(subquery)
            continue
        for group in groups:
            entities = [item for item in subquery.entities if item.entity_type != "recipe_group"] + [group]
            expanded.append(
                subquery.model_copy(
                    update={
                        "subquery_id": f"{subquery.subquery_id}:{group.canonical_id}",
                        "entities": entities,
                    }
                )
            )
    return query.model_copy(update={"subqueries": expanded})


def _semantic_clauses(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    parts = [
        item.strip(" ,;，。；、")
        for item in re.split(r"[，。；;、]|\b(?:and|also)\b|以及|并且", normalized)
        if item.strip(" ,;，。；、")
    ]
    return parts or [normalized]


def _link_entities(clause: str, inherited_medium: TypedEntity | None) -> list[TypedEntity]:
    entities = []
    medium = _link_medium(clause) or inherited_medium
    if medium:
        entities.append(medium)
    entities.extend(_link_source_entities(clause))
    entities.extend(_link_recipe_entities(clause))
    entities.extend(_link_sop_entities(clause))
    entities.extend(_link_condition(clause))
    unique = []
    seen = set()
    for item in entities:
        key = (item.entity_type, item.canonical_id, item.role)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _link_medium(text: str) -> TypedEntity | None:
    compact = canonicalize_entity_text(text)
    rows = _recipe_rows()
    candidates = []
    for row in rows:
        entity = str(row.get("entity") or "")
        if entity:
            candidates.append(entity)
    candidates.extend(["TAP medium", "tap_medium"])
    for candidate in dict.fromkeys(candidates):
        aliases = {
            canonicalize_entity_text(candidate),
            canonicalize_entity_text(candidate.replace("_", " ")),
            canonicalize_entity_text(candidate.replace("medium", "")),
        }
        if any(alias and _alias_in_text(alias, text) for alias in aliases):
            return TypedEntity(
                entity_type="culture_medium",
                surface=candidate.replace("_", " "),
                canonical_id=candidate.replace("_", " "),
                role="subject",
            )
    return None


def _link_recipe_entities(text: str) -> list[TypedEntity]:
    normalized = normalize_unicode_text(text)
    compact = canonicalize_entity_text(normalized)
    formula_text = canonicalize_chemical_formula(normalized)
    entities = []
    linked_formula_ids = set()
    for row in _recipe_rows():
        name = str(row.get("component_name") or "")
        canonical = canonicalize_chemical_formula(name)
        name_compact = canonicalize_entity_text(name)
        is_formula = bool(re.search(r"[A-Z].*\d|^[A-Z][a-z]?[A-Z]", normalize_unicode_text(name)))
        matched = canonical and canonical in formula_text if is_formula else name_compact and name_compact in compact
        if matched:
            linked_id = str(row.get("normalized_component_name") or canonical)
            entities.append(
                TypedEntity(
                    entity_type="chemical" if is_formula else "protocol_material",
                    surface=name,
                    canonical_id=(
                        linked_id
                        if is_formula
                        else str(row.get("normalized_component_name") or name)
                    ),
                    metadata={"group_name": row.get("group_name")},
                )
            )
            if is_formula:
                linked_formula_ids.add(canonical)
    for formula in extract_formula_candidates(text):
        canonical = canonicalize_chemical_formula(formula)
        if canonical and canonical not in linked_formula_ids and canonical not in {"tap", "is", "does", "has"}:
            entities.append(
                TypedEntity(
                    entity_type="chemical",
                    surface=formula,
                    canonical_id=canonical,
                    confidence=0.9,
                )
            )
    for group, aliases in GROUP_ALIASES.items():
        match = next((alias for alias in aliases if canonicalize_entity_text(alias) in compact), None)
        if match:
            entities.append(
                TypedEntity(
                    entity_type="recipe_group",
                    surface=match,
                    canonical_id=group,
                )
            )
    return entities


def _link_sop_entities(text: str) -> list[TypedEntity]:
    compact = canonicalize_entity_text(text)
    scored = []
    generic_terms = {"tap", "medium", "培养基", "配制", "用于", "相关", "操作"}
    for row in _sop_rows():
        value = str(row.get("value") or "")
        tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalize_unicode_text(value).casefold()))
        overlap = sum(
            1
            for token in tokens
            if canonicalize_entity_text(token) not in generic_terms
            and canonicalize_entity_text(token) in compact
        )
        if overlap >= 1:
            scored.append((overlap, row))
    if not scored:
        return []
    scored.sort(key=lambda item: -item[0])
    best = scored[0][1]
    entity_id = str(best.get("entity") or "protocol_material")
    entity_type = "medium_variant" if entity_id.endswith("_medium") else "protocol_material"
    return [
        TypedEntity(
            entity_type=entity_type,
            surface=str(best.get("value") or best.get("entity")),
            canonical_id=entity_id,
            metadata={"attribute": best.get("attribute")},
        )
    ]


def _link_condition(text: str) -> list[TypedEntity]:
    normalized = normalize_unicode_text(text)
    match = re.search(r"(?:适合|适用于|suitable\s+for|applicable\s+to)\s*([^?？。]+)", normalized, re.I)
    if not match:
        return []
    surface = match.group(1).strip()
    if not surface:
        return []
    entity_type = "organism" if re.search(r"藻|菌|品系|strain|species", surface, re.I) else "experimental_condition"
    canonical_surface = surface
    if entity_type == "organism":
        canonical_surface = re.sub(r"^(?:所有|全部|任意|任何)\s*", "", canonical_surface)
        canonical_surface = re.sub(r"(?:全部)?品系$", "", canonical_surface).strip()
    return [
        TypedEntity(
            entity_type=entity_type,
            surface=surface,
            canonical_id=canonicalize_entity_text(canonical_surface),
            role="condition",
        )
    ]


def _infer_relation(text: str, entities: list[TypedEntity]) -> str:
    normalized = normalize_unicode_text(text)
    chemicals = [item for item in entities if item.entity_type == "chemical"]
    groups = [item for item in entities if item.entity_type == "recipe_group"]
    materials = [item for item in entities if item.entity_type == "protocol_material"]
    has_medium = any(item.entity_type == "culture_medium" for item in entities)
    has_manual_source = any(item.entity_type == "manual" for item in entities)
    if re.search(r"标准|standard|等同|equivalent", normalized, re.I) and re.search(r"吗|是否|\?", normalized, re.I):
        return "standard_equivalence"
    if has_medium and re.search(r"适合|适用于|suitable|applicable", normalized, re.I):
        return "suitable_for"
    if has_medium and chemicals and re.search(r"用量|浓度|多少|amount|concentration|value", normalized, re.I):
        return "has_component_amount"
    if has_medium and chemicals and re.search(r"有没有|是否有|是否包含|含不含|has|contain|exists?", normalized, re.I):
        return "has_component"
    if groups:
        return "has_component_group"
    if has_manual_source and re.search(r"步骤|场景|操作|流程|哪些|列表|procedure|usage|operation", normalized, re.I):
        return "used_in_step"
    if re.search(r"步骤|使用场景|用在|如何使用|procedure|used\s+in|usage", normalized, re.I):
        return "used_in_step"
    if materials and re.search(r"是什么|介绍|概述|what\s+is|overview", normalized, re.I):
        return "overview_of"
    if re.search(r"是什么|介绍|概述|简介|what\s+is|overview|introduction", normalized, re.I):
        return "overview_of"
    if has_medium:
        return "overview_of"
    return "unknown"


def _requested_output(text: str, relation: str) -> RequestedOutput:
    normalized = text or ""
    shape_by_relation = {
        "has_component_amount": "value",
        "has_component": "yes_no",
        "has_component_group": "list",
        "used_in_step": "procedure",
        "suitable_for": "yes_no",
        "standard_equivalence": "comparison",
        "overview_of": "overview",
    }
    point_match = re.search(r"(?:点|条|points?)\s*(\d+)", normalized, re.I)
    return RequestedOutput(
        shape=shape_by_relation.get(relation, "general"),
        concise=bool(re.search(r"简洁|简要|简短|concise|brief", normalized, re.I)),
        max_points=_parse_point_count(point_match.group(1)) if point_match else None,
    )


def _evidence_contract(relation: str) -> EvidenceContract:
    contracts = {
        "has_component_amount": EvidenceContract(
            relation="has_component_amount",
            required_fact_types=["recipe_component"],
            required_slots=["subject", "component", "amount", "unit"],
            preferred_sources=["media_recipe"],
        ),
        "has_component": EvidenceContract(
            relation="has_component",
            required_fact_types=["recipe_component"],
            required_slots=["subject", "component"],
            preferred_sources=["media_recipe"],
        ),
        "has_component_group": EvidenceContract(
            relation="has_component_group",
            required_fact_types=["recipe_component"],
            required_slots=["subject", "group"],
            preferred_sources=["media_recipe"],
        ),
        "used_in_step": EvidenceContract(
            relation="used_in_step",
            required_fact_types=["sop_fact"],
            required_slots=["subject", "procedure_step"],
            preferred_sources=["manual"],
            direct_statement_required=True,
        ),
        "suitable_for": EvidenceContract(
            relation="suitable_for",
            required_fact_types=["sop_fact", "text_chunk"],
            required_slots=["subject", "condition", "relation"],
            preferred_sources=[],
            direct_statement_required=True,
        ),
        "standard_equivalence": EvidenceContract(
            relation="standard_equivalence",
            required_fact_types=["recipe_component", "sop_fact"],
            required_slots=["subject", "variant", "relation"],
            preferred_sources=["media_recipe", "manual"],
            direct_statement_required=True,
        ),
        "overview_of": EvidenceContract(
            relation="overview_of",
            required_fact_types=["recipe_component", "sop_fact"],
            required_slots=["subject"],
            preferred_sources=[],
        ),
    }
    return contracts.get(relation, EvidenceContract(relation="unknown"))


def _entity(subquery: SemanticSubquery, entity_type: str, role: str | None = None) -> TypedEntity | None:
    return next(
        (
            item
            for item in subquery.entities
            if item.entity_type == entity_type and (role is None or item.role == role)
        ),
        None,
    )


def _entity_is_grounded_in_question(text: str, entity: TypedEntity) -> bool:
    normalized_text = normalize_unicode_text(text or "").casefold()
    compact_text = canonicalize_entity_text(normalized_text)
    canonical = canonicalize_entity_text(entity.canonical_id)
    surface = canonicalize_entity_text(entity.surface)
    if entity.entity_type == "culture_medium" and canonical in {"tapmedium", "tap"}:
        return bool(re.search(r"\bTAP\b|tris[- ]acetate[- ]phosphate", text or "", re.I))
    return any(
        value and value in compact_text
        for value in {
            canonical,
            surface,
            canonicalize_entity_text(entity.canonical_id.replace("_", " ")),
        }
    )


def _merge(current: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in [*current, *additions] if item))


def _link_source_entities(text: str) -> list[TypedEntity]:
    normalized = normalize_unicode_text(text)
    if re.search(r"实验手册|手册|\bSOP\b|\bmanual\b|\bprotocol\b", normalized, re.I):
        return [
            TypedEntity(
                entity_type="manual",
                surface="manual",
                canonical_id="manual",
                role="source",
            )
        ]
    return []


def _alias_in_text(alias: str, text: str) -> bool:
    normalized_alias = normalize_unicode_text(alias).casefold().strip()
    normalized_text = normalize_unicode_text(text).casefold()
    if re.fullmatch(r"[a-z0-9]{1,4}", normalized_alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", normalized_text))
    return canonicalize_entity_text(normalized_alias) in canonicalize_entity_text(normalized_text)


def _parse_point_count(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return digits.get(value)


def _recipe_rows() -> list[dict]:
    try:
        return list_rag_recipe_components(doc_types=["media_recipe"])
    except Exception:
        return []


def _sop_rows() -> list[dict]:
    try:
        return list_rag_sop_facts(doc_types=["manual"])
    except Exception:
        return []
