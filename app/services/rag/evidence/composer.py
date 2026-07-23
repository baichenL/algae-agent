import re

from app.models.rag_evidence_schema import (
    AnswerabilityResult,
    EvidenceUnit,
    GroundedAnswer,
    GroundedCitation,
    QuestionAspects,
    QueryFrame,
    SufficiencyResult,
)
from app.services.rag.evidence.evidence_sufficiency import requires_direct_support
from app.services.rag.evidence.adapters.recipe_adapter import normalize_component_name, prettify_component_name
from app.services.rag.evidence.adapters.table_schema_adapter import normalize_column_name
from app.services.rag.table_rag import aggregate_data_values


def compose_grounded_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
    aspects: QuestionAspects | None = None,
    sufficiency: SufficiencyResult | None = None,
) -> GroundedAnswer:
    usable = _usable_evidence(answerability, evidence)
    if answerability.status == "blocked":
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="该请求涉及执行层或事实层写操作，RAG 知识层不能处理。",
            evidence=[],
            uncertainty=["请走 pending approval / workflow / email 的受控流程。"],
            citations=[],
        )

    if frame.target_attribute == "standard_equivalence":
        return _compose_standard_equivalence_answer(frame, answerability, usable or evidence)

    if frame.target_entity == "paper" and frame.target_attribute == "project_parameter":
        return _compose_paper_answer(frame, answerability, usable or evidence)

    if _should_gate_for_sufficiency(aspects, sufficiency, answerability):
        # Lab answers must refuse or downgrade when direct SOP/data evidence is missing;
        # this is a safety boundary, not just generic hallucination control.
        insufficiency = sufficiency or SufficiencyResult()
        selected = _compact_background_evidence(
            _evidence_by_ids(usable or evidence, set(insufficiency.background_evidence_ids))
        )
        missing = "、".join(_missing_aspect_label(item) for item in insufficiency.missing_aspects) or "关键语义要素"
        target = frame.target_value or "所询问的条件"
        subject = _display_entity_name(frame.target_entity or "该对象")
        if frame.question_type == "suitability" and selected:
            conclusion = (
                f"当前知识库没有直接证据（no direct evidence）证明 {subject} 适合{target}。"
                "检索到的资料只能作为背景，不能据此形成完整适用性结论。"
            )
        else:
            conclusion = f"当前知识库没有足够的直接证据（no direct evidence）支持该结论；缺少：{missing}。"
        return GroundedAnswer(
            answerability=AnswerabilityResult(
                status="partial" if selected else "not_found",
                reason=insufficiency.reason or "insufficient_direct_evidence",
                usable_evidence_ids=[item.evidence_id for item in selected],
                missing_evidence=list(insufficiency.missing_aspects),
                closed_world_negative=False,
            ),
            direct_answer=conclusion,
            evidence=selected,
            uncertainty=[
                "背景资料只能说明相关性或使用记录，不能替代直接的适用性、迁移性或因果证据。",
            ],
            citations=_build_citations(selected),
        )
    if frame.evidence_requirement == "structured_schema" and frame.target_attribute in {
        "column",
        "environment_variables",
        "column_equivalence",
        "column_value",
    }:
        return _compose_table_schema_answer(frame, answerability, usable or evidence)

    if frame.evidence_requirement == "structured_schema" and frame.target_attribute in {
        "component",
        "component_amount",
        "component_group",
    }:
        return _compose_recipe_answer(frame, answerability, usable or evidence)

    if frame.evidence_requirement == "structured_overview" and frame.target_attribute == "entity_overview":
        return _compose_entity_overview(frame, answerability, usable or evidence)

    if frame.evidence_requirement == "structured_data_analysis":
        return _compose_data_analysis_answer(frame, answerability, usable or evidence)

    if frame.target_entity == "manual":
        return _compose_sop_answer(frame, answerability, usable or evidence)

    if frame.target_entity == "paper":
        return _compose_paper_answer(frame, answerability, usable or evidence)

    if frame.question_type == "suitability" and answerability.status == "not_found":
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="不能确认。当前知识库没有检索到足够的直接证据支持该适用性结论。",
            evidence=[],
            uncertainty=["需要补充直接 SOP、protocol 或明确适用性声明。"],
            citations=[],
        )

    if answerability.status in {"not_found", "partial"}:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前知识库没有形成足够可引用的直接回答。",
            evidence=usable,
            uncertainty=["检索结果可能只提供背景信息，不能替代直接证据。"],
            citations=_build_citations(usable),
        )

    return GroundedAnswer(
        answerability=answerability,
        direct_answer="当前知识库只能给出有限结论，请结合引用来源复核。",
        evidence=usable,
        uncertainty=[],
        citations=_build_citations(usable),
    )

def _should_gate_for_sufficiency(
    aspects: QuestionAspects | None,
    sufficiency: SufficiencyResult | None,
    answerability: AnswerabilityResult,
) -> bool:
    if not aspects or not sufficiency:
        return False
    if answerability.closed_world_negative:
        return False
    if not requires_direct_support(aspects):
        return False
    return sufficiency.status in {"background_only", "insufficient", "contradicted"}


def _evidence_by_ids(evidence: list[EvidenceUnit], ids: set[str]) -> list[EvidenceUnit]:
    if not ids:
        return []
    return [item for item in evidence if item.evidence_id in ids]


def _compact_background_evidence(evidence: list[EvidenceUnit], limit: int = 3) -> list[EvidenceUnit]:
    selected = []
    seen = set()
    for item in evidence:
        key = (
            item.source_file,
            item.location.get("page_number"),
            item.location.get("section"),
            item.fact_type,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _missing_aspect_label(aspect: str) -> str:
    return {
        "subject": "对象",
        "condition": "条件",
        "relation": "明确关系",
        "target": "目标",
    }.get(aspect, aspect)


def _display_entity_name(value: str) -> str:
    words = value.replace("_", " ").split()
    return " ".join(word.upper() if len(word) <= 4 and word.isascii() else word for word in words)


def _has_local_usage_context(item: EvidenceUnit, target: str) -> bool:
    text = item.text_span or item.value or ""
    target_terms = [term for term in re.split(r"\s+", target.casefold()) if term]
    for segment in re.split(r"[。！？!?\n]", text):
        lower = segment.casefold()
        if "tap" not in lower:
            continue
        if not all(term in lower for term in target_terms):
            continue
        if re.search(r"but\s+no|no\s+(?:direct\s+)?(?:statement|evidence)|not\s+suitable|没有.*(?:证据|说明)|不能证明", segment, re.I):
            continue
        if re.search(r"用于|使用|培养|配制|used?\s+(?:in|for)|cultivated?\s+(?:in|under)", segment, re.I):
            return True
    return False


def _compose_table_schema_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    table_columns = [item for item in evidence if item.fact_type == "table_column"]
    if frame.target_attribute == "column_equivalence":
        return _compose_column_equivalence_answer(frame, answerability, table_columns)
    if frame.target_attribute == "column_value":
        return _compose_column_value_answer(frame, answerability, table_columns)

    if frame.answer_shape == "list":
        selected = table_columns
        label = "字段"
        if frame.target_attribute == "environment_variables":
            selected = [
                item
                for item in table_columns
                if item.metadata.get("inferred_role")
                in {"light_or_irradiance", "nitrogen", "phosphorus", "temperature", "tf", "ph", "co2"}
            ]
            label = "环境变量字段"
        if not selected:
            return GroundedAnswer(
                answerability=answerability,
                direct_answer=f"当前结构化表格 schema 中没有可确认的{label}。",
                evidence=[],
                uncertainty=["该判断只基于已索引表格 schema。"],
                citations=[],
            )
        lines = [f"根据结构化表格 schema，{label}包括："]
        for item in selected:
            role = item.metadata.get("inferred_role")
            role_text = f" (role={role})" if role and role != "unknown" else ""
            lines.append(f"- {item.value}{role_text}")
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="\n".join(lines),
            evidence=selected,
            uncertainty=["该列表来自表格 schema，不会补充未出现的列。"],
            citations=_build_citations(selected),
        )

    target = frame.target_value or "目标字段"
    exact = _matching_columns(table_columns, target)
    if exact:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer=f"当前结构化表格 schema 中存在字段：{exact[0].value}。",
            evidence=exact,
            uncertainty=["字段存在不等于已经回答某个具体行或样本的数值。"],
            citations=_build_citations(exact),
        )

    related_note = ""
    if table_columns and normalize_column_name(target) in {"od750", "od"}:
        biomass = [item.value for item in table_columns if item.metadata.get("inferred_role") == "biomass"]
        if biomass:
            related_note = f" 检索到相关生长数据字段 {', '.join(biomass)}，但它不能等同于 OD750。"
    direct = f"当前结构化表格 schema 中未发现字段：{target}。{related_note}".strip()
    return GroundedAnswer(
        answerability=answerability,
        direct_answer=direct,
        evidence=table_columns,
        uncertainty=["该判断基于真实列名 schema；不会用相近字段替代目标字段。"],
        citations=_build_citations(table_columns),
    )


def _compose_sop_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    facts = [item for item in evidence if item.fact_type == "sop_fact"]
    if not facts:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前没有检索到可引用的 SOP 操作证据。",
            evidence=[],
            uncertainty=["需要补充或重新索引实验手册/SOP。"],
            citations=[],
        )
    lines = ["根据实验手册，相关的操作包括："]
    for item in _unique_facts(facts)[:8]:
        value = item.value or item.text_span or item.attribute or item.entity or "SOP fact"
        lines.append(f"- {value}")
    return GroundedAnswer(
        answerability=answerability,
        direct_answer="\n".join(lines),
        evidence=facts,
        uncertainty=["SOP 证据仅说明手册中的操作记录，正式执行仍需走审批流程。"],
        citations=_build_citations(facts),
    )


def _compose_paper_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    facts = [item for item in evidence if item.fact_type == "paper_claim"]
    if not facts:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前本地论文库没有检索到可引用的论文证据。",
            evidence=[],
            uncertainty=["论文来源仅作背景参考，不能替代实验室 SOP 或审批后的培养方案。"],
            citations=[],
        )
    titles = _paper_titles(facts)
    lines = ["根据当前本地论文库，命中以下与问题相关的论文："]
    for title in titles[:8]:
        lines.append(f"- {title}")
    return GroundedAnswer(
        answerability=answerability,
        direct_answer="\n".join(lines),
        evidence=facts,
        uncertainty=["论文来源适合作为方法参考，不能直接替代本实验室 SOP、配方或审批后的培养方案。"],
        citations=_build_citations(facts),
    )


def _compose_data_analysis_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    values = [item for item in evidence if item.fact_type == "data_value"]
    if not values:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前没有检索到可用于计算的结构化实验数据。",
            evidence=[],
            uncertainty=["请确认 CSV/XLSX 已入库并抽取了目标字段。"],
            citations=[],
        )
    target = frame.target_value or "biomass"
    matching = _matching_data_values(values, target) or values
    if frame.target_attribute in {"mean_value", "max_value", "min_value", "best_condition", "count_value", "trend", "group_mean"}:
        result = aggregate_data_values(values, target, frame.target_attribute)
        if result.status == "answered":
            direct = f"根据结构化实验数据，{target} 的 {frame.target_attribute} = {result.value}。"
            evidence = result.evidence
            for item in evidence:
                item.metadata["aggregation_trace"] = result.trace
        else:
            direct = f"当前结构化实验数据不足以计算 {target} 的 {frame.target_attribute}：{result.reason}。"
            evidence = matching[:12]
    else:
        direct = f"当前检索到 {len(matching)} 条 {target} 相关结构化数据值。"
        evidence = matching[:12]
    return GroundedAnswer(
        answerability=answerability,
        direct_answer=direct,
        evidence=evidence,
        uncertainty=["该计算仅基于当前已索引的结构化实验数据。"],
        citations=_build_citations(evidence),
    )


def _compose_standard_equivalence_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    variant_facts = [
        item
        for item in evidence
        if item.fact_type == "sop_fact" and item.entity == frame.target_value
    ]
    recipe_components = [item for item in evidence if item.fact_type == "recipe_component"]
    selected = []
    if variant_facts:
        selected.extend(variant_facts[:2])
    if recipe_components:
        selected.extend(_recipe_group_evidence(recipe_components))
    if variant_facts:
        direct = (
            "当前证据显示该名称可作为 TAP 相关标准或变体讨论，"
            "但仍需以引用的 SOP 或配方版本为准。"
        )
    else:
        direct = "当前证据不足以确认它与 TAP 标准完全等同。"
    return GroundedAnswer(
        answerability=answerability,
        direct_answer=direct,
        evidence=selected,
        uncertainty=["标准等同关系需要直接来源确认，不能只靠相似名称推断。"],
        citations=_build_citations(selected),
    )


def _compose_recipe_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    components = [item for item in evidence if item.fact_type == "recipe_component"]
    if not components:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前没有检索到可用的结构化配方成分证据。",
            evidence=[],
            uncertainty=["需要先索引配方表或 SOP 中的成分表。"],
            citations=[],
        )
    if frame.target_attribute == "component_group":
        group_items = [item for item in components if item.metadata.get("group_name") == frame.target_value]
        selected = group_items or components
        return _compose_recipe_group_answer(frame, answerability, selected)
    if frame.target_attribute in {"component", "component_amount"}:
        selected = _matching_recipe_components(components, frame.target_value or "")
        if not selected:
            names = _recipe_component_names(components)
            return GroundedAnswer(
                answerability=answerability,
                direct_answer=(
                    f"没有。当前结构化配方证据中未发现 {frame.target_value or '目标组分'}；"
                    f"已索引成分包括：{', '.join(names)}。"
                ),
                evidence=components,
                uncertainty=["该结论仅限当前已索引配方版本；不会用相似名称替代目标组分。"],
                citations=_build_citations(components),
            )
        lines = []
        for item in selected:
            amount = _recipe_amount_text(item)
            lines.append(f"{prettify_component_name(item.value or item.attribute or '')}: {amount}".strip())
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="\n".join(lines),
            evidence=selected,
            uncertainty=["该结论仅限当前已索引的配方版本。"],
            citations=_build_citations(selected),
        )
    return GroundedAnswer(
        answerability=answerability,
        direct_answer="当前结构化配方证据不足以回答该配方问题。",
        evidence=components,
        uncertainty=["请补充明确的组分、用量或组分组查询条件。"],
        citations=_build_citations(components),
    )


def _compose_entity_overview(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    evidence: list[EvidenceUnit],
) -> GroundedAnswer:
    components = [item for item in evidence if item.fact_type == "recipe_component"]
    text_evidence = [item for item in evidence if item.fact_type == "text_chunk"]
    if not components and text_evidence:
        descriptions = []
        for item in text_evidence[:3]:
            text = " ".join((item.text_span or item.value or "").split())
            if text and text not in descriptions:
                descriptions.append(text[:240])
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前只命中文本片段，未抽取到结构化配方表：\n" + "\n".join(f"- {item}" for item in descriptions),
            evidence=text_evidence[:3],
            uncertainty=["建议补充或重新索引结构化配方表。"],
            citations=_build_citations(text_evidence[:3]),
        )
    if not components:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="当前没有可用的结构化配方证据。",
            evidence=[],
            uncertainty=["需要先完成配方文档索引。"],
            citations=[],
        )
    grouped: dict[str, list[EvidenceUnit]] = {}
    for item in components:
        grouped.setdefault(item.metadata.get("group_name") or "other", []).append(item)
    lines = [f"TAP（Tris-Acetate-Phosphate）配方结构："]
    for group, items in grouped.items():
        names = ", ".join(_recipe_component_names(_unique_recipe_components(items))[:8])
        lines.append(f"- {_recipe_group_label(group)}: {names}")
    return GroundedAnswer(
        answerability=answerability,
        direct_answer="\n".join(lines),
        evidence=components,
        uncertainty=["上述内容来自结构化配方证据，母液用量不等同于最终工作液浓度。"],
        citations=_build_citations(components),
    )


def _unique_recipe_components(components: list[EvidenceUnit]) -> list[EvidenceUnit]:
    seen = set()
    unique = []
    for item in components:
        key = (
            item.metadata.get("group_name"),
            item.metadata.get("normalized_component_name"),
            item.metadata.get("amount_text"),
            item.unit,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _recipe_group_evidence(components: list[EvidenceUnit]) -> list[EvidenceUnit]:
    unique = _unique_recipe_components(components)
    grouped: dict[str, list[EvidenceUnit]] = {}
    for item in unique:
        grouped.setdefault(item.metadata.get("group_name") or "other", []).append(item)
    result = []
    for group in ["working_solution", "salt_solution", "phosphate_solution", "trace_elements", "other"]:
        items = grouped.get(group) or []
        if not items:
            continue
        details = []
        for item in items:
            name = prettify_component_name(item.value or "")
            amount = _recipe_amount_text(item)
            details.append(f"{name} ({amount})" if amount else name)
        result.append(
            _aggregate_recipe_group_evidence(
                group,
                _recipe_group_label(group),
                details,
                items[0],
            )
        )
    return result


def _aggregate_recipe_group_evidence(
    group: str,
    label: str,
    details: list[str],
    source: EvidenceUnit,
) -> EvidenceUnit:
    return source.model_copy(
        update={
            "evidence_id": f"recipe_overview:{source.source_file}:{group}",
            "fact_type": "recipe_group_summary",
            "attribute": group,
            "value": label,
            "text_span": f"{label}: {'; '.join(details)}",
            "metadata": {**source.metadata, "overview_group": group},
        }
    )


def _compose_recipe_group_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    components: list[EvidenceUnit],
) -> GroundedAnswer:
    group = frame.target_value
    group_items = [item for item in components if item.metadata.get("group_name") == group]
    if not group_items:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer=f"没有。当前结构化配方证据中未发现组分组：{group or '目标组'}。",
            evidence=components,
            uncertainty=["该结论仅限当前已索引配方版本。"],
            citations=_build_citations(components),
        )

    label = _recipe_group_label(group)
    lines = [f"{label}包含以下组分："]
    for item in group_items:
        amount = _recipe_amount_text(item)
        suffix = f"：{amount}" if amount else ""
        lines.append(f"- {prettify_component_name(item.value or '')}{suffix}")
    return GroundedAnswer(
        answerability=answerability,
        direct_answer="\n".join(lines),
        evidence=group_items,
        uncertainty=["上述用量按来源表格原文呈现，母液用量不等同于最终工作液浓度。"],
        citations=_build_citations(group_items),
    )


def _matching_recipe_components(components: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
    normalized_target = normalize_component_name(target)
    return [
        item
        for item in components
        if item.metadata.get("normalized_component_name") == normalized_target
    ]


def _recipe_component_names(components: list[EvidenceUnit]) -> list[str]:
    names = []
    for item in components:
        name = prettify_component_name(item.value or "")
        if name and name not in names:
            names.append(name)
    return names


def _recipe_amount_text(item: EvidenceUnit) -> str:
    amount_text = item.metadata.get("amount_text")
    unit = item.unit
    if not amount_text:
        return ""
    if unit and not any(char.isalpha() for char in str(amount_text)):
        return f"{amount_text} {unit}"
    formatted = prettify_component_name(str(amount_text))
    formatted = re.sub(r"(?<=\d)\s*(?:ml|mL)\b", " mL", formatted)
    formatted = re.sub(r"(?<=\d)\s*g\b", " g", formatted)
    formatted = re.sub(r"(?<=\d)\s*L\b", " L", formatted)
    return formatted


def _recipe_group_label(group: str | None) -> str:
    labels = {
        "salt_solution": "盐溶液（母液1）",
        "phosphate_solution": "磷酸盐溶液（母液2）",
        "trace_elements": "Hutner's 微量元素（母液3）",
        "working_solution": "TAP 工作液配制",
    }
    return labels.get(group or "", group or "配方表")


def _compose_column_value_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    table_columns: list[EvidenceUnit],
) -> GroundedAnswer:
    target = frame.target_value or "目标字段"
    if not table_columns:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer=f"不能确认。当前没有可用表格 schema 来判断字段 {target} 是否存在，因此不能回答它的数值。",
            evidence=[],
            uncertainty=["需要先完成实验数据表格 schema 回填或重新索引。"],
            citations=[],
        )

    matching = _matching_columns(table_columns, target)
    if not matching:
        real_columns = [item.value for item in table_columns]
        related_note = ""
        if normalize_column_name(target) in {"od750", "od"}:
            biomass = [item.value for item in table_columns if item.metadata.get("inferred_role") == "biomass"]
            if biomass:
                related_note = f" 检索到相关生长数据字段 {', '.join(biomass)}，但它不能等同于 {target}。"
        direct = (
            f"没有。当前结构化表格 schema 中未发现字段：{target}，因此不能回答它的数值。"
            f"{related_note}\n当前真实列名包括：{', '.join(real_columns)}。"
        )
        return GroundedAnswer(
            answerability=answerability,
            direct_answer=direct,
            evidence=table_columns,
            uncertainty=["这是基于完整表格 schema 的字段级判断；不会用相近字段替代目标字段。"],
            citations=_build_citations(table_columns),
        )

    direct = (
        f"字段存在，但不能给出单一数值。当前表格 schema 中存在字段：{matching[0].value}；"
        "这个问题还需要指定行、时间点、样本，或聚合方式（例如最大值、平均值、最新记录）。"
    )
    return GroundedAnswer(
        answerability=answerability,
        direct_answer=direct,
        evidence=matching,
        uncertainty=["M2 当前只用结构化 schema 判断字段和值请求边界，不会从多行实验数据中随意挑一个值。"],
        citations=_build_citations(matching),
    )


def _compose_column_equivalence_answer(
    frame: QueryFrame,
    answerability: AnswerabilityResult,
    table_columns: list[EvidenceUnit],
) -> GroundedAnswer:
    if not table_columns:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="不能确认。当前没有可用表格 schema 来判断这两个字段是否等同。",
            evidence=[],
            uncertainty=["需要先完成实验数据表格 schema 回填或重新索引。"],
            citations=[],
        )

    targets = [item.strip() for item in (frame.target_value or "").split("|") if item.strip()]
    if len(targets) < 2:
        return GroundedAnswer(
            answerability=answerability,
            direct_answer="不能确认。问题中没有识别出两个可比较的字段名。",
            evidence=table_columns,
            uncertainty=["请明确写出两个字段名，例如：biomass 和 OD750 是否是同一个字段。"],
            citations=_build_citations(table_columns),
        )

    first, second = targets[:2]
    first_matches = _matching_columns(table_columns, first)
    second_matches = _matching_columns(table_columns, second)
    real_columns = [item.value for item in table_columns]

    if first_matches and second_matches and first_matches[0].evidence_id == second_matches[0].evidence_id:
        direct = f"是。当前表格 schema 中 {first} 和 {second} 指向同一个真实字段：{first_matches[0].value}。"
        evidence = [first_matches[0]]
    else:
        first_text = first_matches[0].value if first_matches else "未发现"
        second_text = second_matches[0].value if second_matches else "未发现"
        direct = (
            f"不是同一个字段。当前表格 schema 中，{first} 匹配结果：{first_text}；"
            f"{second} 匹配结果：{second_text}。不能把二者等同。"
            f"\n当前真实列名包括：{', '.join(real_columns)}。"
        )
        evidence = table_columns

    return GroundedAnswer(
        answerability=answerability,
        direct_answer=direct,
        evidence=evidence,
        uncertainty=["该判断只基于真实列名 schema；字段的实验学含义仍应以原始数据字典或 SOP 为准。"],
        citations=_build_citations(evidence),
    )


def _matching_columns(table_columns: list[EvidenceUnit], target: str) -> list[EvidenceUnit]:
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


def _unique_facts(facts: list[EvidenceUnit]) -> list[EvidenceUnit]:
    seen = set()
    unique = []
    for item in facts:
        key = item.value or item.text_span or item.evidence_id
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _paper_titles(facts: list[EvidenceUnit]) -> list[str]:
    titles = []
    for item in facts:
        title = item.metadata.get("paper_title") or item.source_file
        if title not in titles:
            titles.append(title)
    return titles


def _paper_attribute_label(attribute: str | None) -> str:
    labels = {
        "machine_learning": "machine learning",
        "deep_learning": "deep learning",
        "data_driven_modeling": "data-driven modeling",
        "growth_prediction": "growth prediction",
        "cultivation_optimization": "cultivation optimization",
        "od750_prediction": "OD750 / optical density",
        "project_parameter": "project parameter",
    }
    return labels.get(attribute or "", attribute or "paper fact")


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
    import re

    row_match = re.search(r"row:(\d+)", target)
    column_match = re.search(r"column:([^|]+)", target)
    row_index = int(row_match.group(1)) if row_match else None
    column = column_match.group(1).strip() if column_match else None
    return row_index, column


def _format_row_context(row_values: list[EvidenceUnit]) -> str:
    preferred_roles = {"time", "light_or_irradiance", "nitrogen", "phosphorus", "tf", "temperature", "ph", "co2"}
    parts = []
    for item in row_values:
        role = item.metadata.get("inferred_role")
        if role in preferred_roles:
            parts.append(f"{item.attribute}={item.value}")
    return "对应条件：" + "；".join(parts) + "。" if parts else ""


def _usable_evidence(answerability: AnswerabilityResult, evidence: list[EvidenceUnit]) -> list[EvidenceUnit]:
    if not answerability.usable_evidence_ids:
        return evidence
    wanted = set(answerability.usable_evidence_ids)
    return [item for item in evidence if item.evidence_id in wanted]


def _build_citations(evidence: list[EvidenceUnit]) -> list[GroundedCitation]:
    citations = []
    for index, item in enumerate(evidence, start=1):
        citations.append(
            GroundedCitation(
                source_id=index,
                evidence_id=item.evidence_id,
                source_file=item.source_file,
                source_type=item.source_type,
                location=item.location,
            )
        )
    return citations
