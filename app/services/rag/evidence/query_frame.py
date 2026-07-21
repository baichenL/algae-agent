import re

from app.models.rag_evidence_schema import QueryFrame
from app.services.rag.evidence.semantic_query import (
    expand_group_subqueries,
    parse_semantic_query,
    semantic_subquery_to_frame,
)


EXECUTION_PATTERNS = [
    r"执行",
    r"运行",
    r"开始.*流程",
    r"发邮件",
    r"发送邮件",
    r"提醒",
    r"修改",
    r"写入",
    r"更新",
    r"send email",
    r"run workflow",
    r"execute workflow",
]



def parse_query_frame(question: str, source_constraints: list[str] | None = None) -> QueryFrame:
    text = (question or "").strip()
    constraints = list(source_constraints or [])
    lower = text.lower()

    semantic = expand_group_subqueries(parse_semantic_query(text, constraints))
    if len(semantic.subqueries) == 1 and semantic.subqueries[0].confidence >= 0.8:
        semantic_frame = semantic_subquery_to_frame(semantic.subqueries[0], constraints)
        if semantic_frame is not None:
            return semantic_frame

    if _matches_any(text, EXECUTION_PATTERNS):
        return QueryFrame(
            original_question=text,
            question_type="existence",
            target_entity=None,
            target_attribute="write_or_execution_request",
            answer_shape="not_found",
            evidence_requirement="exact_match",
            allow_inference=False,
            source_constraint=constraints,
            raw_terms=_raw_terms(text),
        )

    if _is_table_schema_question(text):
        if _is_experiment_data_analysis_question(text):
            return QueryFrame(
                original_question=text,
                question_type="optimization" if _is_optimization_question(text) else "analysis",
                target_entity="experiment_data",
                target_attribute=_extract_analysis_attribute(text),
                target_value=_extract_data_analysis_target(text),
                answer_shape="not_found",
                evidence_requirement="structured_data_analysis",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["experiment_data"]),
                raw_terms=_raw_terms(text),
            )
        target_value = _extract_column_target(text)
        if _is_column_equivalence_question(text):
            return QueryFrame(
                original_question=text,
                question_type="comparison",
                target_entity="experiment_data",
                target_attribute="column_equivalence",
                target_value=_extract_column_equivalence_target(text),
                answer_shape="yes_no",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["experiment_data"]),
                raw_terms=_raw_terms(text),
            )
        if _is_column_existence_question(text):
            return QueryFrame(
                original_question=text,
                question_type="existence",
                target_entity="experiment_data",
                target_attribute="column",
                target_value=target_value,
                answer_shape="yes_no",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["experiment_data"]),
                raw_terms=_raw_terms(text),
            )
        if _is_list_question(text) or _asks_environment_variables(text):
            return QueryFrame(
                original_question=text,
                question_type="list",
                target_entity="experiment_data",
                target_attribute="environment_variables" if _asks_environment_variables(text) else "column",
                target_value=target_value,
                answer_shape="list",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["experiment_data"]),
                raw_terms=_raw_terms(text),
            )
        if _is_value_question(text):
            return QueryFrame(
                original_question=text,
                question_type="value",
                target_entity="experiment_data",
                target_attribute="column_value",
                target_value=target_value,
                answer_shape="value",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["experiment_data"]),
                raw_terms=_raw_terms(text),
            )

    if _is_manual_question(text):
        if _is_procedure_question(text) or _is_list_question(text):
            return QueryFrame(
                original_question=text,
                question_type="procedure" if not _is_list_question(text) else "list",
                target_entity="manual",
                target_attribute=_extract_procedure_attribute(text),
                answer_shape="procedure" if not _is_list_question(text) else "list",
                evidence_requirement="explicit_statement",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["manual"]),
                raw_terms=_raw_terms(text),
            )

    if _is_paper_question(text):
        return QueryFrame(
            original_question=text,
            question_type="list" if _is_list_question(text) else "existence",
            target_entity="paper",
            target_attribute=_extract_paper_attribute(text),
            answer_shape="list" if _is_list_question(text) else "yes_no",
            evidence_requirement="citation_required",
            allow_inference=False,
            source_constraint=_merge_constraints(constraints, ["paper"]),
            raw_terms=_raw_terms(text),
        )

    if _is_recipe_question(text):
        if _is_recipe_group_list_question(text):
            return QueryFrame(
                original_question=text,
                question_type="list",
                target_entity=_extract_entity(text) or "TAP medium",
                target_attribute="component_group",
                target_value=_extract_recipe_group_target(text),
                answer_shape="list",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["media_recipe"]),
                raw_terms=_raw_terms(text),
            )
        if _is_entity_overview_question(text) or _is_list_question(text):
            return QueryFrame(
                original_question=text,
                question_type="overview",
                target_entity=_extract_entity(text) or "TAP medium",
                target_attribute="entity_overview",
                answer_shape="overview",
                evidence_requirement="structured_overview",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["media_recipe"]),
                raw_terms=_raw_terms(text),
            )
        if _is_value_question(text):
            return QueryFrame(
                original_question=text,
                question_type="value",
                target_entity=_extract_entity(text) or "TAP medium",
                target_attribute="component_amount",
                target_value=_extract_component_target(text),
                answer_shape="value",
                evidence_requirement="structured_schema",
                allow_inference=False,
                source_constraint=_merge_constraints(constraints, ["media_recipe"]),
                raw_terms=_raw_terms(text),
            )

    if _is_suitability_question(text):
        return QueryFrame(
            original_question=text,
            question_type="suitability",
            target_entity=_extract_entity(text),
            target_attribute=_extract_suitability_attribute(text),
            answer_shape="yes_no",
            evidence_requirement="explicit_statement",
            allow_inference=False,
            source_constraint=constraints,
            raw_terms=_raw_terms(text),
        )

    if _is_component_existence_question(text):
        return QueryFrame(
            original_question=text,
            question_type="existence",
            target_entity=_extract_entity(text),
            target_attribute="component",
            target_value=_extract_component_target(text),
            answer_shape="yes_no",
            evidence_requirement="structured_schema",
            allow_inference=False,
            source_constraint=_merge_constraints(constraints, ["media_recipe"]),
            raw_terms=_raw_terms(text),
        )

    if _is_value_question(text):
        return QueryFrame(
            original_question=text,
            question_type="value",
            target_entity=_extract_entity(text),
            target_attribute=_extract_value_attribute(text),
            answer_shape="value",
            evidence_requirement="exact_match",
            allow_inference=False,
            source_constraint=constraints,
            raw_terms=_raw_terms(text),
        )

    if _is_procedure_question(text):
        return QueryFrame(
            original_question=text,
            question_type="procedure",
            target_entity=_extract_entity(text),
            target_attribute=_extract_procedure_attribute(text),
            answer_shape="procedure",
            evidence_requirement="explicit_statement",
            allow_inference=False,
            source_constraint=constraints,
            raw_terms=_raw_terms(text),
        )

    if _is_list_question(text):
        return QueryFrame(
            original_question=text,
            question_type="list",
            target_entity=_extract_entity(text),
            target_attribute="items",
            answer_shape="list",
            evidence_requirement="citation_required",
            allow_inference=False,
            source_constraint=constraints,
            raw_terms=_raw_terms(text),
        )

    return QueryFrame(
        original_question=text,
        question_type="explanation",
        target_entity=_extract_entity(text),
        answer_shape="list",
        evidence_requirement="citation_required",
        allow_inference=False,
        source_constraint=constraints,
        raw_terms=_raw_terms(text),
    )


def is_blocked_frame(frame: QueryFrame) -> bool:
    return frame.target_attribute == "write_or_execution_request"


def _is_table_schema_question(text: str) -> bool:
    lower = text.lower()
    if any(
        marker in lower
        for marker in ["experiment data", "environmental variable", "environment variable"]
    ):
        return True
    return any(
        marker in lower
        for marker in ["实验数据", "表格", "xlsx", "csv", "字段", "列名", "column", "growth curve", "od750", "biomass"]
    )


def _is_experiment_data_analysis_question(text: str) -> bool:
    return bool(
        re.search(
            r"影响|关系|相关|趋势|分组|按.+分|计数|数量|最佳|最优|最高|最低|最大|最小|平均|第\s*\d+\s*行|判断.*条件|best|optimal|effect|correlation|trend|group by|by .+|count|max|min|mean",
            text,
            re.I,
        )
    )


def _is_optimization_question(text: str) -> bool:
    return bool(re.search(r"最佳|最优|最高|最低|最大|最小|best|optimal|max|min", text, re.I))


def _asks_environment_variables(text: str) -> bool:
    return bool(re.search(r"环境变量|environmental variable|变量", text, re.I))


def _is_suitability_question(text: str) -> bool:
    return bool(re.search(r"是否适合|适不适合|是否适用于|适用于|能不能用于|适合|suitable|applicable", text, re.I))


def _is_column_existence_question(text: str) -> bool:
    if re.search(r"has\s+.+\s+column|contains?\s+.+\s+column|column\s+.+\s+exists?", text, re.I):
        return True
    return bool(re.search(r"字段|是否.*包含|有没有|有无|column", text, re.I))


def _is_column_equivalence_question(text: str) -> bool:
    return bool(
        re.search(
            r"鍚屼竴涓瓧娈祙鍚屼竴瀛楁|涓€鏍穦鐩稿悓|绛夊悓|equivalent|same field|same column",
            text,
            re.I,
        )
    )


def _is_manual_question(text: str) -> bool:
    return bool(re.search(r"瀹為獙鎵嬪唽|鎵嬪唽|SOP|manual|protocol|鎿嶄綔|姝ラ|娴佺▼|鐢靛嚮|杞寲|骞虫澘|澶嶈嫃|鎭㈠鍩瑰吇", text, re.I))


def _is_paper_question(text: str) -> bool:
    return bool(
        re.search(
            r"璁烘枃|paper|鏂囩尞|literature|machine learning|deep learning|data-driven|growth prediction|forecasting",
            text,
            re.I,
        )
    )


def _is_recipe_question(text: str) -> bool:
    return bool(
        re.search(
            r"閰嶆柟|鍩瑰吇鍩簗鎴愬垎|缁勫垎|component|recipe|medium|TAP|NaCl|NH4Cl|NH鈧凜l|K2HPO4|K鈧侶PO鈧剕KH2PO4|KH鈧侾O鈧剕MgSO4|CaCl2",
            text,
            re.I,
        )
    )


def _is_entity_overview_question(text: str) -> bool:
    return bool(
        re.search(
            r"浠嬬粛|绠€浠媩姒傝堪|鏄粈涔坾璁茶|璇存槑涓€涓媩浜嗚В涓€涓媩overview|introduction|tell me about|what is",
            text,
            re.I,
        )
    )


def _is_recipe_group_list_question(text: str) -> bool:
    return _is_list_question(text) and bool(
        re.search(r"纾烽吀鐩恷鐩愭憾娑瞸寰噺鍏冪礌|Hutner|trace|phosphate|宸ヤ綔娑瞸姣嶆恫", text, re.I)
    )


def _is_component_existence_question(text: str) -> bool:
    if re.search(r"has\s+.+\s+component|contains?\s+.+\s+component|contain\s+[A-Za-z0-9_\-]+", text, re.I) and bool(
        re.search(r"component|recipe|medium|TAP", text, re.I)
    ):
        return True
    return bool(re.search(r"鏈夋病鏈墊鏄惁鍖呭惈|鏄惁鍚湁|鍖呭惈.*鍚梶鍚笉鍚珅鏈夋棤", text, re.I)) and bool(
        re.search(r"閰嶆柟|鍩瑰吇鍩簗鎴愬垎|缁勫垎|component|recipe|medium|TAP", text, re.I)
    )


def _is_value_question(text: str) -> bool:
    return bool(re.search(r"是多少|多少|用量|参数|浓度|数值|value|amount|parameter", text, re.I))


def _is_procedure_question(text: str) -> bool:
    return bool(re.search(r"怎么做|如何|步骤|流程|操作|用于什么|用于哪个|怎么使用|procedure|protocol", text, re.I))


def _is_list_question(text: str) -> bool:
    return bool(re.search(r"有哪些|哪些|列出|列表|字段|成分|组分|steps|list", text, re.I))


def _extract_column_target(text: str) -> str | None:
    patterns = [
        r"有没有\s*([A-Za-z0-9_\- μµ·.%℃/]+)\s*(?:字段|列|记录)?",
        r"是否(?:有|包含)?\s*([A-Za-z0-9_\- μµ·.%℃/]+)\s*(?:字段|列|记录)?",
        r"(OD750|biomass|pH|CO2|CO₂|temperature|light intensity|I\s*[μµ]?mol[^\s，。？?]*|N mg/L|P mg/L|tf)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip(" ,;:，。？?")
            if value and value not in {"字段", "列", "记录"}:
                return value
    return None


def _extract_analysis_attribute(text: str) -> str:
    if re.search(r"row\s*\d+|第\s*\d+\s*行", text, re.I):
        return "row_lookup"
    if re.search(r"最佳|最优|best|optimal", text, re.I):
        return "best_condition"
    if re.search(r"最高|最大|max", text, re.I):
        return "max_value"
    if re.search(r"最低|最小|min", text, re.I):
        return "min_value"
    if re.search(r"平均|mean|average", text, re.I):
        if re.search(r"分组|按.+分|group by|by\s+(light|temperature|ph|nitrogen|phosphorus)", text, re.I):
            return "group_mean"
        return "mean_value"
    if re.search(r"计数|数量|count|how many", text, re.I):
        return "count_value"
    if re.search(r"影响|关系|相关|effect|correlation", text, re.I):
        return "relationship"
    if re.search(r"趋势|trend", text, re.I):
        return "trend"
    return "analysis"


def _extract_data_analysis_target(text: str) -> str | None:
    row_match = re.search(r"row\s*(\d+)|第\s*(\d+)\s*行", text, re.I)
    terms = _extract_column_like_terms(text)
    lowered_terms = {term.lower() for term in terms}
    if "ph" in lowered_terms and any(term.startswith("biomass") for term in lowered_terms):
        return "pH"
    if row_match and terms:
        row_number = row_match.group(1) or row_match.group(2)
        return f"row:{row_number}|column:{terms[0]}"
    if terms:
        target = terms[0]
        group_by = _extract_group_by_target(text)
        return f"{target}|group_by:{group_by}" if group_by else target
    if re.search(r"biomass|生物量", text, re.I):
        return "biomass"
    return None


def _extract_group_by_target(text: str) -> str | None:
    patterns = [
        r"group by\s+([A-Za-z0-9_\- ]+)",
        r"by\s+(light|temperature|pH|nitrogen|phosphorus)",
        r"按\s*(光照|温度|pH|氮|磷|light|temperature)\s*分",
    ]
    aliases = {
        "光照": "light",
        "温度": "temperature",
        "氮": "nitrogen",
        "磷": "phosphorus",
    }
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            raw = match.group(1).strip(" ,;:，。？?")
            return aliases.get(raw, raw)
    return None


def _extract_column_equivalence_target(text: str) -> str | None:
    candidates = _extract_column_like_terms(text)
    if len(candidates) >= 2:
        return "|".join(candidates[:2])
    return None


def _extract_column_like_terms(text: str) -> list[str]:
    patterns = [
        r"OD750",
        r"\bOD\b",
        r"biomass(?:\s*g/L)?",
        r"pH",
        r"CO2",
        r"CO₂",
        r"temperature",
        r"tf",
        r"I\s*[μµ]?mol[^\s，。？?]*",
        r"N\s*mg/L",
        r"P\s*mg/L",
    ]
    terms = []
    lowered_terms = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            term = match.group(0).strip()
            lowered = term.lower()
            if term and lowered not in lowered_terms:
                terms.append(term)
                lowered_terms.add(lowered)
    return terms


def _extract_component_target(text: str) -> str | None:
    patterns = [
        r"(?:has|contain|contains)\s+([A-Za-z0-9_\-]+)\s+(?:component|ingredient)?",
        r"([A-Za-z0-9_\-]+)\s+(?:component|ingredient)",
        r"有没有\s*([A-Za-z0-9_\- μµ·.%℃/]+)",
        r"是否(?:包含|含有)?\s*([A-Za-z0-9_\- μµ·.%℃/]+)",
        r"(NaCl|FeSO4|FeSO₄|K2HPO4|K₂HPO₄|KH2PO4|KH₂PO₄|NH4Cl|NH₄Cl|MgSO4|MgSO₄|CaCl2|CaCl₂|Tris|冰乙酸)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = match.group(1).strip(" ,;:，。？?")
            if value:
                return value
    return None


def _extract_recipe_group_target(text: str) -> str | None:
    if re.search(r"phosphate|K2HPO4|KH2PO4|K₂HPO₄|KH₂PO₄|磷酸盐", text, re.I):
        return "phosphate_solution"
    if re.search(r"Hutner|trace|EDTA|ZnSO4|FeSO4|微量|元素", text, re.I):
        return "trace_elements"
    if re.search(r"NH4Cl|MgSO4|CaCl2|salt|盐溶液|盐", text, re.I):
        return "salt_solution"
    if re.search(r"working|Tris|工作液|母液|1\s*L", text, re.I):
        return "working_solution"
    return None


def _extract_entity(text: str) -> str | None:
    if re.search(r"\bTAP\b|培养基", text, re.I):
        return "TAP medium"
    if re.search(r"实验数据|表格|xlsx|csv", text, re.I):
        return "experiment_data"
    if re.search(r"手册|SOP|manual|protocol", text, re.I):
        return "manual"
    if re.search(r"论文|paper|文献", text, re.I):
        return "paper"
    return None


def _extract_suitability_attribute(text: str) -> str | None:
    if re.search(r"黑暗|暗培养|dark", text, re.I):
        return "dark_culture"
    if re.search(r"Chlorella|小球藻|所有|全部", text, re.I):
        return "strain_applicability"
    return "suitability"


def _extract_value_attribute(text: str) -> str | None:
    target = _extract_column_target(text)
    if target:
        return target
    if re.search(r"OD750", text, re.I):
        return "OD750"
    if re.search(r"电击|转化", text, re.I):
        return "electroporation_parameter"
    return None


def _extract_procedure_attribute(text: str) -> str | None:
    if re.search(r"恢复培养|recovery", text, re.I):
        return "recovery_culture"
    if re.search(r"潮霉素|平板|涂板|筛选|hygromycin|plating|screening", text, re.I):
        return "plating_or_screening"
    if re.search(r"OD750|电击|转化|蔗糖|electroporation|transformation|sucrose", text, re.I):
        return "electroporation"
    if re.search(r"TAP|培养基", text, re.I) and re.search(r"操作|步骤|operation|procedure", text, re.I):
        return "tap_related_operations"
    return "procedure"


def _extract_paper_attribute(text: str) -> str:
    if re.search(r"OD750|OD 750|optical density", text, re.I):
        return "od750_prediction"
    if re.search(r"directly applicable|project.*(?:culture|cultivation).*parameter|project.*parameter", text, re.I):
        return "project_parameter"
    if re.search(r"鍙洿鎺鏈」鐩畖鍩瑰吇鍙傛暟|鎵ц鍙傛暟|姝ｅ紡瀹為獙|鐩存帴閲囩敤", text, re.I):
        return "project_parameter"
    if re.search(r"妯″瀷|鏂规硶|machine learning|deep learning|data-driven|棰勬祴|prediction|forecast", text, re.I):
        return "method_used"
    return "topic_presence"


def _raw_terms(text: str) -> list[str]:
    terms = []
    for raw in re.split(r"\s+|[锛屻€傦紒锟?,.;锛涳細:()锛堬級]", text):
        token = raw.strip()
        if token and token not in terms:
            terms.append(token)
    return terms


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _merge_constraints(current: list[str], additions: list[str]) -> list[str]:
    merged = []
    for item in [*current, *additions]:
        if item and item not in merged:
            merged.append(item)
    return merged
