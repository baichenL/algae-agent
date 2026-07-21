from __future__ import annotations

from dataclasses import dataclass, field

from app.models.rag_evidence_schema import EvidenceUnit


@dataclass(frozen=True)
class TableAggregationResult:
    status: str
    operation: str
    target: str
    value: float | str | None = None
    evidence: list[EvidenceUnit] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    reason: str | None = None


def aggregate_data_values(
    values: list[EvidenceUnit],
    target: str,
    operation: str,
) -> TableAggregationResult:
    matching = [
        item
        for item in values
        if _matches_target(item, target) and item.metadata.get("numeric_value") is not None
    ]
    trace = {
        "engine": "sqlite_extracted_values",
        "operation": operation,
        "target": target,
        "candidate_count": len(values),
        "numeric_match_count": len(matching),
    }
    if not matching:
        return TableAggregationResult(
            status="not_found",
            operation=operation,
            target=target,
            trace=trace,
            reason="target_numeric_column_not_found",
        )
    if operation == "count_value":
        return TableAggregationResult("answered", operation, target, len(matching), matching[:12], trace)
    if operation == "mean_value":
        value = sum(float(item.metadata["numeric_value"]) for item in matching) / len(matching)
        return TableAggregationResult("answered", operation, target, value, matching, trace)
    if operation == "min_value":
        best = min(matching, key=lambda item: float(item.metadata["numeric_value"]))
        return TableAggregationResult("answered", operation, target, best.metadata["numeric_value"], [best], trace)
    if operation in {"max_value", "best_condition"}:
        best = max(matching, key=lambda item: float(item.metadata["numeric_value"]))
        return TableAggregationResult("answered", operation, target, best.metadata["numeric_value"], [best], trace)
    if operation == "trend":
        return _trend_result(matching, target, trace)
    if operation == "group_mean":
        return _group_mean_result(values, matching, target, trace)
    return TableAggregationResult(
        status="partial",
        operation=operation,
        target=target,
        evidence=matching[:12],
        trace=trace,
        reason="unsupported_table_aggregation",
    )


def _trend_result(matching: list[EvidenceUnit], target: str, trace: dict) -> TableAggregationResult:
    ordered = sorted(matching, key=lambda item: int(item.location.get("row_index") or item.row_start or 0))
    first = float(ordered[0].metadata["numeric_value"])
    last = float(ordered[-1].metadata["numeric_value"])
    if last > first:
        trend = "increasing"
    elif last < first:
        trend = "decreasing"
    else:
        trend = "flat"
    return TableAggregationResult(
        status="answered",
        operation="trend",
        target=target,
        value=trend,
        evidence=[ordered[0], ordered[-1]] if len(ordered) > 1 else ordered,
        trace={**trace, "first_value": first, "last_value": last, "trend_method": "first_last_numeric_value"},
    )


def _group_mean_result(
    values: list[EvidenceUnit],
    matching: list[EvidenceUnit],
    target: str,
    trace: dict,
) -> TableAggregationResult:
    target_name, group_role = _parse_group_target(target)
    if not group_role:
        return TableAggregationResult(
            status="partial",
            operation="group_mean",
            target=target,
            evidence=matching[:12],
            trace=trace,
            reason="group_by_column_not_specified",
        )
    groups: dict[str, list[float]] = {}
    evidence = []
    values_by_row = _values_by_row(values)
    for item in matching:
        row_key = _row_key(item)
        group_value = _group_value(values_by_row.get(row_key, []), group_role)
        if group_value is None:
            continue
        groups.setdefault(group_value, []).append(float(item.metadata["numeric_value"]))
        evidence.append(item)
    if not groups:
        return TableAggregationResult(
            status="not_found",
            operation="group_mean",
            target=target,
            trace={**trace, "group_by": group_role},
            reason="group_values_not_found",
        )
    means = {group: sum(items) / len(items) for group, items in groups.items()}
    return TableAggregationResult(
        status="answered",
        operation="group_mean",
        target=target_name,
        value=means,
        evidence=evidence[:12],
        trace={**trace, "group_by": group_role, "group_count": len(groups)},
    )


def _values_by_row(values: list[EvidenceUnit]) -> dict[tuple, list[EvidenceUnit]]:
    grouped: dict[tuple, list[EvidenceUnit]] = {}
    for item in values:
        grouped.setdefault(_row_key(item), []).append(item)
    return grouped


def _row_key(item: EvidenceUnit) -> tuple:
    return (
        item.source_file,
        item.location.get("sheet_name") or item.sheet_name,
        item.location.get("row_index") or item.row_start,
    )


def _group_value(row_values: list[EvidenceUnit], group_role: str) -> str | None:
    wanted = _normalize(group_role)
    aliases = {
        "light": {"light", "light_or_irradiance", "irradiance"},
        "temperature": {"temperature", "temp"},
        "nitrogen": {"nitrogen", "n", "n_mg_l"},
        "phosphorus": {"phosphorus", "p", "p_mg_l"},
        "ph": {"ph"},
    }.get(wanted, {wanted})
    for item in row_values:
        role = _normalize(item.metadata.get("inferred_role") or "")
        attribute = _normalize(item.attribute or "")
        if role in aliases or attribute in aliases:
            return str(item.value)
    return None


def _parse_group_target(target: str) -> tuple[str, str | None]:
    if "|group_by:" not in str(target or ""):
        return target, None
    target_name, group_by = str(target).split("|group_by:", 1)
    return target_name or "value", group_by or None


def _matches_target(item: EvidenceUnit, target: str) -> bool:
    normalized = _normalize(_parse_group_target(target)[0])
    attribute = _normalize(item.attribute or "")
    role = _normalize(item.metadata.get("inferred_role") or "")
    aliases = {
        "biomass": {"biomass"},
        "od750": {"od", "od750"},
        "od": {"od", "od750"},
        "ph": {"ph"},
    }
    wanted = aliases.get(normalized, {normalized})
    return attribute in wanted or any(attribute.startswith(f"{term}_") for term in wanted) or role in wanted


def _normalize(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "")).strip("_")
