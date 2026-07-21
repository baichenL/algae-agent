from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredicateDefinition:
    memory_type: str
    singleton: bool = True
    auto_extract: bool = True
    allowed_routes: tuple[str, ...] = ("chat", "knowledge_query")


PREDICATES: dict[str, PredicateDefinition] = {
    "response.language": PredicateDefinition("preference", allowed_routes=("router", "chat", "knowledge_query", "workflow_write", "scientific_task")),
    "response.detail_level": PredicateDefinition("preference", allowed_routes=("chat", "knowledge_query", "workflow_write", "scientific_task")),
    "report.format": PredicateDefinition("preference", allowed_routes=("chat", "knowledge_query", "workflow_write", "scientific_task")),
    "units.preferred": PredicateDefinition("preference", allowed_routes=("chat", "knowledge_query", "workflow_write", "scientific_task")),
    "notification.channel": PredicateDefinition("preference", allowed_routes=("chat", "workflow_write", "scientific_task")),
    "notification.quiet_hours": PredicateDefinition("preference", allowed_routes=("chat", "workflow_write", "scientific_task")),
    "visualization.preference": PredicateDefinition("preference", allowed_routes=("chat", "knowledge_query", "scientific_task")),
    "general.note": PredicateDefinition("note", singleton=False, auto_extract=False, allowed_routes=("chat", "knowledge_query")),
}

_PROHIBITED_PATTERNS = (
    re.compile(r"(?i)\b(password|passwd|api[_ -]?key|access[_ -]?token|secret|private[_ -]?key)\b\s*[:=]"),
    re.compile(r"\b\d{15,19}\b"),
    re.compile(r"\b\d{17}[0-9Xx]\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def normalize_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return canonical_json(value).casefold()


def value_hash(value: Any) -> str:
    return hashlib.sha256(normalize_value(value).encode("utf-8")).hexdigest()


def contains_prohibited_sensitive_data(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in _PROHIBITED_PATTERNS)


def validate_memory_input(memory_type: str, predicate: str, value: Any, *, automatic: bool = False) -> PredicateDefinition:
    definition = PREDICATES.get(str(predicate or ""))
    if definition is None:
        raise ValueError("unsupported_memory_predicate")
    if str(memory_type or definition.memory_type) != definition.memory_type:
        raise ValueError("memory_type_predicate_mismatch")
    if automatic and not definition.auto_extract:
        raise ValueError("predicate_not_auto_extractable")
    rendered = canonical_json(value)
    if not rendered or rendered in {"null", '""'}:
        raise ValueError("memory_value_required")
    if len(rendered) > 1000:
        raise ValueError("memory_value_too_large")
    if contains_prohibited_sensitive_data(rendered):
        raise ValueError("prohibited_sensitive_memory")
    return definition


def searchable_text(predicate: str, value: Any) -> str:
    base = f"{predicate} {canonical_json(value)}".casefold()
    cjk_terms: list[str] = []
    for match in _CJK.findall(base):
        cjk_terms.extend(match[index:index + 2] for index in range(max(0, len(match) - 1)))
        cjk_terms.extend(match)
    return " ".join([base, *cjk_terms])[:2000]


def query_terms(text: str) -> list[str]:
    folded = " ".join(str(text or "").casefold().split())
    terms = re.findall(r"[a-z0-9_.-]+", folded)
    for match in _CJK.findall(folded):
        if len(match) == 1:
            terms.append(match)
        else:
            terms.extend(match[index:index + 2] for index in range(len(match) - 1))
    return list(dict.fromkeys(term for term in terms if term))[:24]


def allowed_predicates_for_route(route_kind: str | None) -> set[str]:
    route = str(route_kind or "chat")
    return {name for name, definition in PREDICATES.items() if route in definition.allowed_routes}
