# app/services/intent/write_action_parser.py
# 负责解析解析新增、修改、删除品系这类写操作，提取相关字段并验证必要性
import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.intent.input_normalizer import normalize_text_value


WRITE_TOOL_BY_OPERATION = {
    "add": "add_algae_strain",
    "update": "update_algae_strain",
    "delete": "delete_algae_strain",
}

UPDATE_FIELD_NAMES = {
    "new_strain_id",
    "name_cn",
    "name_en",
    "generation_number",
    "days_since_last_subculture",
}


def parse_write_action(
    message: str,
    context_snapshot: Any,
    strain_id: Optional[str] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    text = normalize_text_value(message)
    candidates = candidates or find_strain_candidates(text, context_snapshot)
    operation = extract_write_operation(text)
    candidate_strain_id = single_candidate_id(candidates)
    target_strain_id = candidate_strain_id or strain_id

    if operation == "add":
        fields = parse_add_fields(text)
        fields.setdefault("generation_number", 1)
        fields.setdefault("days_since_last_subculture", 0)
    elif operation == "update":
        fields = parse_update_fields(text)
        update_target_id = candidate_strain_id or fields.get("strain_id") or strain_id
        if update_target_id:
            fields["strain_id"] = update_target_id
    elif operation == "delete":
        fields = {"strain_id": target_strain_id} if target_strain_id else {}
    else:
        fields = {}

    complete, missing_fields = validate_required_fields(operation, fields)
    return {
        "operation": operation,
        "tool_name": WRITE_TOOL_BY_OPERATION.get(operation),
        "strain_id": fields.get("strain_id") or target_strain_id,
        "fields": fields,
        "missing_fields": [] if complete else missing_fields,
        "candidates": candidates,
    }


def parse_pending_field_updates(
    operation: Optional[str],
    message: str,
    context_snapshot: Any,
    missing_fields: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract only schema-shaped values that can advance an active form."""

    missing_fields = missing_fields or []
    candidates: List[Dict[str, Any]] = []
    if operation == "add":
        fields = parse_add_fields(message)
        if len(missing_fields) == 1:
            missing = missing_fields[0]
            if missing == "strain_id" and not fields.get("strain_id"):
                strain_id = extract_any_strain_id(message)
                if strain_id:
                    fields["strain_id"] = strain_id
            elif missing == "name_en" and not fields.get("name_en"):
                latin_name = _extract_bare_latin_name(message)
                if latin_name:
                    fields["name_en"] = latin_name
        return fields, candidates
    if operation == "update":
        fields = parse_update_fields(message)
        candidates = find_strain_candidates(message, context_snapshot)
        if not fields.get("strain_id") and single_candidate_id(candidates):
            fields["strain_id"] = single_candidate_id(candidates)
        return fields, candidates
    if operation == "delete":
        candidates = find_strain_candidates(message, context_snapshot)
        strain_id = single_candidate_id(candidates) or extract_any_strain_id(message)
        return ({"strain_id": strain_id} if strain_id else {}), candidates
    return {}, candidates


def parse_add_fields(message: str) -> Dict[str, Any]:
    text = normalize_text_value(message)
    fields: Dict[str, Any] = {}

    pipe_fields = _parse_pipe_add_format(text)
    if pipe_fields:
        fields.update(pipe_fields)

    strain_id = extract_any_strain_id(text)
    if strain_id:
        fields.setdefault("strain_id", strain_id)

    name_cn = _extract_value_after_keywords(text, ["中文名", "中文名称", "名称", "名字"])
    if name_cn:
        fields.setdefault("name_cn", name_cn)

    name_en = _extract_latin_name(text)
    if name_en:
        fields.setdefault("name_en", name_en)

    target_name = extract_target_name(text)
    if target_name:
        fields.setdefault("name_cn", target_name)

    generation_number = _extract_number_after_keywords(text, ["代数", "第"])
    if generation_number is not None:
        fields["generation_number"] = generation_number

    days_since_last_subculture = _extract_number_after_keywords(
        text,
        ["传代天数", "距传代", "距离上次传代", "天数"],
    )
    if days_since_last_subculture is not None:
        fields["days_since_last_subculture"] = days_since_last_subculture

    return fields


def parse_update_fields(message: str) -> Dict[str, Any]:
    text = normalize_text_value(message)
    fields: Dict[str, Any] = {}

    new_strain_id = extract_new_strain_id(text)
    if new_strain_id:
        fields["new_strain_id"] = new_strain_id

    name_cn = _extract_value_after_keywords(text, ["中文名", "中文名称"])
    if name_cn:
        fields["name_cn"] = name_cn

    name_en = _extract_latin_name(text)
    if name_en:
        fields["name_en"] = name_en

    generation_number = _extract_number_after_keywords(text, ["代数", "第"])
    if generation_number is not None:
        fields["generation_number"] = generation_number

    days_since_last_subculture = _extract_number_after_keywords(
        text,
        ["传代天数", "距传代", "距离上次传代", "天数"],
    )
    if days_since_last_subculture is not None:
        fields["days_since_last_subculture"] = days_since_last_subculture

    explicit_strain_id = extract_any_strain_id(text)
    if explicit_strain_id and explicit_strain_id != new_strain_id:
        fields["strain_id"] = explicit_strain_id

    return fields


def validate_required_fields(operation: Optional[str], fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if operation == "add":
        missing = [field for field in ["strain_id", "name_cn", "name_en"] if not fields.get(field)]
        return not missing, missing

    if operation == "update":
        missing = []
        if not fields.get("strain_id"):
            missing.append("strain_id")
        if not any(fields.get(field) is not None for field in UPDATE_FIELD_NAMES):
            missing.append("update_fields")
        return not missing, missing

    if operation == "delete":
        missing = [] if fields.get("strain_id") else ["strain_id"]
        return not missing, missing

    return False, ["operation"]


def extract_write_operation(message: str) -> Optional[str]:
    text = normalize_text_value(message)
    if _contains_any(text, ["添加", "增加", "新增"]):
        return "add"
    if _contains_any(text, ["删除", "移除"]):
        return "delete"
    if _contains_any(text, ["修改", "更新", "改为", "改成", "设为", "设置"]):
        return "update"
    return None


def find_strain_candidates(message: str, context_snapshot: Any) -> List[Dict[str, Any]]:
    text = normalize_text_value(message)
    text_lower = text.lower()
    matches = []
    for strain in getattr(context_snapshot, "strains", []) or []:
        strain_id = strain.get("strain_id") or ""
        name_cn = strain.get("name_cn") or ""
        name_en = (strain.get("name_en") or "").lower()
        if strain_id and strain_id in text:
            matches.append(strain)
        elif name_cn and name_cn in text:
            matches.append(strain)
        elif name_en and name_en in text_lower:
            matches.append(strain)
    return matches


def single_candidate_id(candidates: List[Dict[str, Any]]) -> Optional[str]:
    if len(candidates) == 1:
        return candidates[0].get("strain_id")
    return None


def extract_any_strain_id(message: str) -> Optional[str]:
    match = re.search(
        r"\b([A-Za-z][A-Za-z0-9_\-]*_[A-Za-z0-9_\-]+)\b",
        normalize_text_value(message),
    )
    return match.group(1) if match else None


def extract_new_strain_id(message: str) -> Optional[str]:
    text = normalize_text_value(message)
    patterns = [
        r"(?:品系|strain_id|编号)\s*(?:改为|改成|设为|设置为|更新为|为|是|:|：)?\s*([A-Za-z][A-Za-z0-9_\-]*_[A-Za-z0-9_\-]+)",
        r"(?:改为|改成|设为|设置为|更新为)\s*([A-Za-z][A-Za-z0-9_\-]*_[A-Za-z0-9_\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def extract_target_name(message: str) -> Optional[str]:
    text = normalize_text_value(message)
    for prefix in ["添加", "增加", "新增", "删除", "移除", "修改", "更新"]:
        if prefix in text:
            tail = text.split(prefix, 1)[1].strip()
            tail = re.split(r"[，,。；;：:\s]", tail, maxsplit=1)[0]
            if tail and not re.search(r"[A-Za-z0-9_]", tail):
                return tail
    return None


def _parse_pipe_add_format(text: str) -> Optional[Dict[str, str]]:
    patterns = [
        r"([A-Za-z][A-Za-z0-9_\-]+)\s*\|\s*([\u4e00-\u9fffA-Za-z0-9_\-]+)\s*\(([^()]+)\)",
        r"([A-Za-z][A-Za-z0-9_\-]+)\s*[，,]\s*([\u4e00-\u9fffA-Za-z0-9_\-]+)\s*[，,]\s*([^，,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return {
                "strain_id": match.group(1).strip(),
                "name_cn": match.group(2).strip(),
                "name_en": match.group(3).strip(),
            }
    return None


def _extract_latin_name(text: str) -> Optional[str]:
    patterns = [
        r"(?:英文名|拉丁名|学名|name_en)\s*(?:为|是|:|：)?\s*([A-Za-z][A-Za-z\s.\-_]+)",
        r"\(([A-Za-z][A-Za-z\s.\-_]+)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def _extract_bare_latin_name(text: str) -> Optional[str]:
    normalized = normalize_text_value(text)
    if re.fullmatch(r"[A-Z][A-Za-z.-]+(?:\s+[a-z][A-Za-z.-]+)+", normalized):
        return normalized
    return None


def _extract_value_after_keywords(text: str, keywords: List[str]) -> Optional[str]:
    for keyword in keywords:
        match = re.search(rf"{re.escape(keyword)}\s*(?:为|是|:|：)?\s*([\u4e00-\u9fffA-Za-z0-9_\-]+)", text)
        if match:
            return match.group(1).strip()
    return None


def _extract_number_after_keywords(text: str, keywords: List[str]) -> Optional[int]:
    for keyword in keywords:
        match = re.search(rf"{re.escape(keyword)}\D*(\d+)", text)
        if match:
            return int(match.group(1))
    return None


def _contains_any(text: str, keywords: List[str]) -> bool:
    normalized_text = normalize_text_value(text).casefold()
    return any(
        normalize_text_value(keyword).casefold() in normalized_text
        for keyword in keywords
    )
