from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from app.services.intent.input_normalizer import normalize_text_value
from app.services.intent.routing_models import (
    ChatDecision,
    ClarificationDecision,
    CompositeDecision,
    EffectKind,
    EmailDecision,
    EntityRef,
    KnowledgeDecision,
    PendingFormDecision,
    QueryDecision,
    ReasonCode,
    RiskLevel,
    RouteCandidate,
    RouteKind,
    RoutingDecision,
    RoutingInput,
    SpeechAct,
    ToolInfoDecision,
    WorkflowDecision,
    WorkflowAuditDecision,
    ScientificTaskDecision,
    WriteDecision,
)
from app.services.intent.write_action_parser import (
    extract_any_strain_id,
    find_strain_candidates,
    parse_pending_field_updates,
    parse_write_action,
    single_candidate_id,
)
from app.services.intent.llm_candidate_provider import collect_llm_route_candidates, current_router_diagnostics


EXPLANATION_KEYWORDS = [
    "什么是", "为什么", "如何理解", "一般多久", "多久一次",
    "原理", "区别", "介绍", "解释", "能不能讲讲",
]
QUERY_WORDS = ["哪些", "有哪些", "当前", "查看", "查询", "列出"]
LAB_STATUS_OBJECTS = [
    "实验室", "数据库", "当前数据库", "当前实验室", "我们实验室",
    "品系", "藻种", "品系列表", "藻种列表", "第几代", "状态",
    "距上次", "多久没传", "需要传代", "临近传代",
]
STRONG_LAB_STATUS_PHRASES = [
    "品系列表", "藻种列表", "当前数据库", "当前实验室", "我们实验室",
    "第几代", "距上次", "多久没传", "需要传代", "临近传代",
]
DOMAIN_KNOWLEDGE_OBJECTS = [
    "科学界", "生物固碳", "固碳", "微藻", "藻类", "小球藻",
    "培养", "传代", "光合作用", "原理", "机制",
]
DOMAIN_RAG_KEYWORDS = [
    "TAP", "培养基", "配方", "SOP", "实验手册", "手册", "manual",
    "论文", "paper", "文献", "protocol", "machine learning",
    "microalgae cultivation", "OD750", "pH", "biomass", "growth curve",
    "生长曲线", "实验数据", "表格", "字段", "环境变量", "设备说明",
]
FACT_LAYER_QUERY_KEYWORDS = [
    "当前状态", "当前藻种状态", "当前代数", "代数", "generation_number",
    "pending", "提醒状态", "reminder_cycle",
]
PENDING_QUERY_KEYWORDS = ["待确认", "审批", "审核", "pending", "待处理"]
WRITE_ACTION_WORDS = [
    "添加", "增加", "新增", "修改", "更新", "改为", "改成",
    "设为", "设置", "删除", "移除",
]
WRITE_OBJECTS = ["品系", "藻种", "代数", "天数", "记录", "状态"]
WRITE_TARGET_ALIASES = [
    "螺旋藻", "衣藻", "小球藻", "莱茵衣藻", "雨生红球藻", "栅藻",
    "微拟球藻", "Spirulina", "Chlorella", "Chlamydomonas",
    "Haematococcus", "Scenedesmus", "Nannochloropsis",
]
WORKFLOW_EXECUTION_PHRASES = [
    "执行传代", "执行传代流程", "开始传代", "启动传代", "触发传代",
    "运行传代", "现在进行传代", "确认执行传代", "立刻执行传代",
    "强制执行传代",
]
WORKFLOW_EXECUTION_VERBS = [
    "执行", "开始", "启动", "触发", "运行", "进行", "确认执行",
    "立刻执行", "强制执行",
]
TOOL_INFO_KEYWORDS = [
    "有哪些工具", "有什么工具", "你有哪些工具", "你有什么工具",
    "工具列表", "工具清单", "可用工具", "能调用什么工具",
    "可以调用什么工具", "你能调用哪些工具", "当前有哪些工具",
]
TOOL_INFO_HINTS = ["哪些", "什么", "列表", "清单", "可用", "调用", "有什么", "有哪些"]
LIST_STRAINS_QUERY_HINTS = [
    "有哪些", "有哪几种", "列出", "品系列表", "藻种列表", "微藻品系",
    "所有品系", "全部品系",
]
DUE_SUBCULTURE_QUERY_HINTS = [
    "传代", "临近传代", "需要传代", "该传代", "多久没传",
    "距上次传代", "距离上次传代", "提醒",
]
EMAIL_ACTION_KEYWORDS = ["发邮件", "发送邮件", "邮件提醒", "邮箱提醒", "email", "mail"]
DIRECT_EFFECT_MODIFIERS = ["立即发送", "直接发送", "现在发送", "马上发送"]
CANCEL_WORDS = ["取消", "放弃", "不用了", "不做了", "停止"]
WORKFLOW_DIAGNOSTIC_PHRASES = [
    "为什么", "为何", "怎么没", "没有变化", "没变化", "没有执行",
    "是否执行成功", "执行成功了吗", "有没有创建", "没有收到请求",
    "请求在哪", "执行记录", "审计记录",
]
WORKFLOW_CONFIRM_PHRASES = ["确认执行", "确认传代", "批准执行"]

_QUERY_TYPE_KEYWORDS = {
    "generation_number": ("generation_number", "当前代数", "代数", "第几代"),
    "pending": ("pending", "待确认", "待处理", "待审批", "审批", "审核"),
    "reminder_status": ("reminder_cycle", "提醒状态", "提醒周期", "overdue", "silenced"),
    "strain_status": ("当前状态", "当前藻种状态", "品系状态", "藻种状态"),
    "due_subculture": ("需要传代", "该传代", "多久没传代", "距上次传代", "临近传代"),
    "list_strains": ("品系列表", "藻种列表", "所有品系", "全部品系"),
}
_COMPOUND_SEPARATOR = re.compile(
    r"(?:然后|并且|同时|以及|随后|接着)|"
    r"并(?=(?:执行|开始|启动|触发|运行|查询|查看|列出|删除|新增|增加|修改|更新|发送|发邮件))|"
    r"\b(?:and|then)\b|[;.]",
    re.IGNORECASE,
)
_RAW_COMPOUND_SEPARATOR = re.compile(
    r"(?:然后|并且|同时|以及|随后|接着)|"
    r"并(?=(?:执行|开始|启动|触发|运行|查询|查看|列出|删除|新增|增加|修改|更新|发送|发邮件))|"
    r"\b(?:and|then)\b|[;；.。]",
    re.IGNORECASE,
)
_RISK_RANK = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
}

# Normal UTF-8 user inputs are kept alongside the older compatibility keyword
# set above. This keeps the router stable for repeatable project demos while
# preserving existing regression cases that use legacy mojibake strings.
EXPLANATION_KEYWORDS.extend([
    "什么是", "为什么", "如何理解", "一般多久", "多久一次", "原理", "区别", "介绍", "解释",
])
QUERY_WORDS.extend(["哪些", "有哪些", "当前", "查看", "查询", "列出", "list", "show"])
LAB_STATUS_OBJECTS.extend([
    "实验室", "数据库", "当前数据库", "当前实验室", "我们实验室", "品系", "藻种", "品系列表", "藻种列表",
    "第几代", "状态", "距离上次", "多久没传", "需要传代", "临近传代", "strain", "strains",
])
STRONG_LAB_STATUS_PHRASES.extend([
    "品系列表", "藻种列表", "当前数据库", "当前实验室", "我们实验室", "第几代", "距离上次",
    "多久没传", "需要传代", "临近传代", "list strains",
])
DOMAIN_RAG_KEYWORDS.extend([
    "培养基", "配方", "实验手册", "手册", "论文", "文献", "生长曲线", "实验数据", "表格",
    "字段", "环境变量", "设备说明",
])
FACT_LAYER_QUERY_KEYWORDS.extend(["当前状态", "当前藻种状态", "当前代数", "代数", "提醒状态"])
PENDING_QUERY_KEYWORDS.extend(["待确认", "审批", "审核", "待处理"])
WRITE_ACTION_WORDS.extend([
    "添加", "增加", "新增", "修改", "更新", "改为", "改成", "设为", "设置", "删除", "移除",
    "add", "create", "update", "rename", "delete", "remove",
])
WRITE_OBJECTS.extend(["品系", "藻种", "代数", "天数", "记录", "状态", "strain"])
WORKFLOW_EXECUTION_PHRASES.extend([
    "执行传代", "执行传代流程", "开始传代", "启动传代", "触发传代", "运行传代",
    "现在进行传代", "确认执行传代", "立即执行传代", "强制执行传代",
])
WORKFLOW_EXECUTION_VERBS.extend(["执行", "开始", "启动", "触发", "运行", "进行", "确认执行", "立即执行", "强制执行"])
TOOL_INFO_KEYWORDS.extend([
    "有哪些工具", "有什么工具", "你有哪些工具", "你有什么工具", "工具列表", "工具清单",
    "可用工具", "能调用什么工具", "可以调用什么工具", "你能调用哪些工具", "当前有哪些工具",
])
TOOL_INFO_HINTS.extend(["哪些", "什么", "列表", "清单", "可用", "调用", "有什么", "有哪些"])
LIST_STRAINS_QUERY_HINTS.extend([
    "有哪些", "有哪几种", "列出", "品系列表", "藻种列表", "微藻品系", "所有品系", "全部品系", "list strains",
])
EMAIL_ACTION_KEYWORDS.extend(["发邮件", "发送邮件", "邮件提醒", "邮箱提醒"])
WORKFLOW_DIAGNOSTIC_PHRASES.extend([
    "为什么", "为何", "怎么没", "没有变化", "没变化", "没有执行", "是否执行成功", "执行成功了吗",
    "有没有创建", "没有收到请求", "请求在哪", "执行记录", "审计记录",
])
WORKFLOW_CONFIRM_PHRASES.extend(["确认执行", "确认传代", "批准执行"])
_QUERY_TYPE_KEYWORDS["generation_number"] = (
    *_QUERY_TYPE_KEYWORDS["generation_number"],
    "当前代数",
    "代数",
    "第几代",
)
_QUERY_TYPE_KEYWORDS["pending"] = (
    *_QUERY_TYPE_KEYWORDS["pending"],
    "待确认",
    "待处理",
    "待审批",
    "审批",
    "审核",
)
_QUERY_TYPE_KEYWORDS["reminder_status"] = (
    *_QUERY_TYPE_KEYWORDS["reminder_status"],
    "提醒状态",
    "提醒周期",
)
_QUERY_TYPE_KEYWORDS["strain_status"] = (
    *_QUERY_TYPE_KEYWORDS["strain_status"],
    "当前状态",
    "当前藻种状态",
    "品系状态",
    "藻种状态",
)
_QUERY_TYPE_KEYWORDS["due_subculture"] = (
    *_QUERY_TYPE_KEYWORDS["due_subculture"],
    "需要传代",
    "该传代",
    "多久没传代",
    "距离上次传代",
    "临近传代",
)
_QUERY_TYPE_KEYWORDS["list_strains"] = (
    *_QUERY_TYPE_KEYWORDS["list_strains"],
    "品系列表",
    "藻种列表",
    "所有品系",
    "全部品系",
    "list strains",
)


def _contains_any(text: str, keywords: list[str] | tuple[str, ...]) -> bool:
    folded = normalize_text_value(text).casefold()
    return any(normalize_text_value(keyword).casefold() in folded for keyword in keywords)


EXPLANATION_KEYWORDS.extend([
    "什么是", "是什么", "为什么", "如何理解", "一般多久", "多久一次", "多久", "怎么",
    "原理", "区别", "介绍", "解释", "能不能讲讲",
])
QUERY_WORDS.extend(["哪些", "有哪些", "有哪几种", "当前", "现在", "查看", "查询", "列出", "多少", "怎么样", "总结"])
LAB_STATUS_OBJECTS.extend([
    "实验室", "数据库", "当前数据库", "当前实验室", "我们实验室", "品系", "藻种", "藻株",
    "品系列表", "藻种列表", "第几代", "代数", "状态", "距离上次", "多久没传",
    "需要传代", "临近传代", "快到期", "到期",
])
STRONG_LAB_STATUS_PHRASES.extend([
    "品系列表", "藻种列表", "当前数据库", "当前实验室", "我们实验室", "第几代", "代数",
    "距离上次", "多久没传", "需要传代", "临近传代", "快到期", "到期",
])
DOMAIN_KNOWLEDGE_OBJECTS.extend(["科学界", "生物固碳", "固碳", "微藻", "藻类", "小球藻", "培养", "传代", "光合作用", "原理", "机制", "周期"])
DOMAIN_RAG_KEYWORDS.extend([
    "培养基", "配方", "实验手册", "手册", "论文", "文献", "生长曲线", "实验数据",
    "表格", "字段", "环境变量", "设备说明", "用量",
])
FACT_LAYER_QUERY_KEYWORDS.extend(["当前状态", "当前藻种状态", "当前代数", "代数", "提醒状态"])
PENDING_QUERY_KEYWORDS.extend(["待确认", "待处理", "待审批", "审批", "审核"])
WRITE_ACTION_WORDS.extend(["添加", "增加", "新增", "修改", "更新", "改为", "改成", "设为", "设置", "删除", "移除"])
WRITE_OBJECTS.extend(["品系", "藻种", "藻株", "代数", "天数", "记录", "状态"])
WORKFLOW_EXECUTION_PHRASES.extend([
    "执行传代", "执行传代流程", "开始传代", "启动传代", "触发传代", "运行传代",
    "现在进行传代", "确认执行传代", "立即执行传代", "强制执行传代", "过一代",
])
WORKFLOW_EXECUTION_VERBS.extend(["执行", "开始", "启动", "触发", "运行", "进行", "确认执行", "立即执行", "强制执行"])
TOOL_INFO_KEYWORDS.extend([
    "有哪些工具", "有什么工具", "你有哪些工具", "你有什么工具", "工具列表", "工具清单",
    "可用工具", "能调用什么工具", "可以调用什么工具", "你能调用哪些工具", "当前有哪些工具",
])
TOOL_INFO_HINTS.extend(["哪些", "什么", "列表", "清单", "可用", "调用", "有什么", "有哪些"])
LIST_STRAINS_QUERY_HINTS.extend(["有哪些", "有哪几种", "列出", "品系列表", "藻种列表", "所有品系", "全部品系"])
DUE_SUBCULTURE_QUERY_HINTS.extend(["传代", "临近传代", "需要传代", "该传代", "多久没传", "距离上次传代", "快到期", "到期"])
EMAIL_ACTION_KEYWORDS.extend(["发邮件", "发送邮件", "邮件提醒", "邮箱提醒"])
WORKFLOW_DIAGNOSTIC_PHRASES.extend([
    "为什么", "为何", "怎么没", "没有变化", "没变化", "没有执行", "是否执行成功",
    "执行成功了吗", "有没有创建", "没有收到请求", "请求在哪", "执行记录", "审计记录",
])
WORKFLOW_CONFIRM_PHRASES.extend(["确认执行", "确认传代", "批准执行"])
_QUERY_TYPE_KEYWORDS["generation_number"] = (*_QUERY_TYPE_KEYWORDS["generation_number"], "当前代数", "代数", "第几代")
_QUERY_TYPE_KEYWORDS["pending"] = (*_QUERY_TYPE_KEYWORDS["pending"], "待确认", "待处理", "待审批", "审批", "审核")
_QUERY_TYPE_KEYWORDS["reminder_status"] = (*_QUERY_TYPE_KEYWORDS["reminder_status"], "提醒状态", "提醒周期")
_QUERY_TYPE_KEYWORDS["strain_status"] = (*_QUERY_TYPE_KEYWORDS["strain_status"], "当前状态", "当前藻种状态", "品系状态", "藻种状态", "状态", "怎么样", "总结")
_QUERY_TYPE_KEYWORDS["due_subculture"] = (*_QUERY_TYPE_KEYWORDS["due_subculture"], "需要传代", "该传代", "多久没传代", "距离上次传代", "临近传代", "快到期", "到期")
_QUERY_TYPE_KEYWORDS["list_strains"] = (*_QUERY_TYPE_KEYWORDS["list_strains"], "品系列表", "藻种列表", "所有品系", "全部品系")


def _has_tool_info(text: str) -> bool:
    return _contains_any(text, TOOL_INFO_KEYWORDS) or (
        "工具" in text and _contains_any(text, TOOL_INFO_HINTS)
    )


def _has_workflow(text: str) -> bool:
    return _contains_any(text, WORKFLOW_EXECUTION_PHRASES) or (
        "传代" in text and _contains_any(text, WORKFLOW_EXECUTION_VERBS)
    )


def _is_workflow_diagnostic(text: str) -> bool:
    has_audit_cue = _contains_any(text, WORKFLOW_DIAGNOSTIC_PHRASES)
    has_workflow_context = _contains_any(
        text,
        ["传代", "workflow", "pending", "请求", "执行", "数据库"],
    )
    return has_audit_cue and has_workflow_context


def _is_workflow_confirmation(text: str) -> bool:
    return _contains_any(text, WORKFLOW_CONFIRM_PHRASES) or bool(
        re.search(r"确认\s*pending\s*#?\d+", text, re.I)
    )


def _extract_pending_id(text: str) -> int | None:
    match = re.search(r"pending\s*#?(\d+)", text, re.I)
    return int(match.group(1)) if match else None


def _has_pending_query(text: str) -> bool:
    return _contains_any(text, PENDING_QUERY_KEYWORDS)


def _has_write(text: str, strain_id: str | None, candidates: list[dict[str, Any]]) -> bool:
    if not _contains_any(text, WRITE_ACTION_WORDS):
        return False
    if _contains_any(text, WRITE_OBJECTS) or _contains_any(text, WRITE_TARGET_ALIASES):
        return True
    return bool(strain_id or single_candidate_id(candidates))


INTERNAL_LAB_SCOPE_KEYWORDS = (
    "当前数据库",
    "数据库",
    "资源中心",
    "当前资源",
    "实验室",
    "当前实验室",
    "我们实验室",
    "本实验室",
    "品系列表",
    "藻种列表",
    "所有品系",
    "全部品系",
    "list strains",
    "database",
    "resource center",
    "lab strains",
)

EXTERNAL_ECOLOGY_KEYWORDS = (
    "野外",
    "自然",
    "自然界",
    "分布",
    "野生",
    "原生",
    "户外",
    "环境中",
    "水体",
    "河流",
    "湖泊",
    "湿地",
    "池塘",
    "溪流",
    "近海",
    "海域",
    "城市",
    "地区",
    "区域",
    "沈阳",
    "浑河",
    "辽河",
    "太湖",
    "滇池",
    "常见藻种",
    "常见藻类",
    "常见微藻",
    "wild",
    "natural",
    "distribution",
    "distributed",
    "field",
    "freshwater",
    "river",
    "lake",
    "wetland",
    "pond",
    "Shenyang",
    "Hun River",
)


def _has_internal_lab_scope(text: str) -> bool:
    return _contains_any(text, INTERNAL_LAB_SCOPE_KEYWORDS)


def _has_external_ecology_scope(text: str) -> bool:
    return _contains_any(text, EXTERNAL_ECOLOGY_KEYWORDS)


def _has_lab_query(text: str) -> bool:
    if _has_external_ecology_scope(text) and not _has_internal_lab_scope(text):
        return False
    if _contains_any(text, FACT_LAYER_QUERY_KEYWORDS):
        return True
    return (
        _contains_any(text, QUERY_WORDS) and _contains_any(text, LAB_STATUS_OBJECTS)
    ) or _contains_any(text, STRONG_LAB_STATUS_PHRASES)


def _has_knowledge_query(text: str) -> bool:
    if _has_external_ecology_scope(text) and not _has_internal_lab_scope(text) and (
        _contains_any(text, QUERY_WORDS)
        or _contains_any(text, ("哪些", "有哪些", "有哪几种", "常见", "分布", "介绍", "what", "which", "list"))
    ):
        return True
    if _has_lab_query(text):
        return False
    # An explanatory verb is not a domain intent by itself (for example,
    # "介绍一下你自己"). It must bind to a knowledge-domain anchor.
    if _contains_any(text, DOMAIN_RAG_KEYWORDS):
        return True
    has_domain_anchor = _contains_any(text, DOMAIN_KNOWLEDGE_OBJECTS)
    return has_domain_anchor and (
        _contains_any(text, EXPLANATION_KEYWORDS)
        or _contains_any(text, QUERY_WORDS)
    )


def _has_entity_fact_query(text: str, entities: tuple[EntityRef, ...]) -> bool:
    if not any(item.exists and item.canonical_id for item in entities):
        return False
    return _contains_any(
        text,
        (
            "当前", "现在", "状态", "怎么样", "总结", "代数", "第几代", "多少",
            "快到期", "到期", "需要传代", "该传代", "多久没传", "提醒状态",
            "pending", "待确认", "待处理", "待审批", "审批",
        ),
    )


def _has_email(text: str) -> bool:
    return _contains_any(text, EMAIL_ACTION_KEYWORDS)


def _is_cancel(text: str) -> bool:
    return _contains_any(text, CANCEL_WORDS)


def _classify_lab_query(text: str) -> str:
    if _contains_any(text, _QUERY_TYPE_KEYWORDS["generation_number"]):
        return "generation_number"
    if _contains_any(text, _QUERY_TYPE_KEYWORDS["pending"]):
        return "pending"
    if _contains_any(text, _QUERY_TYPE_KEYWORDS["reminder_status"]):
        return "reminder_status"
    if _contains_any(text, _QUERY_TYPE_KEYWORDS["strain_status"] + ("状态",)):
        return "strain_status"
    if _contains_any(text, DUE_SUBCULTURE_QUERY_HINTS):
        return "due_subculture"
    if _contains_any(text, LIST_STRAINS_QUERY_HINTS):
        return "list_strains"
    return "list_strains"


def _detect_explicit_query_types(text: str) -> tuple[str, ...]:
    detected = tuple(
        query_type
        for query_type, keywords in _QUERY_TYPE_KEYWORDS.items()
        if _contains_any(text, keywords)
    )
    if "reminder_status" in detected:
        return tuple(item for item in detected if item != "strain_status")
    return detected


def _resolve_entities_for_text(text: str, context_snapshot: Any) -> tuple[EntityRef, ...]:
    strains = getattr(context_snapshot, "strains", []) or []
    match_text = normalize_text_value(text).casefold()
    refs: list[EntityRef] = []
    known_ids = {
        str(item.get("strain_id") or "").casefold(): item
        for item in strains
        if item.get("strain_id")
    }
    # Resolve authoritative database IDs first. Direct substring matching is
    # deliberate: Chinese characters adjacent to an ASCII ID do not form a
    # Python Unicode word boundary.
    known_matches: list[tuple[int, int, dict[str, Any]]] = []
    for folded_id, strain in known_ids.items():
        strain_id = str(strain.get("strain_id") or "")
        match = re.search(
            rf"(?<![A-Za-z0-9_-]){re.escape(strain_id)}(?![A-Za-z0-9_-])",
            text,
            re.I,
        ) if folded_id else None
        if match:
            known_matches.append((match.start(), -len(strain_id), strain))
    for _, _, strain in sorted(known_matches):
        strain_id = str(strain.get("strain_id") or "")
        refs.append(EntityRef("strain", strain_id, strain_id, "database_id", True))
    for mention in re.findall(
        r"(?<![A-Za-z0-9_-])([A-Za-z][A-Za-z0-9_-]*_[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)(?![A-Za-z0-9_-])",
        text,
    ):
        if any(ref.canonical_id and ref.canonical_id.casefold() == mention.casefold() for ref in refs):
            continue
        matched = known_ids.get(mention.casefold())
        refs.append(
            EntityRef(
                entity_type="strain",
                canonical_id=matched.get("strain_id") if matched else mention,
                mention=mention,
                source="database_id" if matched else "explicit_id",
                exists=bool(matched),
            )
        )
    for strain in strains:
        strain_id = str(strain.get("strain_id") or "")
        name_cn = str(strain.get("name_cn") or "")
        name_en = str(strain.get("name_en") or "")
        if strain_id and any(ref.canonical_id == strain_id for ref in refs):
            continue
        if name_cn and name_cn in text:
            refs.append(EntityRef("strain", strain_id, name_cn, "database_name", True))
        elif name_en and name_en.casefold() in match_text:
            refs.append(EntityRef("strain", strain_id, name_en, "latin_name", True))
    deduped: list[EntityRef] = []
    seen: set[tuple[str | None, str]] = set()
    for ref in refs:
        key = (ref.canonical_id, ref.mention.casefold())
        if key not in seen:
            seen.add(key)
            deduped.append(ref)
    return tuple(deduped)


def resolve_entities(routing_input: RoutingInput) -> tuple[EntityRef, ...]:
    return _resolve_entities_for_text(
        routing_input.message.normalized_text,
        routing_input.context_snapshot,
    )


def _single_target(entities: tuple[EntityRef, ...]) -> EntityRef | None:
    canonical = {item.canonical_id for item in entities if item.canonical_id}
    return entities[0] if len(canonical) == 1 else None


def _detect_fragment(text: str, context_snapshot: Any) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    entities = _resolve_entities_for_text(text, context_snapshot)
    target = _single_target(entities)
    strain_matches = find_strain_candidates(text, context_snapshot)
    explicit_strain_id = extract_any_strain_id(text)

    scientific_terms = (
        "生长异常", "增长异常", "生长变慢", "增长变慢", "异常诊断", "实验优化",
        "优化下一轮", "下一轮实验", "growth anomaly", "diagnose growth", "experiment optimization",
    )
    if any(term in text.casefold() for term in scientific_terms):
        import re
        from app.core.db.scientific import list_datasets
        match = re.search(r"\bds_[0-9a-f]{8,64}\b", text.casefold())
        # Routing can run before application startup has initialized the
        # additive scientific schema (for example in CLI and unit tests).
        try:
            datasets = list_datasets()
        except Exception:
            datasets = []
        dataset_id = match.group(0) if match else (datasets[0]["id"] if len(datasets) == 1 else None)
        optimize = any(term in text.casefold() for term in ("优化", "下一轮", "optimization", "optimize"))
        return [RouteCandidate(
            RouteKind.SCIENTIFIC_TASK,
            ReasonCode.SCIENTIFIC_TASK_MATCHED,
            RiskLevel.MEDIUM if optimize else RiskLevel.LOW,
            text,
            source="rule",
            arguments={
                "dataset_id": dataset_id,
                "mode": "diagnose_and_optimize" if optimize else "diagnose",
                "offline_replay": "离线回放" in text or "offline replay" in text.casefold(),
            },
            missing_fields=() if dataset_id else ("dataset_id",),
        )]

    # Speech act outranks keyword co-occurrence. A retrospective diagnostic or
    # approval acknowledgement must never be reinterpreted as a new command.
    if _is_workflow_diagnostic(text):
        return [RouteCandidate(
            RouteKind.WORKFLOW_AUDIT,
            ReasonCode.WORKFLOW_AUDIT_QUERY,
            RiskLevel.NONE,
            text,
            speech_act=SpeechAct.DIAGNOSTIC,
            target=target,
            entity_options=entities,
            arguments={"pending_id": _extract_pending_id(text)},
        )]
    if _is_workflow_confirmation(text):
        return [RouteCandidate(
            RouteKind.WORKFLOW_AUDIT,
            ReasonCode.WORKFLOW_CONFIRMATION_REQUIRES_UI,
            RiskLevel.HIGH,
            text,
            speech_act=SpeechAct.CONFIRM,
            target=target,
            entity_options=entities,
            arguments={"pending_id": _extract_pending_id(text)},
        )]

    if _has_tool_info(text):
        candidates.append(RouteCandidate(
            RouteKind.TOOL_INFO, ReasonCode.TOOL_INFO_MATCHED,
            RiskLevel.NONE, text,
        ))
    if _has_workflow(text):
        candidates.append(RouteCandidate(
            RouteKind.WORKFLOW, ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
            RiskLevel.HIGH, text, target=target, entity_options=entities,
        ))
    entity_fact_query = (
        _has_entity_fact_query(text, entities)
        and not _has_workflow(text)
        and not _has_email(text)
    )
    lab_query = _has_lab_query(text) or entity_fact_query
    pending_query = _has_pending_query(text)
    if pending_query:
        candidates.append(RouteCandidate(
            RouteKind.PENDING_QUERY, ReasonCode.PENDING_QUERY_MATCHED,
            RiskLevel.LOW, text, target=target, entity_options=entities,
            query_type="pending",
        ))
    write_action = _has_write(text, explicit_strain_id, strain_matches)
    if write_action:
        plan = parse_write_action(text, context_snapshot, explicit_strain_id, strain_matches)
        plan_target_id = plan.get("strain_id")
        write_target = next(
            (item for item in entities if item.canonical_id == plan_target_id and item.exists),
            None,
        )
        if write_target is None and plan_target_id:
            write_target = EntityRef(
                "strain", plan_target_id, plan_target_id, "explicit_id", False,
            )
        candidates.append(RouteCandidate(
            RouteKind.WRITE_ACTION, ReasonCode.WRITE_ACTION_MATCHED,
            RiskLevel.MEDIUM, text, target=write_target, entity_options=entities,
            operation=plan.get("operation"), tool_name=plan.get("tool_name"),
            arguments=dict(plan.get("fields") or {}),
            missing_fields=tuple(plan.get("missing_fields") or ()),
            form_candidates=tuple(plan.get("candidates") or ()),
        ))
    if lab_query and not pending_query and not write_action:
        query_type = _classify_lab_query(text)
        candidates.append(RouteCandidate(
            RouteKind.LAB_QUERY, ReasonCode.LAB_QUERY_MATCHED,
            RiskLevel.LOW, text, target=target, entity_options=entities,
            query_type=query_type,
        ))
        explicit_types = _detect_explicit_query_types(text)
        if len(explicit_types) > 1:
            for item in explicit_types:
                candidates.append(RouteCandidate(
                    RouteKind.PENDING_QUERY if item == "pending" else RouteKind.LAB_QUERY,
                    ReasonCode.PENDING_QUERY_MATCHED if item == "pending" else ReasonCode.LAB_QUERY_MATCHED,
                    RiskLevel.LOW, text, target=target, entity_options=entities,
                    query_type=item,
                ))
    if _has_knowledge_query(text) and not entity_fact_query and not _has_workflow(text):
        candidates.append(RouteCandidate(
            RouteKind.KNOWLEDGE_QUERY, ReasonCode.KNOWLEDGE_QUERY_MATCHED,
            RiskLevel.LOW, text, target=target, entity_options=entities,
        ))
    if _has_email(text):
        candidates.append(RouteCandidate(
            RouteKind.EMAIL, ReasonCode.EMAIL_MATCHED,
            RiskLevel.MEDIUM, text, target=target, entity_options=entities,
        ))
    if (
        ("传代" in text or "subculture" in text.casefold())
        and not _has_workflow(text)
        and not _has_knowledge_query(text)
        and not lab_query
    ):
        candidates.append(RouteCandidate(
            RouteKind.CLARIFICATION, ReasonCode.AMBIGUOUS_SUBCULTURE,
            RiskLevel.HIGH, text, target=target, entity_options=entities,
        ))
    return candidates


def _llm_candidates_for_text(
    *,
    source_text: str,
    normalized_text: str,
    context_snapshot: Any,
    active_pending_form: dict[str, Any] | None = None,
    active_workflow_request: dict[str, Any] | None = None,
) -> list[RouteCandidate]:
    llm_items = collect_llm_route_candidates(
        original_text=source_text,
        normalized_text=normalized_text,
        context_snapshot=context_snapshot,
        active_pending_form=active_pending_form,
        active_workflow_request=active_workflow_request,
    )
    if not llm_items:
        return []
    entities = _resolve_entities_for_text(source_text, context_snapshot)
    if not entities and source_text != normalized_text:
        entities = _resolve_entities_for_text(normalized_text, context_snapshot)
    target = _single_target(entities)
    candidates: list[RouteCandidate] = []
    for item in llm_items:
        if item.confidence < 0.55:
            continue
        reason = {
            RouteKind.WORKFLOW: ReasonCode.WORKFLOW_EXPLICIT_EXECUTION,
            RouteKind.WORKFLOW_AUDIT: ReasonCode.WORKFLOW_AUDIT_QUERY,
            RouteKind.KNOWLEDGE_QUERY: ReasonCode.KNOWLEDGE_QUERY_MATCHED,
            RouteKind.LAB_QUERY: ReasonCode.LAB_QUERY_MATCHED,
            RouteKind.PENDING_QUERY: ReasonCode.PENDING_QUERY_MATCHED,
            RouteKind.WRITE_ACTION: ReasonCode.WRITE_ACTION_MATCHED,
            RouteKind.EMAIL: ReasonCode.EMAIL_MATCHED,
            RouteKind.CLARIFICATION: ReasonCode.AMBIGUOUS_SUBCULTURE,
        }.get(item.route_kind, ReasonCode.DEFAULT_CHAT)
        risk = _risk_for_route(item.route_kind)
        candidates.append(
            RouteCandidate(
                item.route_kind,
                reason,
                risk,
                source_text,
                source="llm",
                score=item.confidence,
                evidence_items=item.evidence or ("llm_semantic_candidate",),
                negative_signals=tuple(
                    signal
                    for signal, enabled in (
                        ("llm_target_unverified", item.route_kind == RouteKind.WORKFLOW and not target),
                        ("high_risk_deterministic_routing_required", risk in {RiskLevel.MEDIUM, RiskLevel.HIGH}),
                    )
                    if enabled
                ),
                speech_act=item.speech_act,
                target=target,
                entity_options=entities,
                missing_fields=item.missing_fields,
                query_type="pending" if item.route_kind == RouteKind.PENDING_QUERY else None,
            )
        )
    return candidates


def _risk_for_route(kind: RouteKind) -> RiskLevel:
    if kind in {RouteKind.WORKFLOW, RouteKind.SCIENTIFIC_TASK}:
        return RiskLevel.HIGH
    if kind in {RouteKind.WRITE_ACTION, RouteKind.EMAIL, RouteKind.PENDING_FORM}:
        return RiskLevel.MEDIUM
    if kind in {RouteKind.CHAT, RouteKind.TOOL_INFO}:
        return RiskLevel.NONE
    return RiskLevel.LOW


def detect_route_candidates(routing_input: RoutingInput) -> tuple[RouteCandidate, ...]:
    text = routing_input.message.normalized_text
    fragments = [
        item.strip(" ,") 
        for item in _COMPOUND_SEPARATOR.split(text) 
        if item.strip(" ,")
    ]
    original_fragments = [
        item.strip()
        for item in _RAW_COMPOUND_SEPARATOR.split(routing_input.message.original_text)
        if item.strip()
    ]

    """
        Once explicit discourse boundaries exist, each fragment becomes its own
    intent frame. Detecting the whole sentence as well would mix entities and
    arguments across otherwise independent steps.
    """
    if len(fragments) > 1:
        candidates: list[RouteCandidate] = []
        for index, fragment in enumerate(fragments):
            source_text = (
                original_fragments[index]
                if len(original_fragments) == len(fragments)
                else fragment
            )
            candidates.extend(
                replace(item, evidence=source_text)
                for item in _detect_fragment(fragment, routing_input.context_snapshot)
            )
            candidates.extend(
                _llm_candidates_for_text(
                    source_text=source_text,
                    normalized_text=fragment,
                    context_snapshot=routing_input.context_snapshot,
                    active_pending_form=routing_input.active_pending_form,
                    active_workflow_request=routing_input.active_workflow_request,
                )
            )
    else:
        candidates = [
            replace(item, evidence=routing_input.message.original_text)
            for item in _detect_fragment(text, routing_input.context_snapshot)
        ]
        candidates.extend(
            _llm_candidates_for_text(
                source_text=routing_input.message.original_text,
                normalized_text=text,
                context_snapshot=routing_input.context_snapshot,
                active_pending_form=routing_input.active_pending_form,
                active_workflow_request=routing_input.active_workflow_request,
            )
        )
    if not candidates:
        candidates.append(RouteCandidate(
            RouteKind.CHAT, ReasonCode.DEFAULT_CHAT, RiskLevel.NONE, text,
            target=_single_target(resolve_entities(routing_input)),
            entity_options=resolve_entities(routing_input),
        ))
    deduped: list[RouteCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            candidate.kind,
            candidate.operation,
            candidate.query_type,
            candidate.target.canonical_id if candidate.target else None,
            tuple(item.canonical_id for item in candidate.entity_options),
            normalize_text_value(candidate.evidence).casefold(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return tuple(deduped)


def _candidate_trace(candidate: RouteCandidate) -> dict[str, Any]:
    return {
        "route_kind": candidate.kind.value,
        "reason_code": candidate.reason_code.value,
        "risk_level": candidate.risk_level.value,
        "source": candidate.source,
        "score": candidate.score,
        "evidence_items": list(candidate.evidence_items),
        "negative_signals": list(candidate.negative_signals),
        "selection_reason": candidate.selection_reason,
        "target": candidate.target.canonical_id if candidate.target else None,
        "entity_options": [
            {
                "canonical_id": item.canonical_id,
                "mention": item.mention,
                "source": item.source,
                "exists": item.exists,
            }
            for item in candidate.entity_options
        ],
    }


def _selection_trace(
    candidates: tuple[RouteCandidate, ...],
    *,
    selected: RouteCandidate | None = None,
    reason: str,
) -> dict[str, Any]:
    return {
        "selector": "hybrid_deterministic_v1",
        "selection_reason": reason,
        "selected": _candidate_trace(selected) if selected else None,
        "candidates": [_candidate_trace(item) for item in candidates],
    }


def _clarification(
    code: ReasonCode,
    explanation: str,
    questions: tuple[str, ...],
    candidates: tuple[RouteCandidate, ...],
    risk: RiskLevel,
    target: EntityRef | None = None,
    response_action: str = "routing_clarification",
    selection_trace: dict[str, Any] | None = None,
) -> ClarificationDecision:
    return ClarificationDecision(
        reason_code=code, risk_level=risk, explanation=explanation,
        questions=questions, candidates=candidates, target=target,
        response_action=response_action,
        selection_trace=selection_trace or _selection_trace(candidates, reason=explanation),
    )


def _decision_from_candidate(
    candidate: RouteCandidate,
    all_candidates: tuple[RouteCandidate, ...] | None = None,
) -> RoutingDecision:
    decision_candidates = all_candidates or (candidate,)
    existing_targets = tuple(
        dict.fromkeys(
            item.canonical_id for item in candidate.entity_options
            if item.exists and item.canonical_id
        )
    )
    target_refs = tuple(
        next(item for item in candidate.entity_options if item.canonical_id == target_id)
        for target_id in existing_targets
    )
    common = {
        "reason_code": candidate.reason_code,
        "risk_level": candidate.risk_level,
        "source_text": candidate.evidence,
        "target": candidate.target,
        "candidates": decision_candidates,
        "selection_trace": _selection_trace(
            decision_candidates,
            selected=candidate,
            reason=candidate.selection_reason or f"selected_{candidate.kind.value}",
        ),
    }
    if candidate.kind == RouteKind.TOOL_INFO:
        return ToolInfoDecision(explanation="The user asked about available tool capabilities.", **common)
    if candidate.kind == RouteKind.EMAIL:
        return EmailDecision(explanation="The request contains an email action.", **common)
    if candidate.kind in {RouteKind.LAB_QUERY, RouteKind.PENDING_QUERY}:
        return QueryDecision(
            kind=candidate.kind, query_type=candidate.query_type or "list_strains",
            targets=target_refs,
            explanation="The request asks for current database-backed facts.", **common,
        )
    if candidate.kind == RouteKind.KNOWLEDGE_QUERY:
        return KnowledgeDecision(explanation="The request asks for domain knowledge backed by RAG.", **common)
    if candidate.kind == RouteKind.SCIENTIFIC_TASK:
        if candidate.missing_fields:
            return ScientificTaskDecision(
                explanation="A scientific task requires an imported dataset or an explicit dataset_id.",
                arguments=candidate.arguments,
                missing_fields=candidate.missing_fields,
                **common,
            )
        return ScientificTaskDecision(
            explanation="The request starts the governed microalgae diagnosis and experiment-optimization loop.",
            arguments=candidate.arguments,
            **common,
        )
    if candidate.kind == RouteKind.WORKFLOW_AUDIT:
        return WorkflowAuditDecision(
            explanation="The user asks about workflow request or execution evidence.",
            speech_act=candidate.speech_act or SpeechAct.DIAGNOSTIC,
            pending_id=candidate.arguments.get("pending_id"),
            **common,
        )
    if candidate.kind == RouteKind.WRITE_ACTION:
        if candidate.missing_fields:
            return PendingFormDecision(
                reason_code=ReasonCode.REQUIRED_FIELDS_MISSING,
                risk_level=RiskLevel.MEDIUM,
                explanation="The write request needs more structured fields.",
                target=candidate.target, candidates=(candidate,), form_action="start",
                operation=candidate.operation, tool_name=candidate.tool_name,
                arguments=candidate.arguments, missing_fields=candidate.missing_fields,
                form_candidates=candidate.form_candidates,
            )
        return WriteDecision(
            explanation="The request proposes a database write through approval.",
            operation=candidate.operation or "", tool_name=candidate.tool_name or "",
            arguments=candidate.arguments, missing_fields=candidate.missing_fields, **common,
        )
    if candidate.kind == RouteKind.WORKFLOW:
        if not candidate.target or not candidate.target.canonical_id:
            return _clarification(
                ReasonCode.WORKFLOW_TARGET_REQUIRED,
                "A high-risk workflow request requires an explicit, verified strain_id.",
                ("请提供要执行传代 workflow 的唯一 strain_id。",),
                (candidate,),
                RiskLevel.HIGH,
                candidate.target,
                "workflow_request_clarification",
            )
        return WorkflowDecision(
            explanation="The user explicitly requested a high-risk workflow.",
            speech_act=candidate.speech_act or SpeechAct.COMMAND,
            **common,
        )
    if candidate.kind == RouteKind.CLARIFICATION:
        return _clarification(
            ReasonCode.AMBIGUOUS_SUBCULTURE,
            "Subculture was mentioned without a clear query or execution request.",
            ("你是想了解传代知识，还是执行某个明确品系的传代流程？",),
            (candidate,), RiskLevel.HIGH, candidate.target, "ambiguous_intent",
        )
    return ChatDecision(explanation="No deterministic business route matched.", **common)


def _candidate_effect(candidate: RouteCandidate) -> EffectKind:
    if candidate.kind == RouteKind.EMAIL:
        return (
            EffectKind.EXECUTE
            if _contains_any(candidate.evidence, DIRECT_EFFECT_MODIFIERS)
            else EffectKind.DRAFT
        )
    if candidate.kind in {RouteKind.WRITE_ACTION, RouteKind.WORKFLOW}:
        # Both handlers create approval requests; neither performs the final
        # database or hardware mutation during routing dispatch.
        return EffectKind.PROPOSE
    return EffectKind.READ


def _candidate_target_error(
    candidate: RouteCandidate,
    all_candidates: tuple[RouteCandidate, ...],
) -> ClarificationDecision | None:
    unknown = next(
        (
            item for item in candidate.entity_options
            if item.source == "explicit_id" and not item.exists
            and item.canonical_id != candidate.arguments.get("new_strain_id")
        ),
        None,
    )
    requires_existing = candidate.kind in {
        RouteKind.LAB_QUERY, RouteKind.PENDING_QUERY, RouteKind.WORKFLOW,
    } or (
        candidate.kind == RouteKind.WRITE_ACTION
        and candidate.operation in {"update", "delete"}
    )
    if unknown and requires_existing:
        return _clarification(
            ReasonCode.TARGET_NOT_FOUND,
            f"The explicit strain target does not exist: {unknown.mention}",
            (f"数据库中未找到 {unknown.mention}，请提供有效的 strain_id。",),
            all_candidates, candidate.risk_level, unknown,
        )

    existing_ids = {
        item.canonical_id for item in candidate.entity_options
        if item.exists and item.canonical_id
    }
    multi_target_capable = candidate.kind in {
        RouteKind.LAB_QUERY, RouteKind.PENDING_QUERY, RouteKind.KNOWLEDGE_QUERY,
    }
    write_missing_target = (
        candidate.kind == RouteKind.WRITE_ACTION
        and "strain_id" in candidate.missing_fields
    )
    if len(existing_ids) > 1 and not multi_target_capable and not write_missing_target:
        return _clarification(
            ReasonCode.MULTIPLE_TARGETS_CONFLICT,
            "This route requires a single target but multiple strains were resolved.",
            ("该操作一次只能处理一个品系，请明确唯一的 strain_id。",),
            all_candidates, candidate.risk_level,
        )
    return None


def _max_risk(candidates: tuple[RouteCandidate, ...]) -> RiskLevel:
    return max((item.risk_level for item in candidates), key=_RISK_RANK.get)


def _active_form_decision(
    routing_input: RoutingInput,
    candidates: tuple[RouteCandidate, ...],
) -> RoutingDecision | None:
    if not routing_input.active_pending_form:
        return None
    if _is_cancel(routing_input.message.normalized_text):
        return PendingFormDecision(
            reason_code=ReasonCode.PENDING_FORM_CANCEL,
            risk_level=RiskLevel.MEDIUM,
            explanation="The user cancelled the active pending form.",
            source_text=routing_input.message.normalized_text,
            candidates=candidates, form_action="cancel",
        )

    proposals = [item for item in candidates if _candidate_effect(item) == EffectKind.PROPOSE]
    if proposals:
        selected = proposals[0]
        if selected.kind == RouteKind.WRITE_ACTION:
            return PendingFormDecision(
                reason_code=ReasonCode.PENDING_FORM_CONFLICT,
                risk_level=RiskLevel.MEDIUM,
                explanation="A new write request conflicts with an active pending form.",
                source_text=selected.evidence,
                target=selected.target, candidates=candidates, form_action="conflict",
                operation=selected.operation, tool_name=selected.tool_name,
                arguments=selected.arguments, missing_fields=selected.missing_fields,
                form_candidates=selected.form_candidates,
            )
        return _clarification(
            ReasonCode.MULTIPLE_INTENTS_CONFLICT,
            "A workflow request conflicts with an active pending form.",
            ("请先补全或取消当前操作，再发起传代流程。",),
            candidates, RiskLevel.HIGH, selected.target,
        )

    # Read and draft frames are independent of the unfinished form. A plain
    # chat frame continues slot filling only when it contributes a value that
    # matches the form schema; unrelated conversation leaves the form intact.
    if len(candidates) == 1 and candidates[0].kind == RouteKind.CHAT:
        state = routing_input.active_pending_form or {}
        updates, resolved_candidates = parse_pending_field_updates(
            state.get("operation"),
            routing_input.message.original_text,
            routing_input.context_snapshot,
            state.get("missing_fields") or [],
        )
        collected = state.get("collected_fields") or {}
        advances_form = any(
            value is not None and collected.get(key) != value
            for key, value in updates.items()
        ) or bool(resolved_candidates)
        if advances_form:
            return PendingFormDecision(
                reason_code=ReasonCode.PENDING_FORM_CONTINUATION,
                risk_level=RiskLevel.MEDIUM,
                explanation="The message supplies schema-valid pending form fields.",
                source_text=routing_input.message.original_text,
                candidates=candidates, form_action="continue",
            )
    return None


def _active_workflow_decision(
    routing_input: RoutingInput,
    candidates: tuple[RouteCandidate, ...],
) -> RoutingDecision | None:
    if not routing_input.active_workflow_request:
        return None
    if _is_cancel(routing_input.message.normalized_text):
        return WorkflowDecision(
            reason_code=ReasonCode.WORKFLOW_REQUEST_CANCEL,
            risk_level=RiskLevel.HIGH,
            explanation="The user cancelled the pre-approval workflow request.",
            source_text=routing_input.message.original_text,
            candidates=candidates,
            speech_act=SpeechAct.CANCEL,
        )
    if len(candidates) == 1 and candidates[0].kind == RouteKind.CHAT:
        candidate = candidates[0]
        existing_targets = tuple(
            item for item in candidate.entity_options if item.exists and item.canonical_id
        )
        if len({item.canonical_id for item in existing_targets}) == 1:
            target = existing_targets[0]
            return WorkflowDecision(
                reason_code=ReasonCode.WORKFLOW_SLOT_CONTINUATION,
                risk_level=RiskLevel.HIGH,
                explanation="The message supplies the missing workflow strain target.",
                source_text=routing_input.message.original_text,
                target=target,
                candidates=candidates,
                speech_act=SpeechAct.SLOT_VALUE,
            )
    return None


def _compose_or_clarify(candidates: tuple[RouteCandidate, ...]) -> RoutingDecision:
    if len(candidates) == 1:
        return _decision_from_candidate(replace(candidates[0], selection_reason="only_candidate"))

    audits = [item for item in candidates if item.kind == RouteKind.WORKFLOW_AUDIT]
    if audits:
        selected = replace(
            sorted(audits, key=lambda item: item.score, reverse=True)[0],
            selection_reason="audit_or_confirmation_outranks_execution",
        )
        return _decision_from_candidate(selected, candidates)

    ambiguous = [item for item in candidates if item.kind == RouteKind.CLARIFICATION]
    proposals = [item for item in candidates if _candidate_effect(item) == EffectKind.PROPOSE]
    executions = [item for item in candidates if _candidate_effect(item) == EffectKind.EXECUTE]
    if ambiguous or len(proposals) > 1 or executions:
        labels = ", ".join(item.kind.value for item in candidates)
        return _clarification(
            ReasonCode.MULTIPLE_INTENTS_CONFLICT,
            f"The intent frames cannot form a safe plan: {labels}",
            ("检测到相互冲突或多个状态变更操作，请明确本次要执行的一个操作。",),
            candidates, _max_risk(candidates),
        )

    steps = tuple(_decision_from_candidate(item) for item in candidates)
    return CompositeDecision(
        reason_code=ReasonCode.COMPOSITE_PLAN,
        risk_level=_max_risk(candidates),
        explanation="Independent intent frames form an ordered execution plan.",
        source_text=" ".join(item.evidence for item in candidates),
        candidates=candidates,
        steps=steps,
    )


def build_routing_decision(routing_input: RoutingInput) -> RoutingDecision:
    candidates = detect_route_candidates(routing_input)

    def with_diagnostics(decision: RoutingDecision) -> RoutingDecision:
        diagnostics = current_router_diagnostics()
        if not diagnostics:
            return decision
        trace = dict(decision.selection_trace)
        trace["intent_router"] = diagnostics
        return replace(decision, selection_trace=trace)

    # Target validity is an invariant and must run before pending-form shortcuts
    # or composition policy. This prevents target=None from silently becoming a
    # list-all query.
    for candidate in candidates:
        target_error = _candidate_target_error(candidate, candidates)
        if target_error:
            return with_diagnostics(target_error)

    high_risk_llm = [
        item for item in candidates
        if item.source == "llm" and item.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
    ]
    deterministic_business = [
        item for item in candidates
        if item.source != "llm" and item.kind not in {RouteKind.CHAT, RouteKind.CLARIFICATION}
    ]
    if high_risk_llm and not deterministic_business:
        return with_diagnostics(_clarification(
            ReasonCode.MULTIPLE_INTENTS_CONFLICT,
            "A high-risk LLM semantic proposal requires deterministic routing evidence.",
            ("该表达可能涉及状态变更，请明确操作类型和目标。",),
            candidates,
            _max_risk(candidates),
        ))

    active_workflow_result = _active_workflow_decision(routing_input, candidates)
    if active_workflow_result:
        return with_diagnostics(active_workflow_result)

    active_form_result = _active_form_decision(routing_input, candidates)
    if active_form_result:
        return with_diagnostics(active_form_result)

    return with_diagnostics(_compose_or_clarify(candidates))
