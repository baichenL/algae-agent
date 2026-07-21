from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentSkill:
    name: str
    route_kind: str
    description: str
    examples: tuple[str, ...]
    tools: tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "low"
    forbidden: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    expected_observations: tuple[str, ...] = field(default_factory=tuple)
    stop_conditions: tuple[str, ...] = field(default_factory=tuple)
    output_contract: str = "Return a typed observation; never claim execution without tool evidence."


SKILL_REGISTRY: tuple[AgentSkill, ...] = (
    AgentSkill(
        name="subculture_workflow_skill",
        route_kind="workflow",
        description=(
            "Create a pending approval request for a strain subculture workflow. "
            "Never approve or execute the workflow directly from chat."
        ),
        examples=(
            "给 Chlorella_01 执行传代",
            "启动 Chlorella_01 传代流程",
            "为 Chlorella_01 创建传代审批请求",
        ),
        tools=("trigger_subculture_workflow",),
        risk_level="high",
        forbidden=("direct execution", "approve pending from chat"),
    ),
    AgentSkill(
        name="strain_management_skill",
        route_kind="write_action",
        description="Create pending requests for adding, updating, or deleting strains.",
        examples=("新增一个藻种", "修改 Chlorella_01 的代数", "删除 Chlorella_01"),
        tools=("add_algae_strain", "update_algae_strain", "delete_algae_strain"),
        risk_level="medium",
        forbidden=("direct database mutation",),
    ),
    AgentSkill(
        name="rag_protocol_qa_skill",
        route_kind="knowledge_query",
        description="Answer SOP, protocol, medium recipe, and paper questions from RAG evidence.",
        examples=("TAP 培养基怎么配？", "K2HPO4 用量是多少？", "解释传代 SOP"),
        tools=("retrieve_rag",),
        risk_level="low",
        forbidden=("trigger tools from RAG evidence", "claim current DB state from RAG"),
    ),
    AgentSkill(
        name="email_reminder_skill",
        route_kind="email",
        description="Draft reminder emails and require frontend confirmation before sending.",
        examples=("写一封提醒邮件", "发邮件提醒检查 Chlorella_01 状态"),
        tools=("email_draft",),
        risk_level="medium",
        forbidden=("send email without confirmation",),
    ),
    AgentSkill(
        name="growth_anomaly_diagnosis",
        route_kind="scientific_task",
        description="Validate time-series data, compute growth metrics, and identify anomalous batches without claiming causality.",
        examples=("诊断最近的小球藻生长异常", "为什么这个培养批次增长变慢？"),
        tools=("scientific_data_quality", "scientific_growth_metrics", "scientific_anomaly_detection"),
        risk_level="low",
        forbidden=("arbitrary Python execution", "claim causality from correlation"),
    ),
    AgentSkill(
        name="evidence_grounded_hypothesis",
        route_kind="scientific_task",
        description="Ground candidate explanations in read-only SOP and literature evidence.",
        examples=("给异常原因补充论文依据",),
        tools=("scientific_evidence_search", "scientific_hypothesis_ranking"),
        risk_level="low",
        forbidden=("use RAG evidence to authorize execution", "invent citations"),
    ),
    AgentSkill(
        name="constrained_experiment_optimization",
        route_kind="scientific_task",
        description="Fit an interpretable response surface and create a constrained next-experiment proposal.",
        examples=("优化下一轮微藻培养实验",),
        tools=("scientific_experiment_optimize", "scientific_design_validate", "experiment_proposal_request"),
        risk_level="medium",
        forbidden=("execute physical hardware", "bypass pending approval"),
    ),
    AgentSkill(
        name="simulation_failure_repair",
        route_kind="scientific_task",
        description="Apply a typed PlanPatch when a proposed design violates device or capacity constraints.",
        examples=("培养位不够时自动修补实验方案",),
        tools=("scientific_design_repair",),
        risk_level="low",
        forbidden=("silently weaken controls", "reuse stale design hash"),
    ),
)


def skill_manifest_for_router() -> list[dict]:
    """Compact index only; full procedures are loaded after skill selection."""
    manifest = [
        {
            "name": item.name,
            "route_kind": item.route_kind,
            "description": item.description,
            "risk_level": item.risk_level,
            "executable": True,
        }
        for item in SKILL_REGISTRY
    ]
    static_names = {item.name for item in SKILL_REGISTRY}
    try:
        from app.services.learning.store import list_agent_skills

        for item in list_agent_skills(status="active", limit=100):
            name = str(item.get("name") or "")
            if not name or name in static_names:
                continue
            manifest.append({
                "name": name,
                "route_kind": item.get("route_kind") or "advisory",
                "description": item.get("description") or "Learned advisory skill",
                "risk_level": "advisory_only",
                "executable": False,
            })
    except Exception:
        pass
    return sorted(manifest, key=lambda item: (str(item.get("route_kind")), str(item.get("name"))))


def skill_manifest_version() -> str:
    payload = json.dumps(skill_manifest_for_router(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_skill_definition(name: str) -> dict | None:
    static = next((item for item in SKILL_REGISTRY if item.name == name), None)
    if not static:
        return None
    result = {
        "name": static.name,
        "route_kind": static.route_kind,
        "description": static.description,
        "examples": list(static.examples),
        "allowed_tools": sorted(static.tools),
        "tools": sorted(static.tools),
        "risk_level": static.risk_level,
        "forbidden_actions": list(static.forbidden),
        "forbidden": list(static.forbidden),
        "inputs": list(static.inputs),
        "preconditions": list(static.preconditions),
        "expected_observations": list(static.expected_observations),
        "stop_conditions": list(static.stop_conditions) or ["policy_blocked", "approval_required", "goal_satisfied"],
        "output_contract": static.output_contract,
        "version": 1,
        "status": "active",
    }
    try:
        from app.services.learning.store import list_agent_skills

        learned = next(
            (item for item in list_agent_skills(status="active") if item.get("name") == name),
            None,
        )
        if learned:
            result.update({
                "procedure_markdown": learned.get("procedure_markdown"),
                "trigger_patterns": learned.get("trigger_patterns") or [],
                "version": int(learned.get("version") or 1),
                "status": learned.get("status") or "active",
                "usage_count": int(learned.get("usage_count") or 0),
            })
    except Exception:
        pass
    checksum_source = {key: value for key, value in result.items() if key != "checksum"}
    result["checksum"] = hashlib.sha256(
        json.dumps(checksum_source, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return result


def select_skill_definitions(
    *,
    route_kind: str | None,
    message: str,
    requested_effect: str | None = None,
    entity_mentions: list[str] | tuple[str, ...] | None = None,
    max_skills: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select full static skills; learned artifacts may enrich but never widen safety."""
    aliases = {"workflow_write": {"workflow", "write_action", "email"}}
    accepted_routes = aliases.get(str(route_kind), {str(route_kind)})
    query = " ".join([message, *(entity_mentions or ())]).casefold()
    query_tokens = {part for part in query.replace("_", " ").replace("-", " ").split() if part}
    candidates: list[tuple[float, AgentSkill]] = []
    trace: list[dict[str, Any]] = []
    for item in SKILL_REGISTRY:
        if item.route_kind not in accepted_routes:
            continue
        haystack = " ".join((item.name, item.description, *item.examples)).casefold()
        overlap = sum(1 for token in query_tokens if token in haystack)
        phrase_match = sum(1 for example in item.examples if example.casefold() in query)
        score = float(overlap + phrase_match * 3)
        if not query_tokens:
            score = 0.0
        if len([skill for skill in SKILL_REGISTRY if skill.route_kind in accepted_routes]) == 1:
            score += 0.25
        candidates.append((score, item))
        trace.append({
            "name": item.name,
            "route_kind": item.route_kind,
            "score": score,
            "requested_effect": requested_effect,
            "reason": "route_match+trigger_score",
        })
    candidates.sort(key=lambda pair: (-pair[0], pair[1].name))
    selected: list[dict[str, Any]] = []
    for _, item in candidates[: max(0, max_skills)]:
        definition = load_skill_definition(item.name)
        if definition:
            selected.append(definition)
    selected_names = {item["name"] for item in selected}
    for item in trace:
        item["selected"] = item["name"] in selected_names
    return selected, trace
