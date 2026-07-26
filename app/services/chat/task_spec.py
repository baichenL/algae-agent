from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.services.intent.routing_models import RouteKind
from app.services.intent.task_relation import parse_task_relation_signals


TaskRelation = Literal[
    "continue_current",
    "side_question",
    "start_new",
    "cancel_previous_and_start",
    "resume_named_task",
]


@dataclass(frozen=True)
class TaskSpec:
    """Canonical task contract shared by routing, persistence, and presentation."""

    intent: str
    target_refs: tuple[str, ...] = ()
    requested_capabilities: tuple[str, ...] = ()
    source_policy: str = "use_available_sources"
    candidate_count: int | None = None
    simulation_policy: str = "when_requested"
    side_effect_policy: str = "read_only"
    task_relation: TaskRelation = "start_new"
    budget_profile: str = "default"
    route_kind: str = RouteKind.CHAT.value
    task_domain: str = "conversation"
    relation_evidence: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ROUTE_DOMAIN = {
    RouteKind.EMAIL.value: "email",
    RouteKind.WRITE_ACTION.value: "strain_mutation",
    RouteKind.PENDING_FORM.value: "strain_mutation",
    RouteKind.WORKFLOW.value: "subculture",
    RouteKind.WORKFLOW_AUDIT.value: "subculture",
    RouteKind.SCIENTIFIC_TASK.value: "scientific",
    RouteKind.KNOWLEDGE_QUERY.value: "knowledge",
    RouteKind.LAB_QUERY.value: "lab_query",
    RouteKind.PENDING_QUERY.value: "pending_query",
}

_READ_ONLY_DOMAINS = {
    "conversation", "knowledge", "lab_query", "pending_query", "scientific",
}


def _normalize_task_domain(task_type: str) -> str:
    if task_type.startswith("strain_"):
        return "strain_mutation"
    if task_type.startswith("subculture"):
        return "subculture"
    if task_type.startswith("email"):
        return "email"
    if task_type.startswith("scientific"):
        return "scientific"
    return task_type


def _kind_value(decision: Any) -> str:
    kind = getattr(decision, "kind", RouteKind.CHAT)
    return str(getattr(kind, "value", kind))


def _effective_decision(decision: Any) -> Any:
    if _kind_value(decision) != RouteKind.COMPOSITE.value:
        return decision
    steps = tuple(getattr(decision, "steps", ()) or ())
    for step in reversed(steps):
        if _kind_value(step) not in {RouteKind.PENDING_FORM.value, RouteKind.WORKFLOW.value}:
            return step
    return steps[-1] if steps else decision


def _targets(decision: Any) -> tuple[str, ...]:
    refs: list[str] = []
    for item in (getattr(decision, "target", None), *(getattr(decision, "targets", ()) or ())):
        canonical = getattr(item, "canonical_id", None)
        if canonical and str(canonical) not in refs:
            refs.append(str(canonical))
    arguments = (
        getattr(decision, "arguments", None)
        or getattr(decision, "request_spec", None)
        or {}
    )
    for key in ("strain_id", "target_strain_id", "target"):
        value = arguments.get(key)
        if value and str(value) not in refs:
            refs.append(str(value))
    return tuple(refs)


def _capabilities(decision: Any, domain: str) -> tuple[str, ...]:
    if domain == "scientific":
        arguments = getattr(decision, "arguments", {}) or {}
        result = ["investigate"]
        if arguments.get("candidate_count"):
            result.append("generate_candidates")
        if arguments.get("simulation_policy") not in {None, "none"}:
            result.append("simulate")
        return tuple(result)
    if domain == "email":
        return ("draft", "request_approval")
    if domain in {"strain_mutation", "subculture"}:
        return ("request_approval",)
    return ("read",)


def _relation_markers(text: str) -> tuple[bool, bool, bool, tuple[str, ...]]:
    signals = parse_task_relation_signals(text)
    return (
        signals.cancel_previous,
        signals.start_new,
        signals.resume_named,
        signals.evidence,
    )


def is_email_approval_followup(message: str) -> bool:
    """Interpret a short follow-up only inside an already established email task."""
    normalized = " ".join(str(message or "").casefold().split())
    return any(
        phrase in normalized
        for phrase in (
            "帮我发送", "为我发送", "生成申请", "创建申请", "提交申请",
            "send it", "send this", "create the approval", "submit for approval",
        )
    )


def build_task_spec(*, message: str, decision: Any, active_task: dict[str, Any] | None) -> TaskSpec:
    decision = _effective_decision(decision)
    route_kind = _kind_value(decision)
    domain = _ROUTE_DOMAIN.get(route_kind, "conversation")
    active_domain = _normalize_task_domain(
        str((active_task or {}).get("task_type") or "")
    )
    normalized_message = str(message or "").casefold()
    if any(phrase in normalized_message for phrase in ("继续邮件任务", "恢复邮件任务", "resume the email task")):
        route_kind = RouteKind.EMAIL.value
        domain = "email"
    elif any(
        phrase in normalized_message
        for phrase in (
            "继续科学任务", "继续调查任务", "恢复科学任务",
            "resume the scientific task",
        )
    ):
        route_kind = RouteKind.SCIENTIFIC_TASK.value
        domain = "scientific"
    if active_domain == "email" and is_email_approval_followup(message):
        route_kind = RouteKind.EMAIL.value
        domain = "email"
    cancel_previous, explicit_new, explicit_resume, evidence = _relation_markers(message)
    if cancel_previous:
        relation: TaskRelation = "cancel_previous_and_start"
    elif explicit_resume:
        relation = "resume_named_task"
    elif explicit_new:
        relation = "start_new"
    elif active_task and active_domain == domain:
        relation = "continue_current"
    elif active_task and domain in _READ_ONLY_DOMAINS:
        relation = "side_question"
    else:
        relation = "start_new"

    arguments = (
        getattr(decision, "arguments", None)
        or getattr(decision, "request_spec", None)
        or {}
    )
    return TaskSpec(
        intent=str(getattr(decision, "explanation", None) or route_kind),
        target_refs=_targets(decision),
        requested_capabilities=_capabilities(decision, domain),
        source_policy=str(arguments.get("source_policy") or "use_available_sources"),
        candidate_count=arguments.get("candidate_count"),
        simulation_policy=str(arguments.get("simulation_policy") or "when_requested"),
        side_effect_policy=(
            "approval_required"
            if domain in {"email", "strain_mutation", "subculture"}
            else "read_only"
        ),
        task_relation=relation,
        budget_profile=str(arguments.get("budget_profile") or "default"),
        route_kind=route_kind,
        task_domain=domain,
        relation_evidence=evidence,
    )
