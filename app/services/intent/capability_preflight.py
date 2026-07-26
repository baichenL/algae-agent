from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.services.intent.input_normalizer import normalize_text_value


Capability = Literal[
    "policy_override",
    "raw_database_mutation",
    "self_approval",
    "unauthorized_external_execution",
]


@dataclass(frozen=True)
class CapabilityRequestSpec:
    requested_capabilities: tuple[Capability, ...] = ()
    requested_actions: tuple[str, ...] = ()
    protected_objects: tuple[str, ...] = ()
    authority_source: str = "user_message"
    explanatory_only: bool = False

    @property
    def forbidden(self) -> bool:
        return bool(self.requested_capabilities) and not self.explanatory_only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_EXPLANATION = re.compile(
    r"(?:为什么|为何|解释|说明|是什么|能否介绍|why|explain|describe|what\s+is)",
    re.I,
)
_MUTATION = re.compile(
    r"(?:忽略|覆盖|绕过|修改|改成|执行|提交|写入|删除|批准|发送|驱动|"
    r"ignore|override|bypass|modify|update|execute|commit|delete|approve|send|actuate)",
    re.I,
)
_POLICY_OBJECT = re.compile(
    r"(?:系统(?:规则|策略|提示)|开发者(?:规则|策略|提示)|system\s+(?:rule|policy|prompt)|"
    r"developer\s+(?:rule|policy|prompt))",
    re.I,
)
_DATABASE_OBJECT = re.compile(
    r"(?:\bsql\b|原始数据库|数据库(?:表|记录)?|database|algae_status|workspace_id)",
    re.I,
)
_APPROVAL_OBJECT = re.compile(
    r"(?:自行审批|自己批准|直接批准|自动审批|无需(?:用户)?确认|不需要(?:用户)?确认|"
    r"self[\s_-]*approv|approve\s+(?:it\s+)?yourself|without\s+(?:user\s+)?confirmation)",
    re.I,
)
_EXTERNAL_OBJECT = re.compile(
    r"(?:真实硬件|硬件|设备|正式邮件|外部系统|hardware|device|external\s+system|"
    r"send\s+(?:the\s+)?email|actuat)",
    re.I,
)


def classify_capability_request(message: str) -> CapabilityRequestSpec:
    """Classify requested authority before any business intent is considered."""

    text = normalize_text_value(message)
    explanatory_only = bool(_EXPLANATION.search(text)) and not bool(_MUTATION.search(text))
    capabilities: list[Capability] = []
    actions: list[str] = []
    objects: list[str] = []
    if _MUTATION.search(text):
        actions.append("state_change")
    if _POLICY_OBJECT.search(text):
        objects.append("system_policy")
        if _MUTATION.search(text):
            capabilities.append("policy_override")
    if _DATABASE_OBJECT.search(text):
        objects.append("raw_database")
        if _MUTATION.search(text):
            capabilities.append("raw_database_mutation")
    if _APPROVAL_OBJECT.search(text):
        objects.append("approval_control")
        capabilities.append("self_approval")
    if _EXTERNAL_OBJECT.search(text):
        objects.append("external_system")
        if _MUTATION.search(text):
            capabilities.append("unauthorized_external_execution")
    return CapabilityRequestSpec(
        requested_capabilities=tuple(dict.fromkeys(capabilities)),
        requested_actions=tuple(dict.fromkeys(actions)),
        protected_objects=tuple(dict.fromkeys(objects)),
        explanatory_only=explanatory_only,
    )
