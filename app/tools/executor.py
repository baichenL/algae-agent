import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from pydantic import ValidationError

from app.services.agent_runtime.events import record_run_event
from app.services.observability.error_events import record_error_event
from app.tools.registry import AGENT_TOOL_REGISTRY
from app.tools.tool_schemas import TOOL_ARG_MODELS


HIGH_RISK = "high"


class ToolExecutionError(Exception):
    pass


@dataclass(frozen=True)
class ToolExecutionContext:
    caller: str = "chat_runtime"
    session_id: str | None = None
    agent_run_id: str | None = None
    approval_id: int | None = None
    source: str | None = None
    principal_id: str | None = None
    workspace_id: str | None = None
    role: str | None = None
    tool_call_id: str | None = None

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "session_id": self.session_id,
            "agent_run_id": self.agent_run_id,
            "approval_id": self.approval_id,
            "source": self.source,
            "principal_id": self.principal_id,
            "workspace_id": self.workspace_id,
            "role": self.role,
            "tool_call_id": self.tool_call_id,
        }


def get_available_tool_schemas(include_high_risk: bool = False) -> List[Dict[str, Any]]:
    schemas = []
    for tool_name, definition in AGENT_TOOL_REGISTRY.items():
        if not include_high_risk and is_high_risk_tool(tool_name):
            continue
        schemas.append(definition["schema"])
    return schemas


def is_high_risk_tool(tool_name: str) -> bool:
    definition = AGENT_TOOL_REGISTRY.get(tool_name) or {}
    return definition.get("risk_level") == HIGH_RISK


def parse_tool_arguments(raw_arguments: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_arguments or "{}")
    except Exception:
        return {}


def _trace_ids(function_args: Dict[str, Any] | None) -> tuple[str | None, str | None]:
    args = function_args or {}
    return args.get("session_id"), args.get("agent_run_id")


def _execution_context(
    function_args: Dict[str, Any] | None,
    context: ToolExecutionContext | None,
) -> ToolExecutionContext:
    args = function_args or {}
    if context:
        return context
    return ToolExecutionContext(
        caller=args.get("caller") or "chat_runtime",
        session_id=args.get("session_id"),
        agent_run_id=args.get("agent_run_id"),
        approval_id=args.get("approval_id"),
        source=args.get("source"),
    )


def _args_with_context(
    function_args: Dict[str, Any] | None,
    context: ToolExecutionContext,
) -> Dict[str, Any]:
    args = dict(function_args or {})
    if context.session_id and not args.get("session_id"):
        args["session_id"] = context.session_id
    if context.agent_run_id and not args.get("agent_run_id"):
        args["agent_run_id"] = context.agent_run_id
    if context.approval_id and not args.get("approval_id"):
        args["approval_id"] = context.approval_id
    if context.tool_call_id and not args.get("created_by_tool_call_id"):
        args["created_by_tool_call_id"] = context.tool_call_id
    return args


def _tool_metadata(definition: Dict[str, Any]) -> Dict[str, Any]:
    metadata = definition.get("metadata")
    if isinstance(metadata, dict) and metadata:
        return metadata
    keys = {
        "risk_level",
        "effect_kind",
        "side_effect",
        "requires_approval",
        "allowed_callers",
        "idempotency_fields",
        "audit_event_type",
        "exposed_to_llm",
    }
    return {key: definition.get(key) for key in keys if key in definition}


def _record_tool_event(
    event_type: str,
    tool_name: str,
    function_args: Dict[str, Any] | None,
    payload: Dict[str, Any] | None = None,
) -> None:
    session_id, agent_run_id = _trace_ids(function_args)
    record_run_event(
        agent_run_id,
        session_id=session_id,
        event_type=event_type,
        layer="tool_calling",
        payload={"tool_name": tool_name, **(payload or {})},
    )


def _record_tool_error(
    tool_name: str,
    function_args: Dict[str, Any] | None,
    *,
    operation: str,
    error_type: str,
    message: str,
    severity: str = "error",
    metadata: Dict[str, Any] | None = None,
) -> int | None:
    session_id, agent_run_id = _trace_ids(function_args)
    return record_error_event(
        session_id=session_id,
        agent_run_id=agent_run_id,
        layer="tool_calling",
        component="tool_executor",
        operation=operation,
        severity=severity,
        error_type=error_type,
        error_message=message,
        metadata={"tool_name": tool_name, **(metadata or {})},
    )


def _error_result(tool_name: str, message: str, error_event_id: int | None = None) -> Dict[str, Any]:
    payload = {
        "action": tool_name,
        "status": "error",
        "message": message,
        "msg": message,
    }
    if error_event_id is not None:
        payload["error_event_id"] = error_event_id
    return {
        "action": tool_name,
        "status": "error",
        "message": message,
        **({"error_event_id": error_event_id} if error_event_id is not None else {}),
        "response_payload": payload,
        "memory_text": message,
    }


def _blocked_result(tool_name: str, message: str, error_event_id: int | None = None) -> Dict[str, Any]:
    result = _error_result(tool_name, message, error_event_id)
    result["status"] = "blocked"
    result["response_payload"]["status"] = "blocked"
    return result


def _more_info_result(tool_name: str, missing_fields: List[str], message: str) -> Dict[str, Any]:
    payload = {
        "action": tool_name,
        "status": "needs_more_info",
        "message": message,
        "msg": message,
        "require_more_info": True,
        "missing_fields": missing_fields,
        "questions": [f"Please provide {field}." for field in missing_fields],
    }
    return {
        "action": tool_name,
        "status": "needs_more_info",
        "message": message,
        "require_more_info": True,
        "missing_fields": missing_fields,
        "questions": payload["questions"],
        "response_payload": payload,
        "memory_text": message,
    }


def _normalize_result(tool_name: str, result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return _error_result(tool_name, f"Tool {tool_name} returned an invalid result")

    response_payload = result.get("response_payload")
    if not isinstance(response_payload, dict):
        response_payload = {
            key: value
            for key, value in result.items()
            if key not in {"memory_text", "response_payload"}
        }

    action = result.get("action") or response_payload.get("action") or tool_name
    status = result.get("status") or response_payload.get("status") or "success"
    message = result.get("message") or response_payload.get("message") or response_payload.get("msg") or result.get("memory_text") or "Tool execution completed"

    response_payload.setdefault("action", action)
    response_payload.setdefault("status", status)
    response_payload.setdefault("message", message)
    response_payload.setdefault("msg", message)

    normalized = {
        **result,
        "action": action,
        "status": status,
        "message": message,
        "response_payload": response_payload,
        "memory_text": result.get("memory_text") or message,
    }
    return normalized


def _authorize_tool_call(
    tool_name: str,
    definition: Dict[str, Any],
    function_args: Dict[str, Any],
    context: ToolExecutionContext,
    *,
    allow_high_risk: bool,
) -> Dict[str, Any] | None:
    metadata = _tool_metadata(definition)
    if not metadata:
        _record_tool_event(
            "tool_metadata_warning",
            tool_name,
            function_args,
            {
                "warning": "metadata_missing",
                "context": context.to_event_payload(),
            },
        )
        if is_high_risk_tool(tool_name) and not allow_high_risk:
            error_event_id = _record_tool_error(
                tool_name,
                function_args,
                operation="authorize_tool",
                error_type="HighRiskToolBlocked",
                message=f"High-risk tool cannot run from this path: {tool_name}",
                severity="warning",
                metadata={"allow_high_risk": allow_high_risk, "context": context.to_event_payload()},
            )
            return _blocked_result(tool_name, f"High-risk tool cannot run from this path: {tool_name}", error_event_id)
        return None

    allowed_callers = set(metadata.get("allowed_callers") or [])
    if allowed_callers and context.caller not in allowed_callers:
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="authorize_tool",
            error_type="ToolCallerNotAllowed",
            message=f"Caller {context.caller} is not allowed to invoke tool {tool_name}",
            severity="warning",
            metadata={
                "allowed_callers": sorted(allowed_callers),
                "context": context.to_event_payload(),
            },
        )
        return _blocked_result(tool_name, f"Caller is not allowed to invoke tool: {tool_name}", error_event_id)

    allowed_roles = {str(item).lower() for item in metadata.get("allowed_roles") or []}
    if allowed_roles and (context.role or "").lower() not in allowed_roles:
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="authorize_tool",
            error_type="ToolRoleNotAllowed",
            message=f"Role {context.role or 'unknown'} is not allowed to invoke tool {tool_name}",
            severity="warning",
            metadata={"allowed_roles": sorted(allowed_roles), "context": context.to_event_payload()},
        )
        return _blocked_result(tool_name, f"Role is not allowed to invoke tool: {tool_name}", error_event_id)

    if context.workspace_id:
        from app.core.workspaces import current_workspace

        if current_workspace().id != context.workspace_id:
            error_event_id = _record_tool_error(
                tool_name,
                function_args,
                operation="authorize_tool",
                error_type="CrossWorkspaceToolCallBlocked",
                message="Tool execution context does not match the active workspace",
                severity="warning",
                metadata={"context": context.to_event_payload()},
            )
            return _blocked_result(tool_name, "Cross-workspace tool call is forbidden", error_event_id)

    effect_kind = metadata.get("effect_kind")
    risk_level = metadata.get("risk_level")
    if effect_kind == "execute" and risk_level == HIGH_RISK and context.caller != "workflow_approval_service":
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="authorize_tool",
            error_type="HighRiskExecuteBlocked",
            message=f"High-risk execute tool requires workflow approval service: {tool_name}",
            severity="warning",
            metadata={"context": context.to_event_payload()},
        )
        return _blocked_result(tool_name, f"High-risk execute tool cannot run from this caller: {tool_name}", error_event_id)

    return None


def _contract_error_for_result(
    tool_name: str,
    metadata: Dict[str, Any],
    normalized: Dict[str, Any],
    context: ToolExecutionContext,
) -> str | None:
    payload = normalized.get("response_payload") or {}
    effect_kind = metadata.get("effect_kind")
    if payload.get("action") == "email_sent" and context.caller == "chat_runtime":
        return "Chat runtime may create an email draft, but it may not send email directly."
    if effect_kind == "propose":
        has_pending = bool(normalized.get("pending_id") or payload.get("pending_id"))
        has_confirmation = bool(payload.get("require_confirmation") or payload.get("requires_confirmation"))
        if normalized.get("status") == "success" and not (has_pending or has_confirmation):
            return "Propose tools must return a pending request or explicit confirmation boundary."
    if effect_kind == "draft":
        has_confirmation = bool(payload.get("require_confirmation") or payload.get("requires_confirmation"))
        if normalized.get("status") == "success" and not has_confirmation:
            return "Draft tools must return an explicit confirmation boundary."
    return None


def _validate_tool_arguments(tool_name: str, function_args: Dict[str, Any]) -> Dict[str, Any]:
    model = TOOL_ARG_MODELS.get(tool_name)
    if not model:
        return function_args or {}
    return model(**(function_args or {})).model_dump(exclude_none=True)


async def execute_registered_tool(
    tool_name: str,
    function_args: Dict[str, Any],
    context: ToolExecutionContext | None = None,
    allow_high_risk: bool = False,
) -> Dict[str, Any]:
    execution_context = _execution_context(function_args, context)
    function_args = _args_with_context(function_args, execution_context)
    definition = AGENT_TOOL_REGISTRY.get(tool_name)
    if not definition:
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="lookup_tool",
            error_type="ToolNotRegistered",
            message=f"Tool is not registered: {tool_name}",
            metadata={"available_tools": sorted(AGENT_TOOL_REGISTRY.keys())},
        )
        return _error_result(tool_name, f"Tool is not registered: {tool_name}", error_event_id)

    authorization_error = _authorize_tool_call(
        tool_name,
        definition,
        function_args,
        execution_context,
        allow_high_risk=allow_high_risk,
    )
    if authorization_error is not None:
        _record_tool_event(
            "tool_call_finished",
            tool_name,
            function_args,
            {
                "status": authorization_error.get("status"),
                "reason": "tool_authorization_failed",
                "context": execution_context.to_event_payload(),
            },
        )
        return authorization_error

    try:
        validated_args = _validate_tool_arguments(tool_name, function_args)
    except ValidationError as exc:
        missing_fields = [
            str(error.get("loc", ["field"])[0])
            for error in exc.errors()
            if error.get("type") == "missing"
        ]
        if missing_fields:
            return _more_info_result(tool_name, missing_fields, "Tool arguments are missing required fields")
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="validate_arguments",
            error_type="ValidationError",
            message=f"Tool argument validation failed: {exc}",
            metadata={"errors": exc.errors()},
        )
        return _error_result(tool_name, f"Tool argument validation failed: {exc}", error_event_id)

    try:
        _record_tool_event(
            "tool_call_started",
            tool_name,
            validated_args,
            {
                "risk_level": definition.get("risk_level", "normal"),
                "context": execution_context.to_event_payload(),
            },
        )
        handler = definition["handler"]
        result = handler(validated_args)
        if inspect.isawaitable(result):
            result = await result
        normalized = _normalize_result(tool_name, result)
        error_event_id = None
        if normalized.get("status") == "error":
            error_event_id = _record_tool_error(
                tool_name,
                validated_args,
                operation="execute_tool",
                error_type="ToolReturnedError",
                message=normalized.get("message") or "Tool returned an error result",
                metadata={"action": normalized.get("action")},
            )
            if error_event_id is not None:
                normalized["error_event_id"] = error_event_id
                normalized["response_payload"]["error_event_id"] = error_event_id
        metadata = _tool_metadata(definition)
        contract_error = _contract_error_for_result(tool_name, metadata, normalized, execution_context)
        if contract_error:
            error_event_id = _record_tool_error(
                tool_name,
                validated_args,
                operation="validate_result_contract",
                error_type="ToolContractViolation",
                message=contract_error,
                severity="warning",
                metadata={
                    "action": normalized.get("action"),
                    "status": normalized.get("status"),
                    "context": execution_context.to_event_payload(),
                },
            )
            return _blocked_result(tool_name, contract_error, error_event_id)
        _record_tool_event(
            "tool_call_finished",
            tool_name,
            validated_args,
            {
                "status": normalized.get("status"),
                "action": normalized.get("action"),
                "pending_id": normalized.get("pending_id")
                or normalized.get("response_payload", {}).get("pending_id"),
                "error_event_id": error_event_id,
                "context": execution_context.to_event_payload(),
            },
        )
        return normalized
    except Exception as exc:
        error_event_id = _record_tool_error(
            tool_name,
            function_args,
            operation="execute_tool",
            error_type=type(exc).__name__,
            message=f"Tool execution failed: {str(exc)}",
        )
        _record_tool_event(
            "tool_call_finished",
            tool_name,
            function_args,
            {"status": "error", "error_event_id": error_event_id},
        )
        return _error_result(tool_name, f"Tool execution failed: {str(exc)}", error_event_id)
