from app.schemas.algae import ChatResponse
from app.services.chat.response_builder import _complete_chat_response
from app.services.intent.intent_handlers import build_tool_info_response, handle_query_status_intent
from app.services.intent.routing_models import QueryDecision, ToolInfoDecision
from app.tools.executor import get_available_tool_schemas


def handle_tool_info_intent(
    session_id: str,
    conversation_history: list,
    decision: ToolInfoDecision,
) -> ChatResponse:
    return _complete_chat_response(
        session_id,
        conversation_history,
        build_tool_info_response(get_available_tool_schemas(include_high_risk=False)),
    )


def handle_query_intent(
    session_id: str,
    conversation_history: list,
    decision: QueryDecision,
    context_snapshot,
    query_status_handler=handle_query_status_intent,
) -> ChatResponse:
    return _complete_chat_response(
        session_id,
        conversation_history,
        query_status_handler(decision, context_snapshot),
    )
