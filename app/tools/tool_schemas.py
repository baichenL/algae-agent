"""OpenAI-style tool schemas used by the agent tool registry."""

from typing import Optional

from pydantic import BaseModel, Field


class TriggerSubcultureWorkflowArgs(BaseModel):
    strain_id: str = Field("Chlorella_01", description="Stable algae strain id to run the subculture workflow for.")
    session_id: Optional[str] = Field(None, description="Session id for traceability.")
    agent_run_id: Optional[str] = Field(None, description="Agent run id for traceability.")
    source_message: Optional[str] = Field(None, description="Original user message that requested the workflow.")
    graph_thread_id: Optional[str] = Field(None, description="Durable LangGraph thread id.")
    execution_idempotency_key: Optional[str] = Field(None, description="Per-run execution idempotency key.")
    domain_dedupe_key: Optional[str] = Field(None, description="Domain-level active pending dedupe key.")


class AddAlgaeStrainArgs(BaseModel):
    strain_id: str = Field(..., description="Stable strain id, for example Chlamy_01 or CC-125.")
    name_cn: str = Field(..., description="Chinese strain display name.")
    name_en: str = Field(..., description="English or Latin strain display name.")
    generation_number: int = Field(1, description="Current or initial generation number.")
    days_since_last_subculture: int = Field(0, description="Days since the last subculture.")
    session_id: Optional[str] = Field(None, description="Session id for traceability.")
    agent_run_id: Optional[str] = Field(None, description="Agent run id for traceability.")
    source_message: Optional[str] = Field(None, description="Original user message that requested the change.")
    graph_thread_id: Optional[str] = Field(None, description="Durable LangGraph thread id.")
    execution_idempotency_key: Optional[str] = Field(None, description="Per-run execution idempotency key.")
    domain_dedupe_key: Optional[str] = Field(None, description="Domain-level active pending dedupe key.")


class UpdateAlgaeStrainArgs(BaseModel):
    strain_id: str = Field(..., description="Stable strain id to update.")
    new_strain_id: Optional[str] = Field(None, description="New stable strain id if the strain id itself should be renamed.")
    name_cn: Optional[str] = Field(None, description="Chinese strain display name.")
    name_en: Optional[str] = Field(None, description="English or Latin strain display name.")
    generation_number: Optional[int] = Field(None, description="Current generation number.")
    days_since_last_subculture: Optional[int] = Field(None, description="Days since the last subculture.")
    session_id: Optional[str] = Field(None, description="Session id for traceability.")
    agent_run_id: Optional[str] = Field(None, description="Agent run id for traceability.")
    source_message: Optional[str] = Field(None, description="Original user message that requested the change.")
    graph_thread_id: Optional[str] = Field(None, description="Durable LangGraph thread id.")
    execution_idempotency_key: Optional[str] = Field(None, description="Per-run execution idempotency key.")
    domain_dedupe_key: Optional[str] = Field(None, description="Domain-level active pending dedupe key.")


class DeleteAlgaeStrainArgs(BaseModel):
    strain_id: str = Field(..., description="Stable strain id to delete after human approval.")
    session_id: Optional[str] = Field(None, description="Session id for traceability.")
    agent_run_id: Optional[str] = Field(None, description="Agent run id for traceability.")
    source_message: Optional[str] = Field(None, description="Original user message that requested the change.")
    graph_thread_id: Optional[str] = Field(None, description="Durable LangGraph thread id.")
    execution_idempotency_key: Optional[str] = Field(None, description="Per-run execution idempotency key.")
    domain_dedupe_key: Optional[str] = Field(None, description="Domain-level active pending dedupe key.")


class EmptyArgs(BaseModel):
    pass


class EmailDraftArgs(BaseModel):
    message: str = Field(..., description="User request or reminder content to turn into an email draft.")
    session_id: str = Field("default_session", description="Session id used to build a read-only context snapshot.")


TOOL_ARG_MODELS = {
    "trigger_subculture_workflow": TriggerSubcultureWorkflowArgs,
    "add_algae_strain": AddAlgaeStrainArgs,
    "update_algae_strain": UpdateAlgaeStrainArgs,
    "delete_algae_strain": DeleteAlgaeStrainArgs,
    "list_algae_strains": EmptyArgs,
    "list_pending_actions": EmptyArgs,
    "email_draft": EmailDraftArgs,
}


def build_tool_schema(name: str, description: str, model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


TRIGGER_SUBCULTURE_WORKFLOW_SCHEMA = build_tool_schema(
    "trigger_subculture_workflow",
    "Call when the user explicitly asks to execute, start, force, or immediately run the subculture hardware workflow for a strain. High-risk hardware workflow.",
    TriggerSubcultureWorkflowArgs,
)

ADD_ALGAE_STRAIN_SCHEMA = build_tool_schema(
    "add_algae_strain",
    "Call when the user asks to add a new algae strain. Creates a pending human-confirmation request only; does not write directly to the main table.",
    AddAlgaeStrainArgs,
)

UPDATE_ALGAE_STRAIN_SCHEMA = build_tool_schema(
    "update_algae_strain",
    "Call when the user asks to update an algae strain. Creates a pending human-confirmation request only; does not write directly to the main table.",
    UpdateAlgaeStrainArgs,
)

DELETE_ALGAE_STRAIN_SCHEMA = build_tool_schema(
    "delete_algae_strain",
    "Call when the user asks to delete an algae strain. Creates a pending human-confirmation request only; does not delete directly from the main table.",
    DeleteAlgaeStrainArgs,
)

LIST_ALGAE_STRAINS_SCHEMA = build_tool_schema(
    "list_algae_strains",
    "Call when the user asks to view, list, or query all current algae strains in the database.",
    EmptyArgs,
)

LIST_PENDING_ACTIONS_SCHEMA = build_tool_schema(
    "list_pending_actions",
    "Call when the user asks to view pending, approval-needed, or human-confirmation strain change requests.",
    EmptyArgs,
)

EMAIL_DRAFT_SCHEMA = build_tool_schema(
    "email_draft",
    "Call when the user asks to write an email, email reminder, or mailbox notification. By default this only creates a draft and waits for frontend confirmation.",
    EmailDraftArgs,
)
