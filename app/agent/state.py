from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AlgaeSubcultureState(TypedDict, total=False):
    """Serializable state for the automated subculture workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    run_id: str
    execution_mode: str
    interaction_mode: str
    strain_id: str
    generation_number: int
    inoculation_timestamp: str
    source_reactor_id: str
    target_reactor_id: str
    media_target_volume: float
    seed_target_volume: float
    wavelength_nm: float
    simulated_absorbance: float
    measured_absorbance: float | None
    days_since_last_subculture: int
    current_step: str
    hardware_logs: list[str]
    simulation_events: list[dict[str, Any]]
    hardware_state: dict[str, Any]
    workflow_status: str
    last_error: dict[str, Any] | None
    pending_manual_task: dict[str, Any] | None
    protocol_id: str
    protocol_hash: str
