from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    blank_spectrophotometer_node,
    check_schedule_node,
    cleanup_workspace_node,
    dispense_medium_node,
    load_materials_node,
    manual_load_liquid_handler_node,
    manual_load_spectrophotometer_node,
    manual_move_to_incubator_node,
    measure_absorbance_node,
    move_to_incubator_node,
    record_experiment_node,
    safe_shutdown_node,
    seal_culture_bottle_node,
    transfer_seed_culture_node,
)
from app.agent.state import AlgaeSubcultureState
from app.hardware import HardwareController, SimulatedHardware


def _route_after_schedule(state: AlgaeSubcultureState) -> str:
    status = state.get("workflow_status")
    if status == "SKIPPED":
        return END
    if status == "FAILED":
        return "SafeShutdown"
    return "ManualLoadSpectrophotometer"


def _next_or_shutdown(next_node: str):
    def route(state: AlgaeSubcultureState) -> str:
        if state.get("workflow_status") == "FAILED":
            return "SafeShutdown"
        return next_node

    return route


def build_subculture_workflow(
    controller: HardwareController | None = None,
    *,
    checkpointer: Any | None = None,
):
    """Build an isolated graph bound to one hardware controller.

    A factory is used instead of a process-global device so concurrent runs do
    not share pump, reactor, or fault state.
    """

    hardware = controller or SimulatedHardware(delay_seconds=0.0)
    workflow = StateGraph(AlgaeSubcultureState)
    workflow.add_node("CheckSchedule", check_schedule_node)
    workflow.add_node(
        "ManualLoadSpectrophotometer",
        partial(manual_load_spectrophotometer_node, controller=hardware),
    )
    workflow.add_node(
        "BlankSpectrophotometer",
        partial(blank_spectrophotometer_node, controller=hardware),
    )
    workflow.add_node(
        "MeasureAbsorbance",
        partial(measure_absorbance_node, controller=hardware),
    )
    workflow.add_node(
        "ManualLoadLiquidHandler",
        partial(manual_load_liquid_handler_node, controller=hardware),
    )
    workflow.add_node(
        "LoadMaterials",
        partial(load_materials_node, controller=hardware),
    )
    workflow.add_node(
        "DispenseMedium",
        partial(dispense_medium_node, controller=hardware),
    )
    workflow.add_node(
        "TransferSeedCulture",
        partial(transfer_seed_culture_node, controller=hardware),
    )
    workflow.add_node(
        "MixAndSeal",
        partial(seal_culture_bottle_node, controller=hardware),
    )
    workflow.add_node(
        "ManualMoveToIncubator",
        partial(manual_move_to_incubator_node, controller=hardware),
    )
    workflow.add_node(
        "MoveToIncubator",
        partial(move_to_incubator_node, controller=hardware),
    )
    workflow.add_node("RecordExperiment", record_experiment_node)
    workflow.add_node(
        "CleanupWorkspace",
        partial(cleanup_workspace_node, controller=hardware),
    )
    workflow.add_node(
        "SafeShutdown",
        partial(safe_shutdown_node, controller=hardware),
    )

    workflow.add_edge(START, "CheckSchedule")
    workflow.add_conditional_edges("CheckSchedule", _route_after_schedule)
    workflow.add_conditional_edges(
        "ManualLoadSpectrophotometer",
        _next_or_shutdown("BlankSpectrophotometer"),
    )
    workflow.add_conditional_edges(
        "BlankSpectrophotometer",
        _next_or_shutdown("MeasureAbsorbance"),
    )
    workflow.add_conditional_edges(
        "MeasureAbsorbance",
        _next_or_shutdown("ManualLoadLiquidHandler"),
    )
    workflow.add_conditional_edges(
        "ManualLoadLiquidHandler",
        _next_or_shutdown("LoadMaterials"),
    )
    workflow.add_conditional_edges(
        "LoadMaterials",
        _next_or_shutdown("DispenseMedium"),
    )
    workflow.add_conditional_edges(
        "DispenseMedium",
        _next_or_shutdown("TransferSeedCulture"),
    )
    workflow.add_conditional_edges(
        "TransferSeedCulture",
        _next_or_shutdown("MixAndSeal"),
    )
    workflow.add_conditional_edges(
        "MixAndSeal",
        _next_or_shutdown("ManualMoveToIncubator"),
    )
    workflow.add_conditional_edges(
        "ManualMoveToIncubator",
        _next_or_shutdown("MoveToIncubator"),
    )
    workflow.add_conditional_edges(
        "MoveToIncubator",
        _next_or_shutdown("RecordExperiment"),
    )
    workflow.add_edge("RecordExperiment", "CleanupWorkspace")
    workflow.add_edge("CleanupWorkspace", END)
    workflow.add_edge("SafeShutdown", END)
    return workflow.compile(checkpointer=checkpointer)


# Backward-compatible zero-delay graph for existing service and tests. New
# simulation runs use build_subculture_workflow() with an isolated controller.
algae_subculture_app = build_subculture_workflow()
