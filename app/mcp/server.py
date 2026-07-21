from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.core import database
from app.core.db import scientific as scientific_db
from app.services.observability.agent_trace import build_agent_run_trace
# Initialize the runtime package before importing the shared Tool Gateway. The
# existing chat stack also owns dispatcher imports, so this order avoids a
# package-initialization cycle when the MCP server is launched standalone.
import app.services.agent_runtime as _runtime_bootstrap  # noqa: F401
from app.tools.executor import ToolExecutionContext, execute_registered_tool


mcp = FastMCP(
    "algae-lab",
    instructions=(
        "Local proposal-only interface for microalgae lab state, growth diagnosis, "
        "experiment previews and pending proposal requests. Approval and physical execution are never exposed."
    ),
    json_response=True,
)


def _ctx(source: str) -> ToolExecutionContext:
    return ToolExecutionContext(caller="mcp_local", source=source, session_id="mcp-local")


@mcp.resource("algae://lab/state")
def lab_state() -> dict[str, Any]:
    return {
        "strains": database.list_algae_status(),
        "datasets": scientific_db.list_datasets(),
        "devices": scientific_db.list_lab_devices(),
        "boundary": "MCP resources are read-only and cannot authorize execution.",
    }


@mcp.resource("algae://datasets/{dataset_id}")
def dataset_resource(dataset_id: str) -> dict[str, Any]:
    return scientific_db.get_dataset(dataset_id, include_measurements=False) or {"error": "dataset_not_found"}


@mcp.resource("algae://scientific-runs/{run_id}")
def scientific_run_resource(run_id: str) -> dict[str, Any]:
    return scientific_db.get_scientific_run(run_id) or {"error": "scientific_run_not_found"}


@mcp.resource("algae://devices")
def devices_resource() -> dict[str, Any]:
    return {"devices": scientific_db.list_lab_devices(), "physical_execution_available": False}


@mcp.resource("algae://traces/{agent_run_id}")
def trace_resource(agent_run_id: str) -> dict[str, Any]:
    return build_agent_run_trace(agent_run_id) or {"error": "agent_trace_not_found"}


@mcp.tool()
async def datasets_list() -> dict[str, Any]:
    """List imported scientific datasets through the governed Tool Gateway."""
    return await execute_registered_tool("scientific_datasets_list", {}, context=_ctx("mcp.datasets_list"))


@mcp.tool()
async def growth_diagnosis_start(
    dataset_id: str,
    target_metric: str | None = None,
    offline_replay: bool = False,
) -> dict[str, Any]:
    """Run deterministic diagnosis only; this tool never creates an approval or executes hardware."""
    return await execute_registered_tool(
        "growth_diagnosis_start",
        {
            "dataset_id": dataset_id,
            "target_metric": target_metric,
            "mode": "diagnose",
            "offline_replay": offline_replay,
            "session_id": "mcp-local",
        },
        context=_ctx("mcp.growth_diagnosis_start"),
    )


@mcp.tool()
async def experiment_design_preview(scientific_run_id: str) -> dict[str, Any]:
    """Read the latest frozen experiment design without creating a pending request."""
    return await execute_registered_tool(
        "experiment_design_preview",
        {"scientific_run_id": scientific_run_id},
        context=_ctx("mcp.experiment_design_preview"),
    )


@mcp.tool()
async def experiment_proposal_request(scientific_run_id: str) -> dict[str, Any]:
    """Create a pending proposal; MCP deliberately exposes no approval or execution tool."""
    return await execute_registered_tool(
        "experiment_proposal_request",
        {"scientific_run_id": scientific_run_id},
        context=_ctx("mcp.experiment_proposal_request"),
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
