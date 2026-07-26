from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.core.workspaces import current_workspace
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
    return ToolExecutionContext(
        caller="mcp_local",
        source=source,
        session_id="mcp-local",
        principal_id="mcp-local",
        workspace_id=current_workspace().id,
        role="scientist",
    )


@mcp.resource("algae://lab/state")
async def lab_state() -> dict[str, Any]:
    strains = await execute_registered_tool(
        "list_algae_strains", {}, context=_ctx("mcp.lab_state.strains")
    )
    datasets = await execute_registered_tool(
        "scientific_datasets_list", {}, context=_ctx("mcp.lab_state.datasets")
    )
    devices = await execute_registered_tool(
        "lab_devices_list", {}, context=_ctx("mcp.lab_state.devices")
    )
    return {
        "strains": (strains.get("response_payload") or {}).get("strains") or [],
        "datasets": (datasets.get("response_payload") or {}).get("datasets") or [],
        "devices": (devices.get("response_payload") or {}).get("devices") or [],
        "boundary": "MCP resources are read-only and cannot authorize execution.",
    }


@mcp.resource("algae://datasets/{dataset_id}")
async def dataset_resource(dataset_id: str) -> dict[str, Any]:
    result = await execute_registered_tool(
        "scientific_dataset_get",
        {"dataset_id": dataset_id},
        context=_ctx("mcp.dataset_resource"),
    )
    return result.get("response_payload") or result


@mcp.resource("algae://scientific-runs/{run_id}")
async def scientific_run_resource(run_id: str) -> dict[str, Any]:
    result = await execute_registered_tool(
        "scientific_run_get",
        {"scientific_run_id": run_id},
        context=_ctx("mcp.scientific_run_resource"),
    )
    return result.get("response_payload") or result


@mcp.resource("algae://devices")
async def devices_resource() -> dict[str, Any]:
    result = await execute_registered_tool(
        "lab_devices_list", {}, context=_ctx("mcp.devices_resource")
    )
    return result.get("response_payload") or result


@mcp.resource("algae://traces/{agent_run_id}")
async def trace_resource(agent_run_id: str) -> dict[str, Any]:
    result = await execute_registered_tool(
        "agent_trace_read",
        {"agent_run_id_query": agent_run_id},
        context=_ctx("mcp.trace_resource"),
    )
    return result.get("response_payload") or result


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
