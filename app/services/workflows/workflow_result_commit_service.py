from __future__ import annotations

from typing import Any

from app.core import database


def commit_workflow_result(
    *,
    strain_id: str,
    expected_generation: int | None,
    tool_result: dict[str, Any],
) -> tuple[bool, str | None]:
    """Commit only verified physical workflow results to laboratory facts."""

    if tool_result.get("status") != "success":
        return False, "tool_result_not_successful"
    if not tool_result.get("physical_execution"):
        return False, "simulation_not_persisted"
    current = database.get_algae_status(strain_id)
    if not current:
        return False, "strain_not_found"
    current_generation = int(current.get("generation_number") or 0)
    if expected_generation is not None and current_generation != int(expected_generation):
        return False, "stale_generation"
    next_generation = tool_result.get("current_generation")
    inoculation_time = tool_result.get("inoculation_time")
    if next_generation is None or not inoculation_time:
        return False, "incomplete_tool_result"

    database.update_algae_status(
        strain_id,
        int(next_generation),
        int(tool_result.get("days_counter") or 0),
        inoculation_time,
    )
    database.insert_experiment({
        "strain": strain_id,
        "generation": int(next_generation),
        "status": "SUCCESS",
        "hardware_logs": tool_result.get("execution_logs") or [],
        "created_at": inoculation_time,
    })
    return True, None
