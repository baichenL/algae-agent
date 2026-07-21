from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.db import scientific as scientific_db
from app.core.db.agent_events import list_agent_run_events
from app.services.scientific.demo_data import scientific_demo_file
from app.services.scientific.importer import parse_scientific_dataset
from app.services.scientific.service import approve_and_simulate_proposal, run_scientific_task
from scripts.run_agent_eval import configure_temp_db


def run_demo(*, run_seed: int = 2025, approve_cycles: int = 2) -> dict[str, Any]:
    content, filename, mapping = scientific_demo_file()
    parsed = parse_scientific_dataset(
        content=content,
        filename=filename,
        dataset_name="25-condition algae growth demo",
        strain_id="Chlorella_01",
        mapping=mapping,
    )
    scientific_db.insert_dataset(parsed.dataset, parsed.batches)
    run = run_scientific_task(
        dataset_id=parsed.dataset["id"],
        mode="diagnose_and_optimize",
        offline_replay=True,
        run_seed=run_seed,
        max_cycles=2,
        session_id="self-driving-lab-demo",
    )
    cycles: list[dict[str, Any]] = []
    pending_id = (run.get("proposal") or {}).get("pending_id")
    for _ in range(max(0, min(approve_cycles, 2))):
        if not pending_id:
            break
        result = approve_and_simulate_proposal(int(pending_id), reviewed_by="demo-approver")
        cycles.append(result)
        pending_id = (result.get("next_pending") or {}).get("pending_id")
    final_run = scientific_db.get_scientific_run(run["id"]) or run
    return {
        "notice": "All experiment results in this demo are simulation_only; no wet-lab execution is claimed.",
        "dataset": {
            "id": parsed.dataset["id"],
            "source_file": filename,
            "rows": parsed.dataset["row_count"],
            "batches": len(parsed.batches),
        },
        "scientific_run": final_run,
        "approved_digital_twin_cycles": cycles,
        "trace": list_agent_run_events(final_run["agent_run_id"]),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the complete simulation-only self-driving algae lab demo.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--approve-cycles", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="algae_scientific_demo_", ignore_cleanup_errors=True) as tmpdir:
        configure_temp_db(Path(tmpdir) / "demo.sqlite3")
        report = run_demo(run_seed=args.seed, approve_cycles=args.approve_cycles)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
