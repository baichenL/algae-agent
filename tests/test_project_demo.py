import json
import subprocess
import sys


def test_project_demo_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/run_project_demo.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert len(payload["demo_steps"]) == 3
    workflow_step = payload["demo_steps"][2]
    assert workflow_step["action"] == "workflow_subculture"
    assert workflow_step["status"] == "pending"
    assert workflow_step["pending_id"]
    assert workflow_step["requires_approval"] is True
    assert payload["workflow_trace_summary"]["final_status"] == "waiting_approval"
    assert payload["workflow_trace_summary"]["requires_approval"] is True
    assert payload["runtime_eval"]["failed_count"] == 0
