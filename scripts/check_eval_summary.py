from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation.reporting import render_markdown_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated evaluation Markdown matches JSON.")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_reports" / "latest.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_reports" / "latest.md",
    )
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = render_markdown_summary(report)
    actual = args.summary.read_text(encoding="utf-8")
    if actual != expected:
        print("Evaluation Markdown summary is stale or was edited manually.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Evaluation summary matches report {report.get('report_id')}.")


if __name__ == "__main__":
    main()
