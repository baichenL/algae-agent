from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _write_junit(path: Path, results: list[dict[str, Any]]) -> None:
    suite = ET.Element(
        "testsuite",
        name="agent-v2-uat",
        tests=str(len(results)),
        failures=str(sum(not item["evaluation"]["passed"] for item in results)),
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            name=f"{result['case_id']}#{result.get('iteration', 1)}",
        )
        if not result["evaluation"]["passed"]:
            failure = ET.SubElement(
                case,
                "failure",
                message="; ".join(result["evaluation"]["failures"]),
            )
            failure.text = json.dumps(
                result["evaluation"], ensure_ascii=False, sort_keys=True
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace failed UAT case executions with focused rechecks."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--junit", required=True)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    replacements: dict[str, list[dict[str, Any]]] = {}
    for override_path in args.override:
        override = _load(override_path)
        current: dict[str, list[dict[str, Any]]] = {}
        for result in override.get("results") or []:
            current.setdefault(result["case_id"], []).append(result)
        replacements.update(current)

    results: list[dict[str, Any]] = []
    replaced_cases: set[str] = set()
    for result in baseline.get("results") or []:
        case_id = result["case_id"]
        if case_id in replacements:
            if case_id not in replaced_cases:
                results.extend(replacements[case_id])
                replaced_cases.add(case_id)
            continue
        results.append(result)

    missing = set(replacements) - replaced_cases
    if missing:
        raise ValueError(f"override cases absent from baseline: {sorted(missing)}")
    case_ids = {item["case_id"] for item in results}
    expected_case_count = int(baseline.get("manifest_case_count") or 0)
    if len(case_ids) != expected_case_count:
        raise ValueError(
            f"case coverage changed: expected {expected_case_count}, got {len(case_ids)}"
        )

    payload = {
        "schema_version": "agent-v2-uat-result/v2",
        "manifest_case_count": expected_case_count,
        "execution_count": len(results),
        "passed": sum(item["evaluation"]["passed"] for item in results),
        "failed": sum(not item["evaluation"]["passed"] for item in results),
        "replaced_cases": sorted(replaced_cases),
        "evidence_sources": [args.baseline, *args.override],
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    junit = ROOT / args.junit
    _write_junit(junit, results)
    print(
        json.dumps(
            {
                "json": str(output),
                "junit": str(junit),
                "case_count": len(case_ids),
                "passed": payload["passed"],
                "failed": payload["failed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
