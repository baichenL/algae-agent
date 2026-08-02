from __future__ import annotations

import html
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def report_directory() -> Path:
    return Path(os.getenv("ALGAE_EVAL_REPORT_DIR") or "data/eval_reports").resolve()


def metric_point(
    *,
    key: str,
    display_name: str,
    value: float | int | None,
    unit: str,
    standard: str,
    reference_url: str,
    evidence_mode: str = "deterministic",
    numerator: int | float | None = None,
    denominator: int | float | None = None,
    ci95: tuple[float, float] | list[float] | None = None,
    project_target: float | None = None,
    status: str | None = None,
    definition: str = "",
) -> dict[str, Any]:
    numeric_value = float(value) if value is not None and math.isfinite(float(value)) else None
    resolved_status = status or ("completed" if numeric_value is not None else "not_run")
    point: dict[str, Any] = {
        "key": key,
        "display_name": display_name,
        "value": numeric_value,
        "unit": unit,
        "standard": standard,
        "reference_url": reference_url,
        "evidence_mode": evidence_mode,
        "status": resolved_status,
        "definition": definition,
    }
    if numerator is not None:
        point["numerator"] = numerator
    if denominator is not None:
        point["denominator"] = denominator
    if ci95 is not None:
        point["ci95"] = [float(ci95[0]), float(ci95[1])]
    if project_target is not None:
        point["project_target"] = project_target
    return point


def not_run_metric(
    *,
    key: str,
    display_name: str,
    unit: str,
    standard: str,
    reference_url: str,
    evidence_mode: str,
    reason: str,
    definition: str = "",
) -> dict[str, Any]:
    point = metric_point(
        key=key,
        display_name=display_name,
        value=None,
        unit=unit,
        standard=standard,
        reference_url=reference_url,
        evidence_mode=evidence_mode,
        status="not_run",
        definition=definition,
    )
    point["status_reason"] = reason
    return point


def save_report_bundle(report: dict[str, Any], output_dir: Path | None = None) -> dict[str, Path]:
    destination = (output_dir or report_directory()).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_id = validate_report_id(str(report["report_id"]))
    json_path = destination / f"{report_id}.json"
    html_path = destination / f"{report_id}.html"
    markdown_path = destination / f"{report_id}.md"
    latest_path = destination / "latest.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    json_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    html_path.write_text(render_self_contained_html(report), encoding="utf-8")
    markdown_path.write_text(render_markdown_summary(report), encoding="utf-8")
    (destination / "latest.md").write_text(render_markdown_summary(report), encoding="utf-8")
    (destination / "latest.html").write_text(render_self_contained_html(report), encoding="utf-8")
    return {
        "json": json_path,
        "html": html_path,
        "markdown": markdown_path,
        "latest": latest_path,
    }


def validate_report_id(report_id: str) -> str:
    if not REPORT_ID_PATTERN.fullmatch(report_id):
        raise ValueError("invalid evaluation report id")
    return report_id


def load_report(report_id: str) -> dict[str, Any] | None:
    path = report_directory() / f"{validate_report_id(report_id)}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    latest = load_latest_report()
    if latest and latest.get("report_id") == report_id:
        return latest
    return None


def load_latest_report() -> dict[str, Any] | None:
    path = report_directory() / "latest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


SENSITIVE_REPORT_KEYS = {
    "api_key",
    "authorization",
    "system_prompt",
    "system_instruction",
    "raw_source",
    "source_content",
    "provider_secret",
}


def sanitize_report(value: Any) -> Any:
    """Defense-in-depth redaction for read-only report APIs and downloads."""
    if isinstance(value, dict):
        return {
            key: sanitize_report(item)
            for key, item in value.items()
            if key.casefold() not in SENSITIVE_REPORT_KEYS
        }
    if isinstance(value, list):
        return [sanitize_report(item) for item in value]
    return value


def _format_value(metric: dict[str, Any]) -> str:
    if metric.get("status") != "completed" or metric.get("value") is None:
        return "NOT RUN"
    value = float(metric["value"])
    unit = metric.get("unit")
    if unit == "ratio":
        rendered = f"{value * 100:.1f}%"
    elif unit == "milliseconds":
        rendered = f"{value:.1f} ms"
    elif unit == "tokens":
        rendered = f"{value:,.0f}"
    else:
        rendered = f"{value:.4g}"
    ci = metric.get("ci95")
    if ci:
        if unit == "ratio":
            rendered += f" (95% CI {ci[0] * 100:.1f}–{ci[1] * 100:.1f}%)"
        else:
            rendered += f" (95% CI {ci[0]:.4g}–{ci[1]:.4g})"
    return rendered


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# 自动生成的评估摘要",
        "",
        f"- Schema: `{report.get('schema_version')}`",
        "- Source: `python scripts/run_eval_showcase.py`",
        "",
        "| Suite | Metric | Result | n | Evidence | Standard |",
        "|---|---|---:|---:|---|---|",
    ]
    for suite in report.get("suites") or []:
        for metric in suite.get("metrics") or []:
            denominator = metric.get("denominator", "—")
            link = metric.get("reference_url") or ""
            standard = f"[{metric.get('standard')}]({link})" if link else metric.get("standard", "")
            lines.append(
                f"| {suite.get('display_name', suite.get('key'))} | {metric.get('display_name')} "
                f"| {_format_value(metric)} | {denominator} | {metric.get('evidence_mode')} | {standard} |"
            )
    lines.extend(
        [
            "",
            "> 本文件由 `scripts/run_eval_showcase.py` 生成；请勿手工修改分数或样本量。",
            "",
        ]
    )
    return "\n".join(lines)


def render_self_contained_html(report: dict[str, Any]) -> str:
    metadata = html.escape(
        f"{report.get('generated_at')} · commit {report.get('git_commit') or 'unknown'}"
    )
    suite_sections = []
    for suite in report.get("suites") or []:
        cards = []
        for metric in suite.get("metrics") or []:
            status = metric.get("status", "completed")
            sample = html.escape(str(metric.get("denominator", "—")))
            standard = html.escape(str(metric.get("standard") or ""))
            reference = html.escape(str(metric.get("reference_url") or ""), quote=True)
            detail = html.escape(str(metric.get("definition") or ""))
            reason = html.escape(str(metric.get("status_reason") or ""))
            cards.append(
                "<article class='metric'>"
                f"<div class='badge {status}'>{html.escape(status.upper())}</div>"
                f"<h3>{html.escape(str(metric.get('display_name')))}</h3>"
                f"<div class='value'>{html.escape(_format_value(metric))}</div>"
                f"<p>n = {sample} · {html.escape(str(metric.get('evidence_mode')))}</p>"
                f"<p>{detail}</p>"
                f"<p class='reason'>{reason}</p>"
                f"<a href='{reference}'>{standard}</a>"
                "</article>"
            )
        suite_sections.append(
            f"<section><h2>{html.escape(str(suite.get('display_name', suite.get('key'))))}</h2>"
            f"<p>{html.escape(str(suite.get('description') or ''))}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    comparison_sections = []
    for comparison in report.get("comparisons") or []:
        variants = comparison.get("variants") or []
        if isinstance(variants, dict):
            variants = [{"name": key, **(value or {})} for key, value in variants.items()]
        rows = []
        numeric_values = []
        for variant in variants:
            value = variant.get("value")
            if value is None:
                value = variant.get("tokens_mean")
            if value is None:
                continue
            numeric_values.append(abs(float(value)))
        maximum = max(numeric_values, default=1.0) or 1.0
        for variant in variants:
            value = variant.get("value")
            if value is None:
                value = variant.get("tokens_mean")
            if value is None:
                continue
            width = min(abs(float(value)) / maximum * 100.0, 100.0)
            rows.append(
                "<div class='bar-row'>"
                f"<span>{html.escape(str(variant.get('name')))}</span>"
                f"<div class='bar-track'><i style='width:{width:.2f}%'></i></div>"
                f"<strong>{float(value):.4g}</strong>"
                "</div>"
            )
        if rows:
            comparison_sections.append(
                f"<section><h2>{html.escape(str(comparison.get('display_name') or comparison.get('key')))}</h2>"
                f"<div class='comparison'>{''.join(rows)}</div></section>"
            )
    payload = html.escape(json.dumps(report, ensure_ascii=False), quote=False)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Algae Agent Evaluation · {html.escape(str(report.get('report_id')))}</title>
<style>
:root{{--ink:#17211f;--muted:#60706c;--brand:#0f766e;--line:#dce5e2;--paper:#fff;--bg:#f4f7f6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,sans-serif}}
main{{max-width:1240px;margin:auto;padding:36px}}header{{background:#123d39;color:white;padding:30px;border-radius:18px}}
h1{{margin:0 0 6px;font-size:34px}}h2{{margin:34px 0 4px}}.meta{{opacity:.82}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.metric{{position:relative;background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:18px;break-inside:avoid}}
.metric h3{{margin:6px 0}}.metric p{{color:var(--muted);min-height:1.4em}}.value{{font-size:23px;font-weight:750}}
.badge{{display:inline-block;font-size:10px;font-weight:800;padding:3px 7px;border-radius:999px;background:#dcfce7;color:#166534}}
.badge.not_run{{background:#f1f5f9;color:#475569}}a{{color:var(--brand)}}.reason{{color:#92400e!important}}
.bar-row{{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 80px;gap:12px;align-items:center;margin:10px 0}}
.bar-track{{height:16px;background:#e7eeec;border-radius:999px;overflow:hidden}}.bar-track i{{display:block;height:100%;background:var(--brand)}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}header{{border-radius:0}}.metric{{box-shadow:none}}a{{color:inherit;text-decoration:none}}}}
</style></head><body><main><header><h1>标准化评估中心</h1><div class="meta">{metadata}</div>
<div class="meta">数据集：{html.escape(json.dumps(report.get('dataset_versions') or {}, ensure_ascii=False))}</div></header>
{''.join(suite_sections)}{''.join(comparison_sections)}</main><script id="evaluation-report" type="application/json">{payload}</script></body></html>"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
