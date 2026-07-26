from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScientificRequestSpec:
    matched: bool
    capabilities: tuple[str, ...] = ()
    candidate_count: int | None = None
    source_policy: str = "use_available_sources"
    simulation_policy: str = "none"
    budget_profile: str = "default"
    seek_counterevidence: bool = False


_COUNT_WORDS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}


def parse_scientific_request(text: str, *, has_target: bool) -> ScientificRequestSpec:
    """Parse scientific capabilities without making a routing authorization decision."""
    normalized = " ".join(str(text or "").casefold().split())
    investigation = any(
        signal in normalized
        for signal in (
            "调查", "诊断", "分析", "研究", "investigate", "diagnose", "analyze",
        )
    )
    scientific_subject = has_target or any(
        signal in normalized
        for signal in (
            "生长", "培养", "藻", "数据集", "growth", "culture", "strain", "dataset",
        )
    )
    candidate = any(signal in normalized for signal in ("候选", "假设", "candidate", "hypothesis"))
    simulate = any(signal in normalized for signal in ("仿真", "模拟", "simulate", "simulation"))
    trend = any(signal in normalized for signal in ("趋势", "异常", "变慢", "trend", "anomaly", "slower"))
    multi_source = any(signal in normalized for signal in ("多源", "多个来源", "multi-source", "multiple sources"))
    matched = scientific_subject and (investigation or candidate or simulate or trend or multi_source)
    if not matched:
        return ScientificRequestSpec(matched=False)

    capabilities = ["investigate"]
    if candidate:
        capabilities.append("generate_candidates")
    if simulate:
        capabilities.append("simulate")
    if multi_source:
        capabilities.append("multi_source")
    seek_counterevidence = any(
        signal in normalized for signal in ("反证", "推翻", "counterevidence", "falsif")
    )
    if seek_counterevidence:
        capabilities.append("seek_counterevidence")

    count = None
    match = re.search(r"([1-9]|[一二两三四五])\s*个?\s*(?:候选|candidate)", normalized)
    if match:
        token = match.group(1)
        count = int(token) if token.isdigit() else _COUNT_WORDS.get(token)
    source_policy = (
        "continue_with_available_sources"
        if any(signal in normalized for signal in ("数据源失败", "来源失败", "source fails", "source failure"))
        else "use_available_sources"
    )
    return ScientificRequestSpec(
        matched=True,
        capabilities=tuple(capabilities),
        candidate_count=count,
        source_policy=source_policy,
        simulation_policy="required" if simulate else "none",
        budget_profile="low" if any(signal in normalized for signal in ("低预算", "low budget")) else "default",
        seek_counterevidence=seek_counterevidence,
    )
