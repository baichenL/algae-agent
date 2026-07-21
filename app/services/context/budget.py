from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else canonical_json(value)
    cjk = len(_CJK_RE.findall(text))
    non_cjk = max(0, len(text) - cjk)
    return cjk + math.ceil(non_cjk / 4) + 4


def stable_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def input_token_budget() -> int:
    return max(1000, int(os.getenv("CONTEXT_INPUT_TOKEN_BUDGET", "16000")))


def soft_budget_ratio() -> float:
    return min(max(float(os.getenv("CONTEXT_SOFT_BUDGET_RATIO", "0.60")), 0.1), 0.95)


def hard_budget_ratio() -> float:
    return min(max(float(os.getenv("CONTEXT_HARD_BUDGET_RATIO", "0.80")), soft_budget_ratio()), 1.0)


@dataclass(frozen=True)
class ContextBudgetReport:
    token_budget: int
    soft_limit: int
    hard_limit: int
    estimated_input_tokens: int
    lane_tokens: dict[str, int] = field(default_factory=dict)
    stable_prefix_tokens: int = 0
    prefix_hash: str = ""
    compression_applied: bool = False
    protected_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "soft_limit": self.soft_limit,
            "hard_limit": self.hard_limit,
            "estimated_input_tokens": self.estimated_input_tokens,
            "lane_tokens": dict(sorted(self.lane_tokens.items())),
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "prefix_hash": self.prefix_hash,
            "compression_applied": self.compression_applied,
            "protected_tokens": self.protected_tokens,
        }


class ContextBudgetExceeded(RuntimeError):
    def __init__(self, report: ContextBudgetReport):
        super().__init__("context_budget_exceeded")
        self.report = report
