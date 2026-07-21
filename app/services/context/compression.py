from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.services.context.budget import canonical_json, estimate_tokens, stable_hash


PROTECTED_KEYS = {
    "goal", "user_goal", "target", "target_id", "target_ids", "pending_id", "pending_ids",
    "approval", "requires_approval", "policy", "policy_history", "last_error", "error_event_id",
    "executed_action_signatures", "citation", "citations", "evidence_id", "evidence_ids",
    "selected_skills", "skill_version", "stop_conditions", "missing_fields",
}


@dataclass(frozen=True)
class CompressionRecord:
    record_id: str
    lane: str
    source_refs: list[str]
    source_hash: str
    method: str
    summary: dict[str, Any]
    generation: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "lane": self.lane,
            "source_refs": self.source_refs,
            "source_hash": self.source_hash,
            "method": self.method,
            "summary": self.summary,
            "generation": self.generation,
            "created_at": self.created_at,
        }


def _record(lane: str, source: Any, source_refs: list[str], summary: dict[str, Any], method: str) -> CompressionRecord:
    return CompressionRecord(
        record_id=f"ctxcmp:{uuid.uuid4()}",
        lane=lane,
        source_refs=source_refs,
        source_hash=stable_hash(source),
        method=method,
        summary=summary,
        generation=1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def split_current_turn(history: list[dict[str, Any]], current_user_message: str) -> list[dict[str, str]]:
    clean = [
        {"role": str(item.get("role")), "content": str(item.get("content"))}
        for item in history
        if isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"} and item.get("content")
    ]
    if clean and clean[-1]["role"] == "user" and clean[-1]["content"] == current_user_message:
        clean.pop()
    return clean


def compress_history(history: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[CompressionRecord]]:
    recent_count = max(2, int(os.getenv("CONTEXT_RECENT_MESSAGE_COUNT", "6")))
    if len(history) <= recent_count:
        return history, []
    older, recent = history[:-recent_count], history[-recent_count:]
    turns = [
        {"role": item["role"], "content": item["content"][:240]}
        for item in older
        if item["role"] in {"user", "assistant"}
    ]
    summary = {"message_count": len(older), "turns": turns[-8:]}
    record = _record("history", older, [f"history:{index}" for index in range(len(older))], summary, "extractive_history_v1")
    return [{"role": "system", "content": "Earlier conversation summary: " + canonical_json(summary)}, *recent], [record]


def compress_observations(observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[CompressionRecord]]:
    recent_count = max(1, int(os.getenv("CONTEXT_RECENT_OBSERVATION_COUNT", "2")))
    if len(observations) <= recent_count:
        return observations, []
    older, recent = observations[:-recent_count], observations[-recent_count:]
    summary_items = []
    refs = []
    for index, item in enumerate(older):
        refs.append(f"observation:{index}")
        summary_items.append({
            key: item.get(key)
            for key in ("status", "action", "route_kind", "pending_id", "error_event_id", "evidence_ids", "citations")
            if item.get(key) not in (None, [], "")
        })
    summary = {"observation_count": len(older), "items": summary_items}
    record = _record("observations", older, refs, summary, "structured_observation_v1")
    return [{"compressed_observations": summary, "source_refs": refs}, *recent], [record]


def protected_view(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PROTECTED_KEYS:
                result[key] = item
            else:
                nested = protected_view(item)
                if nested not in ({}, [], None):
                    result[key] = nested
        return result
    if isinstance(value, list):
        return [item for item in (protected_view(entry) for entry in value) if item not in ({}, [], None)]
    return None


def trim_lane(value: Any, token_limit: int, *, lane: str) -> tuple[Any, list[CompressionRecord]]:
    if estimate_tokens(value) <= token_limit:
        return value, []
    if isinstance(value, list):
        selected = []
        for item in value:
            if estimate_tokens(selected + [item]) > token_limit:
                break
            selected.append(item)
        protected = protected_view(value)
        summary = {"kept": selected, "protected": protected, "omitted_count": max(0, len(value) - len(selected))}
    elif isinstance(value, dict):
        protected = protected_view(value)
        summary = {"protected": protected, "keys": sorted(value.keys())}
        for key in sorted(value):
            if key in PROTECTED_KEYS:
                continue
            candidate = {**summary, key: value[key]}
            if estimate_tokens(candidate) <= token_limit:
                summary[key] = value[key]
    else:
        text = str(value)
        summary = {"excerpt": text[: max(80, token_limit * 3)], "truncated": True}
    record = _record(lane, value, [f"{lane}:raw"], summary, "budget_trim_v1")
    return summary, [record]
