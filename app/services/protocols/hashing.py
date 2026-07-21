from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any


HASH_EXCLUDED_FIELDS = {
    "protocol_hash",
    "protocol_id",
    "created_at",
    "created_by",
    "source_run_id",
    "source_session_id",
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return value


def _normalize(value: Any) -> Any:
    value = _to_plain(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("protocol_contains_non_finite_float")
        return int(value) if value.is_integer() else round(value, 10)
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in HASH_EXCLUDED_FIELDS
        }
    raise TypeError(f"protocol_contains_non_json_value:{type(value).__name__}")


def canonicalize_protocol(protocol: Any) -> dict[str, Any]:
    normalized = _normalize(protocol)
    if not isinstance(normalized, dict):
        raise TypeError("protocol_must_canonicalize_to_object")
    return normalized


def canonical_protocol_json(protocol: Any) -> str:
    return json.dumps(
        canonicalize_protocol(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_protocol_hash(protocol: Any) -> str:
    canonical = canonical_protocol_json(protocol)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_protocol_hash(protocol: Any, expected_hash: str | None = None) -> bool:
    payload = _to_plain(protocol)
    embedded_hash = None
    if isinstance(payload, dict):
        embedded_hash = payload.get("protocol_hash")
    else:
        embedded_hash = getattr(protocol, "protocol_hash", None)
    target_hash = expected_hash or embedded_hash
    if not target_hash:
        return False
    return compute_protocol_hash(protocol) == target_hash
