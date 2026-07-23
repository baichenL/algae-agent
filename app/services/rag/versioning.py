from __future__ import annotations

from datetime import datetime, timezone

from app.core.database import list_rag_knowledge_sources


def find_version_conflicts(
    *, doc_types: list[str] | None = None, source_ids: list[str] | None = None,
    version_policy: str = "current", as_of: datetime | None = None,
) -> list[dict]:
    if version_policy == "all":
        return []
    instant = as_of or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    grouped: dict[str, list[dict]] = {}
    for source in list_rag_knowledge_sources():
        if source.get("lifecycle_status", "active") != "active" or source.get("review_status", "approved") != "approved":
            continue
        if doc_types and source.get("doc_type") not in doc_types:
            continue
        if source_ids and source.get("source_id") not in source_ids:
            continue
        if not _effective(source, instant):
            continue
        grouped.setdefault(str(source.get("asset_key") or source.get("source_id")), []).append(source)
    return [
        {"asset_key": asset_key, "source_ids": [item["source_id"] for item in items],
         "versions": [item.get("version") for item in items]}
        for asset_key, items in grouped.items() if len(items) > 1
    ]


def _effective(source: dict, instant: datetime) -> bool:
    start = _parse(source.get("effective_from"))
    end = _parse(source.get("effective_to"))
    return not ((start and instant < start) or (end and instant >= end))


def _parse(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
