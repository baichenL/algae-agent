from __future__ import annotations

import os

from app.core.database import list_rag_chunks_for_semantic


MAX_PARENT_CONTEXT_CHARS = 4000


def parent_expansion_enabled() -> bool:
    return os.getenv("RAG_PARENT_EXPANSION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def expand_parent_context(chunks: list[dict], max_chars: int = MAX_PARENT_CONTEXT_CHARS) -> list[dict]:
    if not parent_expansion_enabled():
        return [
            {
                **chunk,
                "parent_expansion": {
                    "enabled": False,
                    "expanded": False,
                    "expansion_reason": "feature_flag_disabled",
                },
            }
            for chunk in chunks
        ]
    parent_ids = {
        (chunk.get("metadata") or {}).get("parent_id")
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("parent_id")
    }
    if not parent_ids:
        return [
            {
                **chunk,
                "parent_expansion": {
                    "enabled": True,
                    "expanded": False,
                    "expansion_reason": "no_parent_id",
                },
            }
            for chunk in chunks
        ]
    doc_types = sorted({chunk.get("doc_type") for chunk in chunks if chunk.get("doc_type")})
    all_chunks = list_rag_chunks_for_semantic(doc_types=doc_types or None, limit=5000)
    siblings_by_parent: dict[str, list[dict]] = {}
    for item in all_chunks:
        parent_id = (item.get("metadata") or {}).get("parent_id")
        if parent_id in parent_ids:
            siblings_by_parent.setdefault(parent_id, []).append(item)

    enriched = []
    expanded_parents = set()
    for chunk in chunks:
        parent_id = (chunk.get("metadata") or {}).get("parent_id")
        siblings = siblings_by_parent.get(parent_id) or []
        if not parent_id or not siblings:
            enriched.append(
                {
                    **chunk,
                    "parent_expansion": {
                        "enabled": True,
                        "expanded": False,
                        "expansion_reason": "parent_not_found",
                    },
                }
            )
            continue
        siblings.sort(key=lambda item: int(item.get("chunk_index") or 0))
        context = "\n\n".join(str(item.get("content") or "") for item in siblings)[:max_chars]
        enriched.append(
            {
                **chunk,
                "answer_context": context,
                "parent_expansion": {
                    "enabled": True,
                    "expanded": parent_id not in expanded_parents,
                    "original_hit_id": chunk.get("chunk_id"),
                    "expanded_parent_id": parent_id,
                    "expansion_reason": "same_parent_sibling_context",
                    "child_content": chunk.get("content"),
                    "parent_content": context,
                },
            }
        )
        expanded_parents.add(parent_id)
    return enriched
