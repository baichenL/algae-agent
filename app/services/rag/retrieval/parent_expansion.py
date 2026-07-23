from __future__ import annotations

import os

from app.core.database import list_rag_chunks_for_semantic


MAX_PARENT_CONTEXT_TOKENS = 1800


def parent_expansion_enabled() -> bool:
    return os.getenv("RAG_PARENT_EXPANSION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def expand_parent_context(chunks: list[dict], max_tokens: int = MAX_PARENT_CONTEXT_TOKENS) -> list[dict]:
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
        context, included_ids = _centered_context(siblings, chunk, max_tokens=max_tokens)
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
                    "included_chunk_ids": included_ids,
                    "max_tokens": max_tokens,
                },
            }
        )
        expanded_parents.add(parent_id)
    return enriched


def _centered_context(siblings: list[dict], hit: dict, *, max_tokens: int) -> tuple[str, list[str]]:
    hit_id = hit.get("chunk_id")
    hit_index = next((index for index, item in enumerate(siblings) if item.get("chunk_id") == hit_id), 0)
    selected = [siblings[hit_index]]
    used = _token_count(siblings[hit_index].get("content"))
    left = hit_index - 1
    right = hit_index + 1
    while left >= 0 or right < len(siblings):
        progressed = False
        if left >= 0:
            count = _token_count(siblings[left].get("content"))
            if used + count <= max_tokens:
                selected.insert(0, siblings[left])
                used += count
                progressed = True
            left = -1 if used + count > max_tokens else left - 1
        if right < len(siblings):
            count = _token_count(siblings[right].get("content"))
            if used + count <= max_tokens:
                selected.append(siblings[right])
                used += count
                progressed = True
            right = len(siblings) if used + count > max_tokens else right + 1
        if not progressed:
            break
    context = "\n\n".join(str(item.get("content") or "") for item in selected)
    if str(hit.get("content") or "") not in context:
        context = str(hit.get("content") or "")
        selected = [hit]
    return context, [str(item.get("chunk_id") or "") for item in selected]


def _token_count(value: object) -> int:
    text = str(value or "")
    # Stable dependency-free estimate: CJK characters and word-like spans count as tokens.
    import re

    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|\S", text))
