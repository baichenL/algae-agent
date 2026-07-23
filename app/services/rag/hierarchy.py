from __future__ import annotations

import hashlib


def build_hierarchy_nodes(chunks: list[dict], *, generation_id: str, document_id: int) -> list[dict]:
    """Create the lightweight L0 document / L1 section summary index."""
    if not chunks:
        return []
    root_id = _node_id(generation_id, document_id, "L0", "document")
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        key = str(metadata.get("parent_id") or chunk.get("section") or "document")
        groups.setdefault(key, []).append(chunk)
    nodes = [
        {
            "node_id": root_id, "level": 0, "parent_node_id": None,
            "title": chunks[0].get("title") or chunks[0].get("file_name"),
            "summary": _summary([item.get("content") for item in chunks], 900),
            "metadata": {"node_type": "document_summary", "generation_id": generation_id},
        }
    ]
    for key, items in groups.items():
        nodes.append(
            {
                "node_id": _node_id(generation_id, document_id, "L1", key), "level": 1,
                "parent_node_id": root_id,
                "title": items[0].get("section") or items[0].get("title") or key,
                "summary": _summary([item.get("content") for item in items], 600),
                "metadata": {
                    "node_type": "section_summary", "generation_id": generation_id,
                    "child_chunk_ids": [item.get("chunk_id") for item in items],
                },
            }
        )
    return nodes


def _node_id(generation_id: str, document_id: int, level: str, key: str) -> str:
    return hashlib.sha256(f"{generation_id}:{document_id}:{level}:{key}".encode("utf-8")).hexdigest()[:24]


def _summary(parts, limit: int) -> str:
    text = " ".join(" ".join(str(part or "").split()) for part in parts if str(part or "").strip())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
