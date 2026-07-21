from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.rag_evidence_schema import stable_json_dumps


DocumentElementType = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "table_row",
    "figure_caption",
    "formula",
    "unknown",
]


class DocumentElement(BaseModel):
    element_id: str
    document_id: str
    element_type: DocumentElementType = "unknown"
    content: str
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    source_locator: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str


class NormalizedDocument(BaseModel):
    document_id: str
    document_version: str
    source_path: str
    source_type: str = "local_file"
    title: str | None = None
    elements: list[DocumentElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str
    parser_version: str | None = None
    content_hash: str


def stable_document_id(source_path: str) -> str:
    return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:24]


def stable_element_id(
    *,
    document_id: str,
    element_type: str,
    index: int,
    content: str,
    source_locator: dict[str, Any],
) -> str:
    payload = {
        "document_id": document_id,
        "element_type": element_type,
        "index": index,
        "content": content,
        "source_locator": source_locator,
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:24]


def stable_content_hash(content: str, source_locator: dict[str, Any] | None = None) -> str:
    payload = {"content": content or "", "source_locator": source_locator or {}}
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
