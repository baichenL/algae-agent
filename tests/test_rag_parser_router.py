from pathlib import Path

import pytest

from app.services.rag.ingestion import pdf_router
from app.services.rag.ingestion.document_models import (
    DocumentElement,
    NormalizedDocument,
    stable_content_hash,
    stable_document_id,
    stable_element_id,
)


def _metadata(path: Path) -> dict:
    return {
        "source_path": str(path),
        "file_name": path.name,
        "doc_type": "manual",
        "topic": "router_test",
        "content_hash": "hash",
    }


def _document(path: Path, parser_name: str, content: str = "Parsed PDF text.") -> NormalizedDocument:
    metadata = _metadata(path)
    document_id = stable_document_id(metadata["source_path"])
    locator = {"source_path": str(path), "file_name": path.name, "page_number": 1}
    element = DocumentElement(
        element_id=stable_element_id(
            document_id=document_id,
            element_type="paragraph",
            index=0,
            content=content,
            source_locator=locator,
        ),
        document_id=document_id,
        element_type="paragraph",
        content=content,
        page_number=1,
        section_path=["router_test"],
        source_locator=locator,
        metadata={},
        content_hash=stable_content_hash(content, locator),
    )
    return NormalizedDocument(
        document_id=document_id,
        document_version="hash",
        source_path=str(path),
        title="router_test",
        elements=[element],
        metadata=metadata,
        parser_name=parser_name,
        parser_version="test",
        content_hash="hash",
    )


class _FakeAdapter:
    version = "test"

    def __init__(self, name: str, *, fail: bool = False):
        self.name = name
        self.fail = fail

    def parse(self, source, metadata, probe):
        if self.fail:
            raise RuntimeError(f"{self.name} unavailable")
        return _document(source, self.name)


def test_parser_policy_uses_mineru_for_scanned_or_garbled_pdf():
    probe = pdf_router.PdfProbeReport(status="ok", page_count=2, is_scanned_or_low_text=True)
    assert pdf_router.parser_candidates_for_probe(probe) == ["mineru", "docling"]

    garbled = pdf_router.PdfProbeReport(status="ok", page_count=2, is_garbled=True)
    assert pdf_router.parser_candidates_for_probe(garbled) == ["mineru", "docling"]


def test_complex_pdf_policy_uses_pypdf_only_as_salvage():
    probe = pdf_router.PdfProbeReport(status="ok", page_count=2, has_table_signal=True)
    assert pdf_router.parser_candidates_for_probe(probe) == ["docling", "mineru", "pypdf_salvage"]


def test_degraded_result_is_provisional_until_candidates_exhausted(tmp_path, monkeypatch):
    path = tmp_path / "manual__router__v1__en.pdf"
    path.write_bytes(b"%PDF")
    probe = pdf_router.PdfProbeReport(status="ok", page_count=1, text_char_count=100, has_table_signal=True)
    monkeypatch.setattr(pdf_router, "probe_pdf", lambda source: probe)
    monkeypatch.setattr(pdf_router, "parser_candidates_for_probe", lambda report: ["pypdf", "docling"])
    monkeypatch.setattr(
        pdf_router,
        "_adapter_for",
        lambda name: _FakeAdapter(name, fail=name == "docling"),
    )

    document = pdf_router.parse_pdf_document(path, _metadata(path))

    report = document.metadata["quality_report"]
    assert document.parser_name == "pypdf"
    assert report["status"] == "degraded"
    assert report["decision"] == "accept"
    assert "table_signal_without_table_elements" in report["degraded_flags"]
    assert document.metadata["parser_attempts"][0]["decision"] == "accept"
    assert document.metadata["parser_attempts"][1]["status"] == "failed"


def test_strict_mode_rejects_only_degraded_provisional_result(tmp_path, monkeypatch):
    path = tmp_path / "manual__router_strict__v1__en.pdf"
    path.write_bytes(b"%PDF")
    probe = pdf_router.PdfProbeReport(status="ok", page_count=1, text_char_count=100, has_table_signal=True)
    monkeypatch.setenv("RAG_STRICT_PRODUCTION", "true")
    monkeypatch.setattr(pdf_router, "probe_pdf", lambda source: probe)
    monkeypatch.setattr(pdf_router, "parser_candidates_for_probe", lambda report: ["pypdf"])
    monkeypatch.setattr(pdf_router, "_adapter_for", lambda name: _FakeAdapter(name))

    with pytest.raises(RuntimeError, match="degraded provisional"):
        pdf_router.parse_pdf_document(path, _metadata(path))


def test_pypdf_salvage_is_marked_unreliable_and_rejected_in_strict_mode(tmp_path, monkeypatch):
    path = tmp_path / "manual__router_salvage__v1__en.pdf"
    path.write_bytes(b"%PDF")
    probe = pdf_router.PdfProbeReport(status="ok", page_count=1, text_char_count=100)
    monkeypatch.setenv("RAG_STRICT_PRODUCTION", "true")
    monkeypatch.setattr(pdf_router, "probe_pdf", lambda source: probe)
    monkeypatch.setattr(pdf_router, "parser_candidates_for_probe", lambda report: ["pypdf_salvage"])
    monkeypatch.setattr(pdf_router, "_adapter_for", lambda name: _FakeAdapter(name))

    with pytest.raises(RuntimeError, match="degraded provisional"):
        pdf_router.parse_pdf_document(path, _metadata(path))


def test_mineru_adapter_consumes_structured_json_sidecar(tmp_path):
    path = tmp_path / "manual__mineru__v1__en.pdf"
    path.write_bytes(b"%PDF")
    sidecar = Path(str(path) + ".mineru.json")
    sidecar.write_text(
        """
        {
          "pages": [
            {"page": 1, "blocks": [
              {"type": "title", "text": "Main title", "bbox": [1,2,3,4]},
              {"type": "table", "text": "A | B\\n1 | 2", "bbox": [2,3,4,5]}
            ]}
          ]
        }
        """,
        encoding="utf-8",
    )

    document = pdf_router.MinerUAdapter().parse(
        path,
        _metadata(path),
        pdf_router.PdfProbeReport(status="ok", page_count=1, text_char_count=20),
    )

    assert document.parser_name == "mineru"
    assert [item.element_type for item in document.elements] == ["heading", "table"]
    assert document.elements[0].source_locator["bbox"] == [1, 2, 3, 4]
