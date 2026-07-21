import sqlite3

from app.core import database
from app.core.db import schema
from app.models.rag_evidence_schema import (
    AnswerabilityResult,
    EvidenceUnit,
    GroundedAnswer,
    GroundedCitation,
    GroundedClaim,
)
from app.services.rag.evidence.adapters.chunk_adapter import chunk_to_evidence_unit
from app.services.rag.evidence.citation_validator import validate_grounded_answer


def test_evidence_unit_legacy_constructor_populates_modern_fields():
    unit = EvidenceUnit(
        evidence_id="legacy:1",
        source_type="manual",
        source_file="manual.pdf",
        fact_type="sop_fact",
        text_span="Use microscopy for contamination checks.",
        location={"page_number": 7, "section": "Quality checks"},
        citation={"version": "v1", "source_path": "manual.pdf"},
    )

    assert unit.evidence_type == "sop_fact"
    assert unit.content == "Use microscopy for contamination checks."
    assert unit.content_hash
    assert unit.document_version == "v1"
    assert unit.page_number == 7
    assert unit.section_path == ["Quality checks"]
    assert unit.source_locator["page_number"] == 7


def test_evidence_unit_content_hash_is_stable():
    kwargs = {
        "evidence_id": "stable:1",
        "source_type": "manual",
        "source_file": "manual.txt",
        "fact_type": "text_chunk",
        "text_span": "Stable content.",
        "location": {"section": "A"},
        "metadata": {"version": "v1"},
    }

    first = EvidenceUnit(**kwargs)
    second = EvidenceUnit(**{**kwargs, "evidence_id": "stable:2"})

    assert first.content_hash == second.content_hash


def test_rag_evidence_schema_migration_is_idempotent(isolated_sqlite_db):
    schema.init_db()
    schema.init_db()

    with sqlite3.connect(isolated_sqlite_db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(rag_evidence_units)").fetchall()}
        element_columns = {row[1] for row in conn.execute("PRAGMA table_info(rag_document_elements)").fetchall()}

    assert {"content", "content_hash", "section_path", "source_locator", "element_type"} <= columns
    assert {"element_id", "document_id", "section_path", "source_locator"} <= element_columns


def test_rag_evidence_units_round_trip_modern_fields(isolated_sqlite_db):
    unit = EvidenceUnit(
        evidence_id="chunk:abc",
        document_id=42,
        document_version="v2",
        source_type="manual",
        source_file="manual.txt",
        fact_type="text_chunk",
        content="Manual content",
        text_span="Manual content",
        location={"section": "S1", "chunk_index": 0},
        section_path=["Manual", "S1"],
        parent_id="parent:1",
        element_type="paragraph",
        source_locator={"source_path": "manual.txt", "section": "S1"},
    )

    database.replace_rag_evidence_units(42, "source:abc", [unit])
    rows = database.list_rag_evidence_units()

    assert rows[0]["content"] == "Manual content"
    assert rows[0]["content_hash"] == unit.content_hash
    assert rows[0]["section_path"] == ["Manual", "S1"]
    assert rows[0]["source_locator"]["section"] == "S1"
    assert rows[0]["parent_id"] == "parent:1"
    assert rows[0]["element_type"] == "paragraph"


def test_chunk_adapter_fills_source_locator_and_parent_fields():
    unit = chunk_to_evidence_unit(
        {
            "chunk_id": "abc",
            "document_id": 10,
            "doc_type": "manual",
            "file_name": "manual.pdf",
            "source_path": "data/manual.pdf",
            "title": "Manual",
            "section": "Section A",
            "page_number": 3,
            "chunk_index": 4,
            "content": "A source-backed paragraph.",
            "metadata": {"parent_id": "parent:x", "version": "v1"},
        }
    )

    assert unit.document_id == 10
    assert unit.parent_id == "parent:x"
    assert unit.page_number == 3
    assert unit.section_path == ["Section A"]
    assert unit.source_locator["page_number"] == 3


def test_citation_validator_passes_grounded_answer():
    evidence = EvidenceUnit(
        evidence_id="e1",
        source_type="manual",
        source_file="manual.txt",
        fact_type="sop_fact",
        text_span="Check microscopy.",
        location={"section": "SOP"},
    )
    answer = GroundedAnswer(
        answerability=AnswerabilityResult(status="answered", reason="ok"),
        direct_answer="Check microscopy.",
        evidence=[evidence],
        citations=[GroundedCitation(source_id=1, evidence_id="e1", source_file="manual.txt", source_type="manual", location={"section": "SOP"})],
        claims=[GroundedClaim(claim_type="fact", text="Check microscopy.", evidence_ids=["e1"])],
    )

    result = validate_grounded_answer(answer)

    assert result.valid
    assert result.issues == []


def test_citation_validator_reports_missing_and_mismatched_citations():
    evidence = EvidenceUnit(
        evidence_id="e1",
        source_type="manual",
        source_file="manual.txt",
        fact_type="sop_fact",
        text_span="Check microscopy.",
        location={"page_number": 1},
    )
    answer = GroundedAnswer(
        answerability=AnswerabilityResult(status="answered", reason="ok"),
        direct_answer="Check microscopy.",
        evidence=[evidence],
        citations=[GroundedCitation(source_id=1, evidence_id="missing", source_file="manual.txt", source_type="manual", location={"page_number": 1})],
        claims=[GroundedClaim(claim_type="fact", text="Check microscopy.", evidence_ids=["e1"])],
    )

    result = validate_grounded_answer(answer)

    assert not result.valid
    codes = {issue.code for issue in result.issues}
    assert "citation_evidence_missing" in codes
    assert "claim_citation_missing" in codes


def test_citation_validator_allows_blocked_without_citations():
    answer = GroundedAnswer(
        answerability=AnswerabilityResult(status="blocked", reason="rag_read_only_boundary"),
        direct_answer="Blocked.",
        evidence=[],
        citations=[],
        claims=[],
    )

    assert validate_grounded_answer(answer).valid
