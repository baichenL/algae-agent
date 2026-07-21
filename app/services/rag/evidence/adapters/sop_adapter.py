import re
from pathlib import Path

from app.models.rag_evidence_schema import EvidenceUnit
from app.services.rag.ingestion.loaders import load_source_units


def extract_sop_fact_rows(path: str | Path, metadata: dict, document_id: int) -> list[dict]:
    if metadata.get("doc_type") != "manual":
        return []
    source = Path(path)
    rows = []
    for unit in load_source_units(source, metadata):
        for row in _facts_from_text(source, unit, document_id):
            row.update(
                {
                    "version": metadata.get("version"),
                    "year": metadata.get("year"),
                    "language": metadata.get("language"),
                }
            )
            rows.append(row)
    return rows


def sop_rows_to_evidence_units(rows: list[dict]) -> list[EvidenceUnit]:
    return [sop_row_to_evidence_unit(row) for row in rows]


def sop_row_to_evidence_unit(row: dict) -> EvidenceUnit:
    location = {
        "section": row.get("section"),
        "page_number": row.get("page_number"),
    }
    evidence_id = (
        f"sop:{row.get('source_file')}:{row.get('page_number') or row.get('section')}:"
        f"{row.get('entity')}:{row.get('attribute')}:{abs(hash(row.get('value') or ''))}"
    )
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_type="manual",
        source_file=row["source_file"],
        fact_type="sop_fact",
        entity=row.get("entity"),
        attribute=row.get("attribute"),
        value=row.get("value"),
        text_span=row.get("text_span"),
        location={key: value for key, value in location.items() if value is not None},
        confidence=float(row.get("confidence", 1.0)),
        citation={
            "file_name": row.get("source_file"),
            "source_path": row.get("source_path"),
            "doc_type": "manual",
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            **{key: value for key, value in location.items() if value is not None},
        },
        metadata={
            "section": row.get("section"),
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            "task_type": row.get("attribute"),
            "equipment": row.get("equipment"),
            "medium": row.get("medium"),
            "organism": row.get("organism"),
        },
    )


def _facts_from_text(source: Path, unit, document_id: int) -> list[dict]:
    text = unit.content or ""
    facts = []
    fact_specs = [
        ("tap_medium", "operation", "TAP 液体培养基配�", r"1\.2\.4\s*TAP|TAP 液体培养�"),
        ("hygromycin_tap_plate", "material_used", "潮霉�?TAP 平板用于转化后筛�", r"潮霉素\s*TAP\s*平板|hygromycin\s*TAP"),
        ("sucrose_tap_medium", "material_used", "40 mM 蔗糖 TAP 培养基用于电击转化相关操�", r"蔗糖\s*TAP\s*培养基|40\s*mM.*TAP"),
        ("electroporation", "parameter", "莱茵衣藻培养�?OD750=0.3-0.5 后用于电击转化材料准�", r"OD750\s*=\s*0\.3-0\.5"),
        (
            "recovery_culture",
            "procedure_step",
            "使用蔗糖 TAP 溶液重悬或进行恢复培�",
            r"蔗糖\s*TAP\s*溶液重悬|TAP.{0,100}恢复培养|恢复培养.{0,100}TAP",
        ),
    ]
    for entity, attribute, value, pattern in fact_specs:
        if re.search(pattern, text, re.I):
            facts.append(_row(source, unit, document_id, entity, attribute, value, _snippet(text, pattern)))
    return facts


def _row(source: Path, unit, document_id: int, entity: str, attribute: str, value: str, text_span: str) -> dict:
    return {
        "document_id": document_id,
        "source_file": source.name,
        "source_path": str(source),
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "text_span": text_span,
        "section": unit.section,
        "page_number": unit.page_number,
        "confidence": 0.9,
    }


def _snippet(text: str, pattern: str, max_len: int = 180) -> str:
    match = re.search(pattern, text, re.I)
    if not match:
        return text[:max_len]
    start = max(match.start() - 60, 0)
    end = min(match.end() + 100, len(text))
    return re.sub(r"\s+", " ", text[start:end]).strip()[:max_len]
