import re
from pathlib import Path

from app.models.rag_evidence_schema import EvidenceUnit
from app.services.rag.ingestion.loaders import load_source_units


TAG_PATTERNS = [
    ("machine_learning", r"machine learning|\bML\b|artificial intelligence|\bAI\b"),
    ("deep_learning", r"deep learning|\bDL\b|neural network|ANN|CNN|LSTM"),
    ("data_driven_modeling", r"data[- ]driven|black box|hybrid model"),
    ("growth_prediction", r"growth prediction|growth curve|forecast|forecasting|time series"),
    ("cultivation_optimization", r"optimization|environmental factors|cultivation"),
    ("od750_prediction", r"OD750|OD 750|optical density"),
    (
        "project_parameter",
        r"directly applicable to this project|project-specific (?:culture|cultivation) parameter|本项目.*培养参数|可直接用于",
    ),
]


def extract_paper_fact_rows(path: str | Path, metadata: dict, document_id: int) -> list[dict]:
    if metadata.get("doc_type") != "paper":
        return []
    source = Path(path)
    title = _paper_title(source, metadata)
    rows = []
    topic_text = f"{source.stem} {metadata.get('topic') or ''}".replace("_", " ")
    for attribute, pattern in TAG_PATTERNS:
        if re.search(pattern, topic_text, re.I):
            rows.append(_row(source, metadata, document_id, title, attribute, attribute, topic_text, None, 0.85))

    for unit in load_source_units(source, metadata)[:20]:
        text = unit.content or ""
        for attribute, pattern in TAG_PATTERNS:
            if re.search(pattern, text, re.I):
                rows.append(
                    _row(source, metadata, document_id, title, attribute, attribute, _snippet(text, pattern), unit.page_number, 0.75)
                )
    return _dedupe(rows)


def paper_rows_to_evidence_units(rows: list[dict]) -> list[EvidenceUnit]:
    return [paper_row_to_evidence_unit(row) for row in rows]


def paper_row_to_evidence_unit(row: dict) -> EvidenceUnit:
    location = {"page_number": row.get("page_number")}
    evidence_id = f"paper:{row.get('source_file')}:{row.get('attribute')}:{row.get('page_number') or 'meta'}"
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_type="paper",
        source_file=row["source_file"],
        fact_type="paper_claim",
        entity=row.get("entity") or row.get("paper_title"),
        attribute=row.get("attribute"),
        value=row.get("value"),
        text_span=row.get("text_span"),
        location={key: value for key, value in location.items() if value is not None},
        confidence=float(row.get("confidence", 1.0)),
        citation={
            "file_name": row.get("source_file"),
            "source_path": row.get("source_path"),
            "doc_type": "paper",
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            **{key: value for key, value in location.items() if value is not None},
        },
        metadata={
            "paper_title": row.get("paper_title"),
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            "task_type": row.get("attribute"),
            "measurement": row.get("attribute"),
        },
    )


def _row(
    source: Path,
    metadata: dict,
    document_id: int,
    title: str,
    attribute: str,
    value: str,
    text_span: str,
    page_number: int | None,
    confidence: float,
) -> dict:
    return {
        "document_id": document_id,
        "source_file": source.name,
        "source_path": str(source),
        "paper_title": title,
        "year": metadata.get("year"),
        "version": metadata.get("version"),
        "language": metadata.get("language"),
        "entity": "microalgae cultivation",
        "attribute": attribute,
        "value": value,
        "text_span": text_span,
        "page_number": page_number,
        "confidence": confidence,
    }


def _paper_title(source: Path, metadata: dict) -> str:
    topic = metadata.get("topic") or source.stem
    year = metadata.get("year")
    title = str(topic).replace("_", " ")
    return f"{title} ({year})" if year else title


def _snippet(text: str, pattern: str, max_len: int = 220) -> str:
    match = re.search(pattern, text, re.I)
    if not match:
        return text[:max_len]
    start = max(match.start() - 80, 0)
    end = min(match.end() + 120, len(text))
    return re.sub(r"\s+", " ", text[start:end]).strip()[:max_len]


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = (row["source_file"], row["attribute"], row.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped
