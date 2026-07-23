import csv
import re
from pathlib import Path
from typing import Any

from app.models.rag_evidence_schema import EvidenceUnit


def extract_table_schema_rows(path: str | Path, metadata: dict, document_id: int) -> list[dict]:
    source = Path(path)
    extension = source.suffix.lower()
    if extension in {".xlsx", ".xls"}:
        return _extract_xlsx_schema(source, metadata, document_id)
    if extension == ".csv":
        return _extract_csv_schema(source, metadata, document_id)
    return []


def schema_row_to_evidence_unit(row: dict) -> EvidenceUnit:
    location = {
        "sheet_name": row.get("sheet_name"),
        "column_index": row.get("column_index"),
    }
    citation = {
        "file_name": row.get("source_file"),
        "source_path": row.get("source_path"),
        "doc_type": "experiment_data",
        "version": row.get("version"),
        "year": row.get("year"),
        "language": row.get("language"),
        **{key: value for key, value in location.items() if value is not None},
    }
    evidence_id = (
        f"schema:{row.get('source_file')}:{row.get('sheet_name') or 'csv'}:"
        f"{row.get('column_index')}:{row.get('normalized_column_name')}"
    )
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_type="experiment_data",
        source_file=row["source_file"],
        fact_type="table_column",
        entity=f"{row['source_file']}:{row.get('sheet_name') or 'csv'}",
        attribute="column_name",
        value=row["column_name"],
        unit=row.get("unit"),
        text_span=row["column_name"],
        location=location,
        confidence=float(row.get("confidence", 1.0)),
        citation=citation,
        metadata={
            "normalized_column_name": row.get("normalized_column_name"),
            "inferred_role": row.get("inferred_role"),
            "sample_values": row.get("sample_values") or [],
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            "measurement": row.get("inferred_role"),
        },
    )


def schema_rows_to_evidence_units(rows: list[dict]) -> list[EvidenceUnit]:
    return [schema_row_to_evidence_unit(row) for row in rows]


def normalize_column_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = text.replace("μ", "u").replace("µ", "u")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def infer_column_role(name: str) -> str:
    normalized = normalize_column_name(name)
    raw = (name or "").lower()
    if normalized in {"time_h", "time", "时间_h", "时间"} or "时间" in raw:
        return "time"
    if normalized in {"i", "iumol_m_2_s_1"} or re.search(r"light|irradiance|光照|光强", raw, re.I):
        return "light_or_irradiance"
    if normalized in {"n", "n_mg_l"} or re.search(r"nitrogen|氮", raw, re.I):
        return "nitrogen"
    if normalized in {"p", "p_mg_l"} or re.search(r"phosphorus|磷", raw, re.I):
        return "phosphorus"
    if normalized in {"tf"}:
        return "tf"
    if re.search(r"temperature|temp|温度", raw, re.I):
        return "temperature"
    if normalized in {"ph"} or re.search(r"\bpH\b", name or ""):
        return "ph"
    if "co2" in normalized or "co₂" in raw:
        return "co2"
    if "od750" in normalized or normalized == "od":
        return "od"
    if "biomass" in normalized or "生物量" in raw:
        return "biomass"
    return "unknown"


def extract_unit(name: str) -> str | None:
    if not name:
        return None
    match = re.search(r"([A-Za-zμµ]+/[A-Za-z]+|mg/L|g/L|h|%)", name)
    if match:
        return match.group(1)
    if "μmol" in name or "µmol" in name:
        return "μmol·m⁻²·s⁻¹"
    return None


def _extract_xlsx_schema(path: Path, metadata: dict, document_id: int) -> list[dict]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to ingest XLSX schema") from exc

    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    rows: list[dict] = []
    try:
        for sheet in workbook.worksheets:
            sheet_rows = list(sheet.iter_rows(values_only=True))
            if not sheet_rows:
                continue
            header_index = _find_header_index(sheet_rows)
            headers = [_clean_cell(value) for value in sheet_rows[header_index]]
            samples_by_column = _sample_values(sheet_rows[header_index + 1 :], len(headers))
            rows.extend(
                _build_schema_rows(
                    path,
                    metadata,
                    document_id,
                    headers,
                    samples_by_column,
                    sheet_name=sheet.title,
                )
            )
    finally:
        workbook.close()
    return rows


def _extract_csv_schema(path: Path, metadata: dict, document_id: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as file:
        reader = list(csv.reader(file))
    if not reader:
        return []
    header_index = _find_header_index(reader)
    headers = [_clean_cell(value) for value in reader[header_index]]
    samples_by_column = _sample_values(reader[header_index + 1 :], len(headers))
    return _build_schema_rows(path, metadata, document_id, headers, samples_by_column, sheet_name=None)


def _build_schema_rows(
    path: Path,
    metadata: dict,
    document_id: int,
    headers: list[str],
    samples_by_column: list[list[str]],
    sheet_name: str | None,
) -> list[dict]:
    rows = []
    for index, header in enumerate(headers, start=1):
        if not header:
            continue
        rows.append(
            {
                "document_id": document_id,
                "source_file": path.name,
                "source_path": str(path),
                "sheet_name": sheet_name,
                "column_name": header,
                "normalized_column_name": normalize_column_name(header),
                "unit": extract_unit(header),
                "inferred_role": infer_column_role(header),
                "column_index": index,
                "sample_values": samples_by_column[index - 1] if index - 1 < len(samples_by_column) else [],
                "confidence": 1.0,
                "version": metadata.get("version"),
                "year": metadata.get("year"),
                "language": metadata.get("language"),
            }
        )
    return rows


def _sample_values(rows: list[Any], column_count: int, limit: int = 5) -> list[list[str]]:
    samples = [[] for _ in range(column_count)]
    for row in rows:
        for index in range(column_count):
            if index >= len(row):
                continue
            value = _clean_cell(row[index])
            if value and len(samples[index]) < limit:
                samples[index].append(value)
    return samples


def _find_header_index(rows: list[Any]) -> int:
    for index, row in enumerate(rows[:10]):
        values = [_clean_cell(value) for value in row]
        if sum(1 for value in values if value) >= 2:
            return index
    return 0


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
