import csv
import re
from pathlib import Path
from typing import Any

from app.models.rag_evidence_schema import EvidenceUnit
from app.services.rag.evidence.adapters.table_schema_adapter import (
    extract_unit,
    infer_column_role,
    normalize_column_name,
)


def extract_experiment_data_value_rows(path: str | Path, metadata: dict, document_id: int) -> list[dict]:
    if metadata.get("doc_type") != "experiment_data":
        return []
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return _extract_xlsx_values(source, metadata, document_id)
    if source.suffix.lower() == ".csv":
        return _extract_csv_values(source, metadata, document_id)
    return []


def experiment_value_rows_to_evidence_units(rows: list[dict]) -> list[EvidenceUnit]:
    return [experiment_value_row_to_evidence_unit(row) for row in rows]


def experiment_value_row_to_evidence_unit(row: dict) -> EvidenceUnit:
    location = {
        "sheet_name": row.get("sheet_name"),
        "row_index": row.get("row_index"),
    }
    evidence_id = (
        f"data_value:{row.get('source_file')}:{row.get('sheet_name') or 'csv'}:"
        f"{row.get('row_index')}:{row.get('normalized_column_name')}"
    )
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_type="experiment_data",
        source_file=row["source_file"],
        fact_type="data_value",
        entity=f"{row['source_file']}:{row.get('sheet_name') or 'csv'}:row_{row.get('row_index')}",
        attribute=row.get("column_name"),
        value=row.get("value_text"),
        unit=row.get("unit"),
        text_span=f"{row.get('column_name')}={row.get('value_text')}",
        location={key: value for key, value in location.items() if value is not None},
        confidence=1.0,
        citation={
            "file_name": row.get("source_file"),
            "source_path": row.get("source_path"),
            "doc_type": "experiment_data",
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            **{key: value for key, value in location.items() if value is not None},
        },
        metadata={
            "normalized_column_name": row.get("normalized_column_name"),
            "inferred_role": row.get("inferred_role"),
            "numeric_value": row.get("numeric_value"),
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            "measurement": row.get("inferred_role"),
        },
    )


def _extract_xlsx_values(path: Path, metadata: dict, document_id: int) -> list[dict]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to ingest XLSX data values") from exc
    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    rows: list[dict] = []
    try:
        for sheet in workbook.worksheets:
            sheet_rows = list(sheet.iter_rows(values_only=True))
            if not sheet_rows:
                continue
            header_index = _find_header_index(sheet_rows)
            headers = [_clean_cell(value) for value in sheet_rows[header_index]]
            for row_offset, row in enumerate(sheet_rows[header_index + 1 :], start=header_index + 2):
                rows.extend(_build_value_rows(path, metadata, document_id, sheet.title, row_offset, headers, row))
    finally:
        workbook.close()
    return rows


def _extract_csv_values(path: Path, metadata: dict, document_id: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as file:
        reader = list(csv.reader(file))
    if not reader:
        return []
    header_index = _find_header_index(reader)
    headers = [_clean_cell(value) for value in reader[header_index]]
    rows = []
    for row_offset, row in enumerate(reader[header_index + 1 :], start=header_index + 2):
        rows.extend(_build_value_rows(path, metadata, document_id, None, row_offset, headers, row))
    return rows


def _build_value_rows(
    path: Path,
    metadata: dict,
    document_id: int,
    sheet_name: str | None,
    row_index: int,
    headers: list[str],
    row: Any,
) -> list[dict]:
    result = []
    for index, header in enumerate(headers):
        if not header or index >= len(row):
            continue
        value_text = _clean_cell(row[index])
        if value_text == "":
            continue
        result.append(
            {
                "document_id": document_id,
                "source_file": path.name,
                "source_path": str(path),
                "sheet_name": sheet_name,
                "row_index": row_index,
                "column_name": header,
                "normalized_column_name": normalize_column_name(header),
                "inferred_role": infer_column_role(header),
                "value_text": value_text,
                "numeric_value": _parse_float(value_text),
                "unit": extract_unit(header),
                "version": metadata.get("version"),
                "year": metadata.get("year"),
                "language": metadata.get("language"),
            }
        )
    return result


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


def _parse_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
