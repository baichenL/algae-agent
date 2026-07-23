import re
import unicodedata
from pathlib import Path
from typing import Any

from app.models.rag_evidence_schema import EvidenceUnit
from app.services.rag.evidence.adapters.table_schema_adapter import normalize_column_name


def extract_recipe_component_rows(path: str | Path, metadata: dict, document_id: int) -> list[dict]:
    source = Path(path)
    if metadata.get("doc_type") != "media_recipe":
        return []
    if source.suffix.lower() == ".docx":
        return _extract_docx_recipe_components(source, metadata, document_id)
    return []


def recipe_row_to_evidence_unit(row: dict) -> EvidenceUnit:
    location = {
        "section": row.get("section"),
        "table_index": row.get("table_index"),
    }
    citation = {
        "file_name": row.get("source_file"),
        "source_path": row.get("source_path"),
        "doc_type": "media_recipe",
        "version": row.get("version"),
        "year": row.get("year"),
        "language": row.get("language"),
        **{key: value for key, value in location.items() if value is not None},
    }
    evidence_id = (
        f"recipe:{row.get('source_file')}:{row.get('table_index')}:{row.get('group_name')}:"
        f"{row.get('normalized_component_name')}"
    )
    text_span = _format_recipe_span(row)
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_type="media_recipe",
        source_file=row["source_file"],
        fact_type="recipe_component",
        entity=row.get("entity") or "recipe",
        attribute=row.get("group_name") or "component",
        value=row["component_name"],
        unit=row.get("unit"),
        text_span=text_span,
        location=location,
        confidence=float(row.get("confidence", 1.0)),
        citation=citation,
        metadata={
            "normalized_component_name": row.get("normalized_component_name"),
            "amount": row.get("amount"),
            "amount_text": row.get("amount_text"),
            "solution_type": row.get("solution_type"),
            "group_name": row.get("group_name"),
            "working_addition": row.get("working_addition"),
            "version": row.get("version"),
            "year": row.get("year"),
            "language": row.get("language"),
            "medium": row.get("entity"),
        },
    )


def recipe_rows_to_evidence_units(rows: list[dict]) -> list[EvidenceUnit]:
    return [recipe_row_to_evidence_unit(row) for row in rows]


def normalize_component_name(name: str) -> str:
    normalized = normalize_column_name(_canonical_component_name(name))
    return normalized


def normalize_recipe_group(name: str | None) -> str | None:
    if not name:
        return None
    text = name.lower()
    if re.search(r"磷酸盐|phosphate|k2hpo4|kh2po4", text, re.I):
        return "phosphate_solution"
    if re.search(r"hutner|微量|trace|edta|znso4|feso4|金属", text, re.I):
        return "trace_elements"
    if re.search(r"盐溶液|氮源|nh4cl|mgso4|cacl2|salt", text, re.I):
        return "salt_solution"
    if re.search(r"工作液|tris|冰乙酸|acetic|1\s*l|母液", text, re.I):
        return "working_solution"
    return normalize_column_name(name)


def prettify_component_name(text: str) -> str:
    replacements = {
        "(NH4)6Mo7O24·4H2O": "(NH\u2084)\u2086Mo\u2087O\u2082\u2084\u00b74H\u2082O",
        "(NH4)6Mo7O24": "(NH\u2084)\u2086Mo\u2087O\u2082\u2084",
        "Na2-EDTA2H2O": "Na\u2082-EDTA\u00b72H\u2082O",
        "MgSO4 7H2O": "MgSO\u2084\u00b77H\u2082O",
        "MgSO4·7H2O": "MgSO\u2084\u00b77H\u2082O",
        "CaCl2 2H2O": "CaCl\u2082\u00b72H\u2082O",
        "CaCl2·2H2O": "CaCl\u2082\u00b72H\u2082O",
        "ZnSO4 7H2O": "ZnSO\u2084\u00b77H\u2082O",
        "FeSO4 7H2O": "FeSO\u2084\u00b77H\u2082O",
        "MnCl2 4H2O": "MnCl\u2082\u00b74H\u2082O",
        "CoCl2 6H2O": "CoCl\u2082\u00b76H\u2082O",
        "CuSO4 5H2O": "CuSO\u2084\u00b75H\u2082O",
        "K2HPO4": "K\u2082HPO\u2084",
        "K₂HPO₄": "K\u2082HPO\u2084",
        "KH2PO4": "KH\u2082PO\u2084",
        "KH₂PO₄": "KH\u2082PO\u2084",
        "NH4Cl": "NH\u2084Cl",
        "NH₄Cl": "NH\u2084Cl",
        "H3BO3": "H\u2083BO\u2083",
        "H₃BO₃": "H\u2083BO\u2083",
        "H2O": "H\u2082O",
        "H₂O": "H\u2082O",
        "Hutner’s": "Hutner's",
        "Tris(sigma)": "Tris",
        "（母液1）": "母液1",
        "（母液2）": "母液2",
        "（母液3）": "母液3",
    }
    formatted = str(text)
    for raw, pretty in replacements.items():
        formatted = formatted.replace(raw, pretty)
    return formatted


def _extract_docx_recipe_components(path: Path, metadata: dict, document_id: int) -> list[dict]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required to ingest DOCX recipe components") from exc

    document = Document(str(path))
    rows: list[dict] = []
    for table_index, table in enumerate(document.tables, start=1):
        table_rows = [
            [_clean_cell(cell.text) for cell in row.cells]
            for row in table.rows
        ]
        rows.extend(_build_table_component_rows(path, metadata, document_id, table_index, table_rows))
    return rows


def _build_table_component_rows(
    path: Path,
    metadata: dict,
    document_id: int,
    table_index: int,
    table_rows: list[list[str]],
) -> list[dict]:
    group_name = _group_for_table(table_index, table_rows)
    result = []
    for row in table_rows[1:]:
        cells = [cell for cell in row if cell]
        if len(cells) < 2:
            continue
        component = cells[0]
        if component == "组分" or component.startswith("定容"):
            continue
        amount_text = cells[1]
        unit = _unit_for_table(table_index, amount_text)
        amount = _parse_amount(amount_text)
        working_addition = _working_addition(component, amount_text, group_name)
        result.append(
            {
                "document_id": document_id,
                "source_file": path.name,
                "source_path": str(path),
                "entity": metadata.get("topic") or "TAP medium",
                "group_name": group_name,
                "component_name": component,
                "normalized_component_name": normalize_component_name(component),
                "amount": amount,
                "unit": unit,
                "amount_text": amount_text,
                "solution_type": group_name,
                "working_addition": working_addition,
                "section": f"{metadata.get('topic') or path.stem} table {table_index}",
                "table_index": table_index,
                "confidence": 1.0,
                "version": metadata.get("version"),
                "year": metadata.get("year"),
                "language": metadata.get("language"),
            }
        )
    return result


def _group_for_table(table_index: int, rows: list[list[str]]) -> str:
    table_text = " ".join(" ".join(row) for row in rows).lower()
    if table_index == 1 or "nh4cl" in table_text:
        return "salt_solution"
    if table_index == 2 or "k2hpo4" in table_text:
        return "phosphate_solution"
    if table_index == 3 or "edta" in table_text or "znso4" in table_text:
        return "trace_elements"
    if table_index == 4 or "tris" in table_text:
        return "working_solution"
    return f"table_{table_index}"


def _unit_for_table(table_index: int, amount_text: str) -> str | None:
    text = amount_text.strip()
    if re.search(r"\bml\b|mL|升|l\b", text, re.I):
        return "mL"
    if re.search(r"\bg\b|克", text, re.I):
        return "g"
    if table_index in {1, 2, 3} and _parse_amount(text) is not None:
        return "g"
    return None


def _parse_amount(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _working_addition(component: str, amount_text: str, group_name: str) -> str | None:
    if group_name == "working_solution":
        return amount_text
    return None


def _format_recipe_span(row: dict) -> str:
    name = prettify_component_name(row.get("component_name") or "")
    amount = row.get("amount_text") or ""
    unit = row.get("unit")
    group = row.get("group_name") or ""
    if amount and unit and not re.search(r"[a-zA-Zμµ毫升克]", amount):
        amount = f"{amount} {unit}"
    return f"{name}: {prettify_component_name(amount)} ({group})"


def _canonical_component_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = text.replace("·", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_cell(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()
