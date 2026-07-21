from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from app.services.rag.ingestion.document_models import (
    DocumentElement,
    NormalizedDocument,
    stable_content_hash,
    stable_document_id,
    stable_element_id,
)
from app.services.rag.ingestion.loaders import load_source_units


ParseQualityStatus = Literal["ok", "degraded", "failed"]
ParseAttemptDecision = Literal["accept", "retry_next", "reject"]

PDF_ROUTER_VERSION = "2026-07-parser-router-v1"


@dataclass
class PdfProbeReport:
    status: str
    page_count: int = 0
    text_char_count: int = 0
    text_chars_per_page: float = 0.0
    garbled_ratio: float = 0.0
    image_block_ratio: float = 0.0
    has_table_signal: bool = False
    has_formula_signal: bool = False
    has_multicolumn_signal: bool = False
    has_cjk_signal: bool = False
    is_scanned_or_low_text: bool = False
    is_garbled: bool = False
    error: str | None = None

    def model_dump(self) -> dict:
        return {
            "status": self.status,
            "page_count": self.page_count,
            "text_char_count": self.text_char_count,
            "text_chars_per_page": self.text_chars_per_page,
            "garbled_ratio": self.garbled_ratio,
            "image_block_ratio": self.image_block_ratio,
            "has_table_signal": self.has_table_signal,
            "has_formula_signal": self.has_formula_signal,
            "has_multicolumn_signal": self.has_multicolumn_signal,
            "has_cjk_signal": self.has_cjk_signal,
            "is_scanned_or_low_text": self.is_scanned_or_low_text,
            "is_garbled": self.is_garbled,
            "error": self.error,
        }


@dataclass
class ParseQualityReport:
    parser_name: str
    route_reason: str
    status: ParseQualityStatus
    decision: ParseAttemptDecision = "reject"
    hard_failures: list[str] = field(default_factory=list)
    soft_scores: dict[str, float] = field(default_factory=dict)
    degraded_flags: list[str] = field(default_factory=list)
    provisional_rank: int | None = None

    def model_dump(self) -> dict:
        return {
            "parser_name": self.parser_name,
            "route_reason": self.route_reason,
            "status": self.status,
            "decision": self.decision,
            "hard_failures": self.hard_failures,
            "soft_scores": self.soft_scores,
            "degraded_flags": self.degraded_flags,
            "provisional_rank": self.provisional_rank,
        }


@dataclass
class ParseAttempt:
    parser_name: str
    document: NormalizedDocument | None
    quality: ParseQualityReport
    error: str | None = None


class PdfParserAdapter(Protocol):
    name: str
    version: str

    def parse(self, source: Path, metadata: dict, probe: PdfProbeReport) -> NormalizedDocument:
        ...


class ParserUnavailable(RuntimeError):
    pass


def parse_pdf_document(source_path: str | Path, metadata: dict) -> NormalizedDocument:
    source = Path(source_path)
    strict = _strict_mode()
    probe = probe_pdf(source)
    candidates = parser_candidates_for_probe(probe)
    attempts: list[ParseAttempt] = []
    provisional: list[ParseAttempt] = []

    for index, candidate in enumerate(candidates):
        adapter = _adapter_for(candidate)
        route_reason = _route_reason(probe, candidate)
        try:
            document = adapter.parse(source, metadata, probe)
            quality = evaluate_parse_quality(
                document=document,
                parser_name=adapter.name,
                route_reason=route_reason,
                probe=probe,
                is_salvage=candidate == "pypdf_salvage",
            )
        except Exception as exc:
            quality = ParseQualityReport(
                parser_name=candidate,
                route_reason=route_reason,
                status="failed",
                hard_failures=[type(exc).__name__],
                degraded_flags=[],
            )
            attempts.append(ParseAttempt(candidate, None, quality, error=str(exc)))
            continue

        has_next = index < len(candidates) - 1
        if quality.status == "ok":
            quality.decision = "accept"
            attempts.append(ParseAttempt(adapter.name, document, quality))
            return _with_router_metadata(document, probe, attempts, accepted_quality=quality)

        if quality.status == "degraded":
            quality.decision = "retry_next" if has_next else "reject"
            attempt = ParseAttempt(adapter.name, document, quality)
            attempts.append(attempt)
            provisional.append(attempt)
            if has_next:
                continue
            break

        quality.decision = "reject"
        attempts.append(ParseAttempt(adapter.name, document, quality))

    if provisional and not strict:
        selected = _best_provisional(provisional)
        for rank, item in enumerate(_rank_provisional(provisional), start=1):
            item.quality.provisional_rank = rank
        selected.quality.decision = "accept"
        return _with_router_metadata(selected.document, probe, attempts, accepted_quality=selected.quality)

    if provisional and strict:
        raise RuntimeError("PDF parsing produced only degraded provisional results in strict production mode")

    failure_summary = "; ".join(
        f"{attempt.parser_name}:{attempt.error or ','.join(attempt.quality.hard_failures)}"
        for attempt in attempts
    )
    raise RuntimeError(f"PDF parser router could not parse {source.name}: {failure_summary or 'no candidates'}")


def probe_pdf(source: Path) -> PdfProbeReport:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        return PdfProbeReport(status="probe_unavailable", error=str(exc))

    try:
        document = fitz.open(str(source))
    except Exception as exc:
        return PdfProbeReport(status="failed", error=str(exc))

    page_count = int(getattr(document, "page_count", 0) or 0)
    text_parts: list[str] = []
    image_blocks = 0
    text_blocks = 0
    multicolumn_pages = 0
    try:
        for page in document:
            text = page.get_text("text") or ""
            text_parts.append(text)
            try:
                blocks = page.get_text("dict").get("blocks") or []
            except Exception:
                blocks = []
            x_centers = []
            for block in blocks:
                block_type = block.get("type")
                if block_type == 1:
                    image_blocks += 1
                elif block_type == 0:
                    text_blocks += 1
                    bbox = block.get("bbox") or []
                    if len(bbox) >= 4:
                        x_centers.append((float(bbox[0]) + float(bbox[2])) / 2.0)
            if _looks_multicolumn(x_centers):
                multicolumn_pages += 1
    finally:
        document.close()

    full_text = "\n".join(text_parts)
    text_char_count = len(re.sub(r"\s+", "", full_text))
    text_chars_per_page = text_char_count / max(page_count, 1)
    image_block_ratio = image_blocks / max(image_blocks + text_blocks, 1)
    garbled_ratio = _garbled_ratio(full_text)
    has_table_signal = _has_table_signal(full_text)
    has_formula_signal = bool(re.search(r"(?:∑|∫|≤|≥|≈|µ|μ|[A-Za-z]\s*=\s*[-+*/()0-9A-Za-z])", full_text))
    has_cjk_signal = bool(re.search(r"[\u4e00-\u9fff]", full_text))
    is_scanned_or_low_text = page_count > 0 and text_chars_per_page < 40 and image_block_ratio > 0.2
    is_garbled = garbled_ratio > 0.08
    return PdfProbeReport(
        status="ok",
        page_count=page_count,
        text_char_count=text_char_count,
        text_chars_per_page=text_chars_per_page,
        garbled_ratio=garbled_ratio,
        image_block_ratio=image_block_ratio,
        has_table_signal=has_table_signal,
        has_formula_signal=has_formula_signal,
        has_multicolumn_signal=multicolumn_pages > 0,
        has_cjk_signal=has_cjk_signal,
        is_scanned_or_low_text=is_scanned_or_low_text,
        is_garbled=is_garbled,
    )


def parser_candidates_for_probe(probe: PdfProbeReport) -> list[str]:
    if probe.status != "ok":
        return ["pypdf", "docling", "mineru"]
    if probe.is_scanned_or_low_text or probe.is_garbled:
        return ["mineru", "docling"]
    if probe.has_cjk_signal and (probe.has_formula_signal or probe.has_table_signal or probe.has_multicolumn_signal):
        return ["mineru", "docling"]
    if probe.has_multicolumn_signal or probe.has_table_signal or probe.has_formula_signal:
        return ["docling", "mineru", "pypdf_salvage"]
    return ["pypdf", "docling", "mineru"]


def evaluate_parse_quality(
    *,
    document: NormalizedDocument,
    parser_name: str,
    route_reason: str,
    probe: PdfProbeReport,
    is_salvage: bool = False,
) -> ParseQualityReport:
    hard_failures = _hard_failures(document, probe)
    if hard_failures:
        return ParseQualityReport(
            parser_name=parser_name,
            route_reason=route_reason,
            status="failed",
            hard_failures=hard_failures,
        )

    text = "\n".join(element.content for element in document.elements if element.content)
    text_chars = len(re.sub(r"\s+", "", text))
    source_text_chars = max(probe.text_char_count, text_chars, 1)
    text_coverage = min(text_chars / source_text_chars, 1.0)
    garbled_ratio = _garbled_ratio(text)
    elements_with_bbox = sum(1 for element in document.elements if (element.source_locator or {}).get("bbox"))
    bbox_coverage = elements_with_bbox / max(len(document.elements), 1)
    table_count = sum(1 for element in document.elements if element.element_type in {"table", "table_row"})
    formula_count = sum(1 for element in document.elements if element.element_type == "formula")
    degraded_flags = []
    if text_coverage < 0.60:
        degraded_flags.append("low_text_coverage")
    if garbled_ratio > 0.08:
        degraded_flags.append("high_garbled_ratio")
    if probe.has_table_signal and table_count == 0:
        degraded_flags.append("table_signal_without_table_elements")
    if probe.has_formula_signal and formula_count == 0 and parser_name != "pypdf":
        degraded_flags.append("formula_signal_without_formula_elements")
    if parser_name in {"docling", "mineru"} and bbox_coverage < 0.20:
        degraded_flags.append("low_bbox_coverage")
    if is_salvage:
        degraded_flags.extend(["salvage_parser", "structure_unreliable", "tables_unreliable", "formulas_unreliable"])

    return ParseQualityReport(
        parser_name=parser_name,
        route_reason=route_reason,
        status="degraded" if degraded_flags else "ok",
        soft_scores={
            "text_coverage": round(text_coverage, 4),
            "garbled_ratio": round(garbled_ratio, 4),
            "bbox_coverage": round(bbox_coverage, 4),
            "table_count": float(table_count),
            "formula_count": float(formula_count),
        },
        degraded_flags=list(dict.fromkeys(degraded_flags)),
    )


class PypdfAdapter:
    name = "pypdf"
    version = "2026-07-parser-router-v1"

    def parse(self, source: Path, metadata: dict, probe: PdfProbeReport) -> NormalizedDocument:
        elements = []
        for unit in load_source_units(source, metadata):
            locator = {"page_number": unit.page_number} if unit.page_number else {}
            elements.append(
                _make_element(
                    source=source,
                    metadata=metadata,
                    index=len(elements),
                    element_type="paragraph",
                    content=unit.content,
                    section_path=[unit.section or unit.title or source.stem],
                    page_number=unit.page_number,
                    source_locator=locator,
                    element_metadata={"adapter_output": "pypdf_text"},
                )
            )
        return _base_document(source, metadata, elements, self.name, self.version)


class PypdfSalvageAdapter(PypdfAdapter):
    name = "pypdf_salvage"

    def parse(self, source: Path, metadata: dict, probe: PdfProbeReport) -> NormalizedDocument:
        document = super().parse(source, metadata, probe)
        return document.model_copy(
            update={
                "parser_name": self.name,
                "metadata": {
                    **document.metadata,
                    "salvage": True,
                    "structure_unreliable": True,
                    "tables_unreliable": True,
                    "formulas_unreliable": True,
                },
            }
        )


class DoclingAdapter:
    name = "docling"
    version = "2026-07-parser-router-v1"

    def parse(self, source: Path, metadata: dict, probe: PdfProbeReport) -> NormalizedDocument:
        if not _env_bool("RAG_DOCLING_ENABLED", default=False):
            raise ParserUnavailable("Docling adapter is disabled; set RAG_DOCLING_ENABLED=true")
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise ParserUnavailable(f"Docling dependency is unavailable: {exc}") from exc

        converter = DocumentConverter()
        result = converter.convert(str(source))
        payload = _structured_payload_from_docling(result)
        elements = _elements_from_structured_payload(source, metadata, payload, parser_name=self.name)
        return _base_document(source, metadata, elements, self.name, self.version, extra_metadata={"adapter_output": "docling_json"})


class MinerUAdapter:
    name = "mineru"
    version = "2026-07-parser-router-v1"

    def parse(self, source: Path, metadata: dict, probe: PdfProbeReport) -> NormalizedDocument:
        payload = _load_mineru_payload(source)
        elements = _elements_from_structured_payload(source, metadata, payload, parser_name=self.name)
        return _base_document(source, metadata, elements, self.name, self.version, extra_metadata={"adapter_output": "mineru_json"})


def _load_mineru_payload(source: Path) -> Any:
    sidecar = Path(str(source) + ".mineru.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    command = os.getenv("RAG_MINERU_COMMAND")
    if not command:
        raise ParserUnavailable("MinerU adapter requires a sidecar .mineru.json or RAG_MINERU_COMMAND")
    timeout = float(os.getenv("RAG_MINERU_TIMEOUT_SECONDS", "180"))
    with tempfile.TemporaryDirectory(prefix="rag_mineru_") as temp_dir:
        output_path = Path(temp_dir) / "mineru_output.json"
        completed = subprocess.run(
            [command, str(source), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "MinerU command failed")[:1000])
        if not output_path.exists():
            raise RuntimeError("MinerU command did not produce structured JSON output")
        return json.loads(output_path.read_text(encoding="utf-8"))


def _structured_payload_from_docling(result: Any) -> Any:
    document = getattr(result, "document", result)
    for method_name in ("export_to_dict", "model_dump", "dict"):
        method = getattr(document, method_name, None)
        if callable(method):
            return method()
    json_method = getattr(document, "export_to_json", None)
    if callable(json_method):
        return json.loads(json_method())
    raise RuntimeError("Docling result did not expose structured document JSON")


def _elements_from_structured_payload(source: Path, metadata: dict, payload: Any, parser_name: str) -> list[DocumentElement]:
    items = list(_iter_structured_items(payload))
    elements = []
    for item in items:
        content = str(item.get("text") or item.get("content") or item.get("html") or "").strip()
        if not content:
            continue
        element_type = _coerce_element_type(item.get("type") or item.get("label") or item.get("category"))
        section_path = _coerce_section_path(item.get("section_path") or item.get("section") or item.get("heading"))
        locator = _compact_dict(
            {
                "page_number": item.get("page_number") or item.get("page") or item.get("page_idx"),
                "bbox": item.get("bbox") or item.get("bounding_box"),
                "table_index": item.get("table_index"),
                "section_path": section_path,
            }
        )
        elements.append(
            _make_element(
                source=source,
                metadata=metadata,
                index=len(elements),
                element_type=element_type,
                content=content,
                section_path=section_path or [metadata.get("topic") or source.stem],
                page_number=_coerce_int(locator.get("page_number")),
                source_locator=locator,
                element_metadata={
                    "adapter_output": f"{parser_name}_json",
                    "raw_type": item.get("type") or item.get("label") or item.get("category"),
                    "bbox": locator.get("bbox"),
                },
            )
        )
    return elements


def _iter_structured_items(payload: Any, inherited: dict | None = None):
    inherited = inherited or {}
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_structured_items(item, inherited)
        return
    if not isinstance(payload, dict):
        return
    current = dict(inherited)
    for key in ("page", "page_number", "page_idx", "section", "section_path", "heading"):
        value = payload.get(key)
        if value is not None and value != "" and value != []:
            current[key] = payload.get(key)
    if any(key in payload for key in ("text", "content", "html")):
        yield {**current, **payload}
    for key in ("texts", "elements", "body", "children", "pages", "blocks", "tables", "formulas"):
        child = payload.get(key)
        if child is not None:
            yield from _iter_structured_items(child, current)


def _hard_failures(document: NormalizedDocument, probe: PdfProbeReport) -> list[str]:
    failures = []
    if not document.elements:
        failures.append("empty_document")
    if not any((element.content or "").strip() for element in document.elements):
        failures.append("no_text_content")
    if probe.page_count:
        page_numbers = [element.page_number for element in document.elements if element.page_number is not None]
        if any(page < 1 or page > probe.page_count for page in page_numbers):
            failures.append("illegal_page_number")
        missing_pages = (document.metadata or {}).get("missing_pages") or []
        if missing_pages:
            failures.append("missing_pages")
    try:
        NormalizedDocument.model_validate(document.model_dump())
    except Exception:
        failures.append("schema_validation_failed")
    return failures


def _adapter_for(name: str) -> PdfParserAdapter:
    if name == "pypdf":
        return PypdfAdapter()
    if name == "pypdf_salvage":
        return PypdfSalvageAdapter()
    if name == "docling":
        return DoclingAdapter()
    if name == "mineru":
        return MinerUAdapter()
    raise ValueError(f"Unknown PDF parser adapter: {name}")


def _with_router_metadata(
    document: NormalizedDocument | None,
    probe: PdfProbeReport,
    attempts: list[ParseAttempt],
    *,
    accepted_quality: ParseQualityReport,
) -> NormalizedDocument:
    if document is None:
        raise RuntimeError("Accepted parser attempt did not produce a document")
    attempt_payload = []
    for attempt in attempts:
        payload = attempt.quality.model_dump()
        if attempt.error:
            payload["error"] = attempt.error
        attempt_payload.append(payload)
    return document.model_copy(
        update={
            "metadata": {
                **(document.metadata or {}),
                "pdf_probe": probe.model_dump(),
                "parser_router_version": PDF_ROUTER_VERSION,
                "parser_attempts": attempt_payload,
                "quality_report": accepted_quality.model_dump(),
            }
        }
    )


def _base_document(
    source: Path,
    metadata: dict,
    elements: list[DocumentElement],
    parser_name: str,
    parser_version: str,
    extra_metadata: dict | None = None,
) -> NormalizedDocument:
    document_id = stable_document_id(metadata["source_path"])
    content_hash = metadata.get("content_hash") or ""
    version = metadata.get("version") or metadata.get("year") or content_hash[:16] or "unversioned"
    return NormalizedDocument(
        document_id=document_id,
        document_version=version,
        source_path=metadata["source_path"],
        source_type="local_file",
        title=metadata.get("topic") or source.stem,
        elements=_link_elements(elements),
        metadata={**dict(metadata), **(extra_metadata or {})},
        parser_name=parser_name,
        parser_version=parser_version,
        content_hash=content_hash,
    )


def _make_element(
    *,
    source: Path,
    metadata: dict,
    index: int,
    element_type: str,
    content: str,
    section_path: list[str] | None = None,
    parent_id: str | None = None,
    page_number: int | None = None,
    source_locator: dict[str, Any] | None = None,
    element_metadata: dict[str, Any] | None = None,
) -> DocumentElement:
    document_id = stable_document_id(metadata["source_path"])
    locator = {
        "source_path": metadata["source_path"],
        "file_name": source.name,
        **(source_locator or {}),
    }
    clean_content = _normalize_text(content)
    return DocumentElement(
        element_id=stable_element_id(
            document_id=document_id,
            element_type=element_type,
            index=index,
            content=clean_content,
            source_locator=locator,
        ),
        document_id=document_id,
        element_type=element_type,
        content=clean_content,
        page_number=page_number,
        section_path=section_path or [],
        parent_id=parent_id,
        source_locator=locator,
        metadata=element_metadata or {},
        content_hash=stable_content_hash(clean_content, locator),
    )


def _link_elements(elements: list[DocumentElement]) -> list[DocumentElement]:
    linked = []
    previous_id = None
    for index, element in enumerate(elements):
        next_id = elements[index + 1].element_id if index + 1 < len(elements) else None
        parent_id = element.parent_id
        if not parent_id and element.section_path:
            parent_id = stable_element_id(
                document_id=element.document_id,
                element_type="section",
                index=-1,
                content="/".join(element.section_path),
                source_locator={"section_path": element.section_path},
            )
        linked.append(element.model_copy(update={"previous_id": previous_id, "next_id": next_id, "parent_id": parent_id}))
        previous_id = element.element_id
    return linked


def _rank_provisional(attempts: list[ParseAttempt]) -> list[ParseAttempt]:
    return sorted(
        attempts,
        key=lambda item: (
            len(item.quality.degraded_flags),
            -float(item.quality.soft_scores.get("text_coverage", 0.0)),
            -float(item.quality.soft_scores.get("bbox_coverage", 0.0)),
            item.parser_name,
        ),
    )


def _best_provisional(attempts: list[ParseAttempt]) -> ParseAttempt:
    return _rank_provisional(attempts)[0]


def _route_reason(probe: PdfProbeReport, candidate: str) -> str:
    if probe.status != "ok":
        return "probe_unavailable_or_failed"
    if probe.is_scanned_or_low_text:
        return "scanned_or_low_text_pdf"
    if probe.is_garbled:
        return "garbled_pdf"
    if probe.has_cjk_signal and (probe.has_formula_signal or probe.has_table_signal or probe.has_multicolumn_signal):
        return "cjk_complex_pdf"
    if candidate == "pypdf_salvage":
        return "complex_pdf_salvage"
    if probe.has_table_signal or probe.has_formula_signal or probe.has_multicolumn_signal:
        return "native_complex_pdf"
    return "simple_text_pdf"


def _looks_multicolumn(x_centers: list[float]) -> bool:
    if len(x_centers) < 6:
        return False
    midpoint = (min(x_centers) + max(x_centers)) / 2.0
    left = sum(1 for value in x_centers if value < midpoint)
    right = sum(1 for value in x_centers if value >= midpoint)
    return left >= 3 and right >= 3


def _has_table_signal(text: str) -> bool:
    if re.search(r"\b(table|tab\.)\s*\d+", text or "", re.I):
        return True
    lines = [line for line in (text or "").splitlines() if line.strip()]
    pipe_like = sum(1 for line in lines if line.count("|") >= 2 or len(re.split(r"\s{2,}", line.strip())) >= 3)
    return pipe_like >= 3


def _garbled_ratio(text: str) -> float:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    bad = sum(1 for char in compact if char in {"�", "□", "\ufffd"} or ord(char) < 32)
    mojibake = len(re.findall(r"[锟閿�]{1,}", compact))
    return (bad + mojibake) / max(len(compact), 1)


def _coerce_element_type(raw: Any) -> str:
    value = str(raw or "").lower()
    if "heading" in value or value in {"title", "section_header"}:
        return "heading"
    if "table" in value:
        return "table"
    if "formula" in value or "equation" in value:
        return "formula"
    if "caption" in value:
        return "figure_caption"
    if "list" in value:
        return "list"
    if "text" in value or "paragraph" in value:
        return "paragraph"
    return "unknown"


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != "" and value != []
    }


def _coerce_section_path(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    return [part.strip() for part in str(value).replace(">", "/").split("/") if part.strip()]


def _coerce_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        item = int(value)
    except (TypeError, ValueError):
        return None
    if item == 0:
        return 1
    return item


def _normalize_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip()).strip()


def _strict_mode() -> bool:
    return _env_bool("RAG_STRICT_PRODUCTION", default=False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
