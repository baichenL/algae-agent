from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models.rag_evidence_schema import (
    EvidenceUnit,
    ExtractionReport,
    KnowledgeSource,
    ParsedDocument,
)
from app.services.rag.evidence.adapters.experiment_data_adapter import (
    experiment_value_rows_to_evidence_units,
    extract_experiment_data_value_rows,
)
from app.services.rag.evidence.adapters.paper_adapter import (
    extract_paper_fact_rows,
    paper_rows_to_evidence_units,
)
from app.services.rag.evidence.adapters.recipe_adapter import (
    extract_recipe_component_rows,
    recipe_rows_to_evidence_units,
)
from app.services.rag.evidence.adapters.sop_adapter import (
    extract_sop_fact_rows,
    sop_rows_to_evidence_units,
)
from app.services.rag.evidence.adapters.table_schema_adapter import (
    extract_table_schema_rows,
    schema_rows_to_evidence_units,
)


class EvidenceExtractor(ABC):
    """Plugin boundary: parsers load source shape, extractors emit EvidenceUnit only."""

    name: str = "base"

    @abstractmethod
    def supports(self, source: KnowledgeSource) -> bool:
        """Return True only when this extractor owns the source type/doc type."""

    def parse(self, source: KnowledgeSource) -> ParsedDocument:
        """Keep parsing lightweight here; heavy loaders remain in existing ingestion code."""
        return ParsedDocument(source=source, metadata=source.metadata)

    @abstractmethod
    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        """Convert source-specific records into normalized EvidenceUnit objects."""

    def validate(self, evidence: list[EvidenceUnit]) -> ExtractionReport:
        """Validation is intentionally shallow in MVP; eval catches behavioral regressions."""
        return ExtractionReport(
            extractor_name=self.name,
            status="success",
            evidence_count=len(evidence),
        )


class TableSchemaExtractor(EvidenceExtractor):
    name = "table_schema_extractor"

    def supports(self, source: KnowledgeSource) -> bool:
        return source.doc_type == "experiment_data" and Path(source.source_path).suffix.lower() in {".csv", ".xlsx", ".xls"}

    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        rows = extract_table_schema_rows(parsed.source.source_path, parsed.source.model_dump(), document_id)
        return schema_rows_to_evidence_units(rows)


class ExperimentDataExtractor(EvidenceExtractor):
    name = "experiment_data_extractor"

    def supports(self, source: KnowledgeSource) -> bool:
        return source.doc_type == "experiment_data" and Path(source.source_path).suffix.lower() in {".csv", ".xlsx", ".xls"}

    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        rows = extract_experiment_data_value_rows(parsed.source.source_path, parsed.source.model_dump(), document_id)
        return experiment_value_rows_to_evidence_units(rows)


class RecipeExtractor(EvidenceExtractor):
    name = "recipe_extractor"

    def supports(self, source: KnowledgeSource) -> bool:
        return source.doc_type == "media_recipe"

    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        # Recipe tables encode lab-specific stock/working-solution groups; keep this as a domain extractor.
        rows = extract_recipe_component_rows(parsed.source.source_path, parsed.source.model_dump(), document_id)
        return recipe_rows_to_evidence_units(rows)


class SOPExtractor(EvidenceExtractor):
    name = "sop_extractor"

    def supports(self, source: KnowledgeSource) -> bool:
        return source.doc_type == "manual"

    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        rows = extract_sop_fact_rows(parsed.source.source_path, parsed.source.model_dump(), document_id)
        return sop_rows_to_evidence_units(rows)


class PaperClaimExtractor(EvidenceExtractor):
    name = "paper_claim_extractor"

    def supports(self, source: KnowledgeSource) -> bool:
        return source.doc_type == "paper"

    def extract(self, parsed: ParsedDocument, document_id: int) -> list[EvidenceUnit]:
        rows = extract_paper_fact_rows(parsed.source.source_path, parsed.source.model_dump(), document_id)
        return paper_rows_to_evidence_units(rows)


DEFAULT_EXTRACTORS: tuple[EvidenceExtractor, ...] = (
    RecipeExtractor(),
    SOPExtractor(),
    PaperClaimExtractor(),
    TableSchemaExtractor(),
    ExperimentDataExtractor(),
)


def matching_extractors(source: KnowledgeSource) -> list[EvidenceExtractor]:
    return [extractor for extractor in DEFAULT_EXTRACTORS if extractor.supports(source)]
