from app.core.database import (
    list_rag_experiment_data_values,
    list_rag_paper_facts,
    list_rag_recipe_components,
    list_rag_sop_facts,
    list_rag_source_schemas,
)
from app.models.rag_evidence_schema import EvidencePlan, EvidenceUnit, QueryFrame
from app.services.rag.retrieval.retriever import retrieve_chunks
from app.services.rag.evidence.adapters.chunk_adapter import chunks_to_evidence_units
from app.services.rag.evidence.adapters.experiment_data_adapter import experiment_value_rows_to_evidence_units
from app.services.rag.evidence.adapters.paper_adapter import paper_rows_to_evidence_units
from app.services.rag.evidence.adapters.recipe_adapter import recipe_rows_to_evidence_units
from app.services.rag.evidence.adapters.sop_adapter import sop_rows_to_evidence_units
from app.services.rag.evidence.adapters.table_schema_adapter import schema_rows_to_evidence_units


def retrieve_evidence(frame: QueryFrame, plan: EvidencePlan, top_k: int = 5) -> list[EvidenceUnit]:
    evidence: list[EvidenceUnit] = []
    if "table_schema" in plan.preferred_evidence_types:
        rows = list_rag_source_schemas(doc_types=["experiment_data"])
        evidence.extend(schema_rows_to_evidence_units(rows))

    if "recipe_component" in plan.preferred_evidence_types:
        rows = list_rag_recipe_components(doc_types=["media_recipe"])
        evidence.extend(recipe_rows_to_evidence_units(rows))

    if "sop_fact" in plan.preferred_evidence_types:
        rows = list_rag_sop_facts(doc_types=["manual"])
        evidence.extend(sop_rows_to_evidence_units(rows))

    if "paper_fact" in plan.preferred_evidence_types:
        rows = list_rag_paper_facts(doc_types=["paper"])
        evidence.extend(paper_rows_to_evidence_units(rows))

    if "data_analysis" in plan.preferred_evidence_types:
        rows = list_rag_experiment_data_values(doc_types=["experiment_data"])
        evidence.extend(experiment_value_rows_to_evidence_units(rows))

    needs_text_adjudication = "explicit_statement" in plan.required_capabilities
    if (
        "text_chunk" in plan.preferred_evidence_types
        and plan.fallback_allowed
        and (not evidence or needs_text_adjudication)
    ):
        chunks = retrieve_chunks(
            frame.original_question,
            top_k=top_k,
            doc_types=plan.source_constraints or None,
        )
        evidence.extend(chunks_to_evidence_units(chunks))
    return evidence
