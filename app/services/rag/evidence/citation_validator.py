from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.rag_evidence_schema import EvidenceUnit, GroundedAnswer, GroundedClaim, GroundedCitation


class CitationValidationIssue(BaseModel):
    code: str
    message: str
    claim_id: str | None = None
    citation_id: str | None = None
    evidence_id: str | None = None
    severity: str = "warning"


class CitationValidationResult(BaseModel):
    valid: bool
    issues: list[CitationValidationIssue] = Field(default_factory=list)


def validate_grounded_answer(answer: GroundedAnswer) -> CitationValidationResult:
    if answer.answerability.status == "blocked" or not answer.direct_answer.strip():
        return CitationValidationResult(valid=True)
    evidence_by_id = {item.evidence_id: item for item in answer.evidence}
    citations_by_evidence_id = {item.evidence_id: item for item in answer.citations}
    issues: list[CitationValidationIssue] = []

    _validate_citation_records(answer.citations, evidence_by_id, issues)
    _validate_claims(answer.claims, evidence_by_id, citations_by_evidence_id, issues)

    has_error = any(item.severity == "error" for item in issues)
    return CitationValidationResult(valid=not has_error, issues=issues)


def _validate_citation_records(
    citations: list[GroundedCitation],
    evidence_by_id: dict[str, EvidenceUnit],
    issues: list[CitationValidationIssue],
) -> None:
    seen_source_ids: set[int] = set()
    for citation in citations:
        citation_id = str(citation.source_id)
        if citation.source_id in seen_source_ids:
            issues.append(
                CitationValidationIssue(
                    code="duplicate_citation_id",
                    message=f"Citation source_id {citation.source_id} appears more than once.",
                    citation_id=citation_id,
                    evidence_id=citation.evidence_id,
                    severity="error",
                )
            )
        seen_source_ids.add(citation.source_id)

        evidence = evidence_by_id.get(citation.evidence_id)
        if evidence is None:
            issues.append(
                CitationValidationIssue(
                    code="citation_evidence_missing",
                    message="Citation points to an evidence_id that is not present in the answer evidence.",
                    citation_id=citation_id,
                    evidence_id=citation.evidence_id,
                    severity="error",
                )
            )
            continue
        if evidence.source_file != citation.source_file or evidence.source_type != citation.source_type:
            issues.append(
                CitationValidationIssue(
                    code="citation_source_mismatch",
                    message="Citation source file/type does not match the referenced evidence.",
                    citation_id=citation_id,
                    evidence_id=citation.evidence_id,
                    severity="error",
                )
            )
        for key, cited_value in citation.location.items():
            evidence_value = evidence.location.get(key)
            if cited_value is not None and evidence_value is not None and cited_value != evidence_value:
                issues.append(
                    CitationValidationIssue(
                        code="citation_location_mismatch",
                        message=f"Citation location field {key} does not match evidence location.",
                        citation_id=citation_id,
                        evidence_id=citation.evidence_id,
                        severity="error",
                    )
                )


def _validate_claims(
    claims: list[GroundedClaim],
    evidence_by_id: dict[str, EvidenceUnit],
    citations_by_evidence_id: dict[str, GroundedCitation],
    issues: list[CitationValidationIssue],
) -> None:
    for index, claim in enumerate(claims, start=1):
        claim_id = f"claim:{index}"
        if claim.claim_type == "fact" and not claim.evidence_ids:
            issues.append(
                CitationValidationIssue(
                    code="fact_claim_missing_citation",
                    message="Fact claim has no supporting evidence ids.",
                    claim_id=claim_id,
                    severity="error",
                )
            )
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_by_id:
                issues.append(
                    CitationValidationIssue(
                        code="claim_evidence_missing",
                        message="Claim points to an evidence_id that is not present in selected evidence.",
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        severity="error",
                    )
                )
                continue
            if evidence_id not in citations_by_evidence_id:
                issues.append(
                    CitationValidationIssue(
                        code="claim_citation_missing",
                        message="Claim evidence does not have a matching CitationRecord.",
                        claim_id=claim_id,
                        evidence_id=evidence_id,
                        severity="error" if claim.claim_type == "fact" else "warning",
                    )
                )
