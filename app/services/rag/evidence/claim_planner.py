from app.models.rag_evidence_schema import GroundedAnswer, GroundedClaim, QueryFrame, SufficiencyResult


def build_claim_plan(
    answer: GroundedAnswer,
    frame: QueryFrame,
    sufficiency: SufficiencyResult,
) -> list[GroundedClaim]:
    evidence_ids = [item.evidence_id for item in answer.evidence]
    if answer.answerability.status == "answered":
        if len(answer.evidence) == 1:
            sentences = [item.strip() for item in answer.direct_answer.split("。") if item.strip()]
            claims = [
                GroundedClaim(
                    claim_type="fact",
                    text=f"{sentences[0]}。" if sentences else answer.direct_answer,
                    evidence_ids=evidence_ids,
                )
            ]
            if len(sentences) > 1:
                claims.append(
                    GroundedClaim(
                        claim_type="explanation",
                        text="。".join(sentences[1:]) + "。",
                        evidence_ids=evidence_ids,
                    )
                )
            return claims
        claims = []
        for item in answer.evidence:
            text = _evidence_claim_text(item)
            claims.append(
                GroundedClaim(
                    claim_type="fact",
                    text=text,
                    evidence_ids=[item.evidence_id],
                )
            )
        return claims

    return [
        GroundedClaim(
            claim_type="background",
            text=_evidence_claim_text(item),
            evidence_ids=[item.evidence_id],
        )
        for item in answer.evidence
    ]


def _evidence_claim_text(item) -> str:
    if item.fact_type in {"sop_fact", "paper_claim", "data_value"}:
        return item.value or item.text_span or item.attribute or item.fact_type
    return item.text_span or item.value or item.attribute or item.fact_type
