import re
from collections.abc import Iterable

from app.models.rag_evidence_schema import (
    AnswerabilityResult,
    EvidenceAssessment,
    EvidenceUnit,
    QueryFrame,
    QuestionAspects,
    SufficiencyResult,
)


DIRECT_SUPPORT_REQUIREMENTS = {"explicit_statement", "direct_statement"}
STRUCTURED_SUPPORT_REQUIREMENTS = {"structured_schema", "structured_data_analysis", "structured_overview"}


def assess_evidence_sufficiency(
    aspects: QuestionAspects,
    evidence: list[EvidenceUnit],
    answerability: AnswerabilityResult | None = None,
    frame: QueryFrame | None = None,
) -> SufficiencyResult:
    if _structured_answer_is_sufficient(aspects, evidence, answerability, frame):
        ids = [item.evidence_id for item in evidence]
        return SufficiencyResult(
            status="sufficient",
            direct_evidence_ids=ids,
            reason="structured_evidence_or_closed_world_answer",
            assessments=[
                EvidenceAssessment(
                    evidence_id=item.evidence_id,
                    supports_direct_answer=True,
                    reason="structured evidence is the required evidence type",
                )
                for item in evidence
            ],
        )

    if not evidence:
        missing = _required_aspect_names(aspects)
        return SufficiencyResult(
            status="insufficient",
            missing_aspects=missing,
            reason="no_evidence_retrieved",
        )

    assessments = [_assess_single_evidence(aspects, item) for item in evidence]
    direct_ids = [item.evidence_id for item in assessments if item.supports_direct_answer]
    background_ids = [item.evidence_id for item in assessments if item.supports_background]
    missing = _merge_unique(part for item in assessments for part in item.missing_aspects)
    contradicted = _merge_unique(part for item in assessments for part in item.contradicted_aspects)

    if contradicted:
        status = "contradicted"
        reason = "evidence_contradicts_required_aspects"
    elif direct_ids:
        status = "sufficient"
        reason = "direct_evidence_covers_required_aspects"
    elif background_ids:
        status = "background_only"
        reason = "retrieved_evidence_is_relevant_but_missing_required_aspects"
    else:
        status = "insufficient"
        reason = "retrieved_evidence_does_not_cover_required_aspects"

    return SufficiencyResult(
        status=status,
        direct_evidence_ids=direct_ids,
        background_evidence_ids=background_ids,
        missing_aspects=missing,
        contradicted_aspects=contradicted,
        reason=reason,
        assessments=assessments,
    )


def requires_direct_support(aspects: QuestionAspects) -> bool:
    return aspects.required_support in DIRECT_SUPPORT_REQUIREMENTS


def _structured_answer_is_sufficient(
    aspects: QuestionAspects,
    evidence: list[EvidenceUnit],
    answerability: AnswerabilityResult | None,
    frame: QueryFrame | None,
) -> bool:
    if not evidence or not answerability:
        return False
    if (
        frame
        and frame.target_entity == "manual"
        and answerability.status == "answered"
        and any(item.fact_type == "sop_fact" for item in evidence)
    ):
        return True
    if aspects.required_support in DIRECT_SUPPORT_REQUIREMENTS:
        return False
    if answerability.closed_world_negative:
        return True
    if answerability.status != "answered":
        return False
    if aspects.required_support in STRUCTURED_SUPPORT_REQUIREMENTS:
        return True
    if frame and frame.evidence_requirement in STRUCTURED_SUPPORT_REQUIREMENTS:
        return True
    return False


def _assess_single_evidence(aspects: QuestionAspects, item: EvidenceUnit) -> EvidenceAssessment:
    text = _evidence_text(item)
    coverage = {
        "subject": _matches_aspect(text, aspects.subject),
        "condition": _matches_aspect(text, aspects.condition),
        "relation": _matches_relation(text, aspects.relation),
        "target": _matches_aspect(text, aspects.target),
    }
    required = _required_aspect_names(aspects)
    missing = [name for name in required if not coverage.get(name)]
    contradicted = _contradicted_aspects(text, aspects)

    if not requires_direct_support(aspects):
        supports_direct = bool(coverage["subject"] or coverage["target"] or coverage["condition"])
    else:
        supports_direct = not missing and not contradicted

    if requires_direct_support(aspects):
        background_match = bool(coverage["subject"])
    else:
        background_match = bool(
            coverage["subject"] or coverage["target"] or coverage["condition"] or _raw_term_overlap(text, aspects.raw_terms)
        )
    supports_background = not supports_direct and not contradicted and background_match

    reason = _assessment_reason(supports_direct, supports_background, missing, contradicted)
    return EvidenceAssessment(
        evidence_id=item.evidence_id,
        supports_direct_answer=supports_direct,
        supports_background=supports_background,
        missing_aspects=missing,
        contradicted_aspects=contradicted,
        reason=reason,
    )


def _required_aspect_names(aspects: QuestionAspects) -> list[str]:
    required = []
    if aspects.subject:
        required.append("subject")
    if aspects.required_support in DIRECT_SUPPORT_REQUIREMENTS:
        if aspects.condition:
            required.append("condition")
        if aspects.relation:
            required.append("relation")
    elif aspects.target and aspects.target != aspects.subject:
        required.append("target")
    return required


def _matches_aspect(text: str, value: str | None) -> bool:
    if not value:
        return False
    return any(_contains_term(text, term) for term in _aspect_terms(value))


def _matches_relation(text: str, relation: str | None) -> bool:
    if not relation:
        return True
    patterns = {
        "suitability": [r"\bsuitable\b", r"\bapplicable\b", r"\bfit\s+for\b", r"\bcan\s+be\s+used\b", r"适合", r"适用", r"可用"],
        "support": [r"\bsupports?\b", r"\bshows?\b", r"\bdemonstrates?\b", r"\bevidence\b", r"\bproves?\b", r"支持", r"证明", r"证据", r"说明"],
        "migration": [r"\btransfer(?:red)?\b", r"\bmigrat(?:e|ion)\b", r"\bapply\s+to\b", r"\badapt(?:ed)?\b", r"迁移", r"套用", r"用于当前"],
        "relationship": [r"\brelationship\b", r"\bcorrelation\b", r"\beffect\b", r"\bimpact\b", r"关系", r"相关", r"影响"],
    }
    return any(re.search(pattern, text, re.I) for pattern in patterns.get(relation, [re.escape(relation)]))


def _contradicted_aspects(text: str, aspects: QuestionAspects) -> list[str]:
    contradicted = []
    if aspects.relation == "suitability" and _matches_relation(text, "suitability"):
        if re.search(r"\bnot\s+suitable\b|\bunsuitable\b|\bnot\s+applicable\b|涓嶉€傚悎|涓嶉€傜敤", text, re.I):
            contradicted.append("relation")
    return contradicted


def _raw_term_overlap(text: str, terms: list[str]) -> bool:
    return any(_contains_term(text, term) for term in terms if len(term) >= 3)


def _contains_term(text: str, term: str) -> bool:
    term = str(term or "").strip().lower()
    if not term:
        return False
    if re.search(r"[\u4e00-\u9fff]", term):
        return term in text.lower()
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.I))


def _aspect_terms(value: str) -> list[str]:
    normalized = str(value or "").lower()
    aliases = {
        "tap medium": ["tap", "tap medium", "tris-acetate-phosphate"],
        "dark culture": ["dark", "dark culture", "darkness", "without light", "no light", "黑暗", "暗培养", "避光"],
        "light condition": ["light", "illumination", "irradiance", "光照"],
        "current project": ["current project", "current experiment", "当前项目", "当前实验"],
        "chlorella": ["chlorella", "小球藻"],
        "chlamydomonas": ["chlamydomonas", "衣藻", "莱茵衣藻"],
    }
    terms = aliases.get(normalized, [normalized])
    if "_" in normalized:
        terms = [*terms, normalized.replace("_", " "), *normalized.split("_")]
    tokens = re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", normalized)
    return _merge_unique([*terms, *tokens])


def _evidence_text(item: EvidenceUnit) -> str:
    metadata_values = " ".join(str(value) for value in (item.metadata or {}).values() if value is not None)
    location_values = " ".join(str(value) for value in (item.location or {}).values() if value is not None)
    citation_values = " ".join(str(value) for value in (item.citation or {}).values() if value is not None)
    return " ".join(
        str(part or "")
        for part in [
            item.source_type,
            item.source_file,
            item.fact_type,
            item.entity,
            item.attribute,
            item.value,
            item.unit,
            item.text_span,
            metadata_values,
            location_values,
            citation_values,
        ]
    )


def _assessment_reason(
    supports_direct: bool,
    supports_background: bool,
    missing: list[str],
    contradicted: list[str],
) -> str:
    if contradicted:
        return "contradicts required aspects: " + ", ".join(contradicted)
    if supports_direct:
        return "covers required aspects for a direct answer"
    if supports_background:
        return "relevant background only; missing " + ", ".join(missing)
    return "not relevant enough; missing " + ", ".join(missing)


def _merge_unique(items: Iterable[str]) -> list[str]:
    merged = []
    for item in items:
        if item and item not in merged:
            merged.append(item)
    return merged
