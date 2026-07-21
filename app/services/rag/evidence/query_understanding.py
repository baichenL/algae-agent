import re

from app.models.rag_evidence_schema import QueryFrame, QuestionAspects


JUDGEMENT_PATTERNS = {
    "suitability": r"suitable|applicable|fit for|can .*use|适合|适用|能否使用|可以使用|可用",
    "support": r"support|prove|evidence|show|支持|证明|证据|是否说明|能否说明",
    "migration": r"transfer|migrate|apply to current|迁移|套用|迁移到|用于当前",
    "relationship": r"effect|correlation|relationship|impact|影响|关系|相关",
}


CONDITION_PATTERNS = [
    (r"dark(?:ness)?|dark culture|without light|no light|黑暗|暗培养|避光", "dark culture"),
    (r"light|illumination|irradiance|光照", "light condition"),
    (r"chlorella|小球藻", "Chlorella"),
    (r"chlamydomonas|衣藻|莱茵衣藻", "Chlamydomonas"),
    (r"current project|current experiment|当前项目|当前实验", "current project"),
    (r"pH", "pH"),
    (r"CO2|CO₂", "CO2"),
    (r"temperature|温度", "temperature"),
]


SUBJECT_PATTERNS = [
    (r"\bTAP\b|TAP medium|培养基", "TAP medium"),
    (r"Chlorella[_\-\s]?\d*|小球藻", "Chlorella"),
    (r"contamination|polluted culture|污染", "contamination"),
    (r"microalgae cultivation|微藻培养", "microalgae cultivation"),
]


def parse_question_aspects(question: str, frame: QueryFrame | None = None) -> QuestionAspects:
    text = question or ""
    lower = text.lower()
    relation = _relation(text, frame)
    condition = _first_match(text, CONDITION_PATTERNS) or _generic_condition(text)
    subject = _subject(text, frame)
    target = _target(text, frame, condition)
    expected = _expected_answer_type(text, frame, relation)
    required = _required_support(expected, relation, frame)
    raw_terms = _raw_terms(text, subject, condition, relation, target)
    return QuestionAspects(
        subject=subject,
        condition=condition,
        relation=relation,
        target=target,
        expected_answer_type=expected,
        required_support=required,
        raw_terms=raw_terms,
    )


def _relation(text: str, frame: QueryFrame | None) -> str | None:
    if frame and frame.question_type == "suitability":
        return "suitability"
    if frame and frame.evidence_requirement == "structured_data_analysis":
        return "data_support"
    for label, pattern in JUDGEMENT_PATTERNS.items():
        if re.search(pattern, text, re.I):
            return label
    return frame.target_attribute if frame and frame.target_attribute else None


def _subject(text: str, frame: QueryFrame | None) -> str | None:
    if frame and frame.target_entity:
        return frame.target_entity
    return _first_match(text, SUBJECT_PATTERNS)


def _target(text: str, frame: QueryFrame | None, condition: str | None) -> str | None:
    if frame and frame.target_value:
        return frame.target_value
    if condition:
        return condition
    for pattern, label in CONDITION_PATTERNS:
        if re.search(pattern, text, re.I):
            return label
    return None


def _generic_condition(text: str) -> str | None:
    patterns = [
        r"\bsupports?\s+([a-z][a-z0-9_\- ]{2,60})(?:\?|\.|,|$)",
        r"\b(?:evidence|proof)\s+(?:for|of)\s+([a-z][a-z0-9_\- ]{2,60})(?:\?|\.|,|$)",
        r"\b(?:for|under|in|during)\s+([a-z][a-z0-9_\- ]{2,60})(?:\?|\.|,|$)",
        r"\b(?:use|used|using)\s+(?:it\s+)?(?:for|under|in)\s+([a-z][a-z0-9_\- ]{2,60})(?:\?|\.|,|$)",
        r"(?:适合|适用|用于|在)\s*([\u4e00-\u9fffA-Za-z0-9_\- ]{2,30})(?:吗|？|\?|。|$)",
    ]
    stop_words = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "current",
        "any",
        "all",
        "medium",
        "protocol",
        "documentation",
        "document",
    }
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        condition = match.group(1).strip(" ,.;:，。？?")
        if re.fullmatch(r"(?:什么|哪些|哪个|哪种|何种)(?:步骤|场景|条件|方法|用途)?", condition):
            continue
        words = [word for word in condition.split() if word.lower() not in stop_words]
        if words:
            condition = " ".join(words)
        if condition and len(condition) <= 80:
            return condition
    return None


def _expected_answer_type(text: str, frame: QueryFrame | None, relation: str | None) -> str:
    if frame and frame.answer_shape in {"yes_no", "value", "list", "procedure", "overview"}:
        return frame.answer_shape
    if relation in {"suitability", "support", "migration"}:
        return "yes_no"
    if re.search(r"\bwhether\b|鏄惁|鏈夋病鏈墊鑳戒笉鑳絴\bhas\b|\bdoes\b", text, re.I):
        return "yes_no"
    return "general"


def _required_support(expected: str, relation: str | None, frame: QueryFrame | None) -> str:
    if relation in {"suitability", "support", "migration"}:
        return "explicit_statement"
    if relation == "data_support":
        return "structured_data_analysis"
    if frame and frame.evidence_requirement in {
        "structured_schema",
        "structured_data_analysis",
        "structured_overview",
        "explicit_statement",
    }:
        return frame.evidence_requirement
    if frame and frame.target_entity == "paper":
        return "relevant_context"
    if expected == "yes_no":
        return "direct_statement"
    return "relevant_context"


def _first_match(text: str, patterns: list[tuple[str, str]]) -> str | None:
    for pattern, label in patterns:
        if re.search(pattern, text, re.I):
            return label
    return None


def _raw_terms(*parts: str | None) -> list[str]:
    terms = []
    for part in parts:
        for token in re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", str(part or "").lower()):
            token = token.strip("_-+")
            if token and token not in terms:
                terms.append(token)
    return terms
