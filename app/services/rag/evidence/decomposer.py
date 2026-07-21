import re

from app.models.rag_evidence_schema import AtomicQuestion


QUESTION_START_RE = re.compile(
    r"(?:TAP\s*)?(?:recipe|medium|composition|component|phosphate|trace|Hutner|"
    r"NH4Cl|MgSO4|K2HPO4|KH2PO4|NaCl)[^?？。;\n]*",
    re.I,
)


def decompose_question(question: str) -> list[AtomicQuestion]:
    text = (question or "").strip()
    if not text:
        return []

    parts = _split_question_text(text)
    if len(parts) <= 1:
        return [AtomicQuestion(question_id="q1", text=text, original_span=text)]

    context = _infer_context(parts)
    atomics = []
    for index, part in enumerate(parts, start=1):
        atomic_text = _apply_context(part, context)
        atomics.append(
            AtomicQuestion(
                question_id=f"q{index}",
                text=atomic_text,
                inherited_context=context if atomic_text != part else None,
                original_span=part,
            )
        )
    return atomics


def _split_question_text(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    rough_parts = [
        part.strip(" ,;?？。")
        for part in re.split(r"[;；?？。]\s*|[\r\n]+", normalized)
        if part.strip(" ,;?？。")
    ]
    parts: list[str] = []
    for part in rough_parts:
        parts.extend(_split_compound_part(part))
    return [part for part in parts if part]


def _split_compound_part(part: str) -> list[str]:
    matches = list(QUESTION_START_RE.finditer(part))
    if len(matches) <= 1:
        return [part.strip()]
    pieces = []
    for match in matches:
        piece = match.group(0).strip(" ,;?？。")
        if piece:
            pieces.append(piece)
    prefix = part[: matches[0].start()].strip(" ,;?？。")
    if prefix and pieces:
        pieces[0] = f"{prefix}{pieces[0]}"
    return pieces


def _infer_context(parts: list[str]) -> str | None:
    for part in parts:
        match = re.search(r"(TAP\s*(?:recipe|medium|composition)?)", part, re.I)
        if match:
            context = match.group(1).strip()
            return context if context.endswith(" ") else f"{context} "
    return None


def _apply_context(part: str, context: str | None) -> str:
    if not context:
        return part
    if re.search(r"TAP|recipe|medium|composition", part, re.I):
        return part
    if re.search(r"NaCl|NH4Cl|NH₄Cl|MgSO4|MgSO₄|K2HPO4|K₂HPO₄|KH2PO4|KH₂PO₄|phosphate|trace|Hutner", part, re.I):
        return f"{context}{part}"
    return part
