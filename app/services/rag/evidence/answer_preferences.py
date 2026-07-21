import re

from app.models.rag_evidence_schema import GroundedAnswer


def apply_answer_preferences(answer: GroundedAnswer, semantic_subquery: dict | None) -> GroundedAnswer:
    if not semantic_subquery:
        return answer
    requested = semantic_subquery.get("requested_output") or {}
    max_points = requested.get("max_points")
    if max_points and max_points > 0:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", answer.direct_answer) if block.strip()]
        if len(blocks) >= max_points:
            selected = blocks[:max_points]
            answer.direct_answer = "\n".join(
                f"{index}. {block}" for index, block in enumerate(selected, start=1)
            )
    if requested.get("concise") and len(answer.direct_answer) > 700:
        paragraphs = [item.strip() for item in answer.direct_answer.split("\n\n") if item.strip()]
        answer.direct_answer = "\n\n".join(paragraphs[:3])
    return answer
