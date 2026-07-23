from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field


SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

ALIASES: dict[str, tuple[str, ...]] = {
    "paper": ("论文", "文献", "paper", "literature", "article", "study"),
    "manual": ("实验手册", "手册", "标准操作规程", "操作规程", "sop", "protocol", "manual"),
    "recipe": ("培养基", "配方", "组分", "成分", "medium", "recipe", "composition"),
    "experiment_data": ("实验数据", "数据表", "生长曲线", "experiment data", "growth curve", "dataset"),
    "photobioreactor": ("光生物反应器", "photobioreactor", "pbr"),
    "biomass": ("生物量", "生长", "biomass", "growth", "od750", "optical density"),
    "procedure": ("步骤", "操作", "流程", "procedure", "step", "operation"),
    "comparison": ("比较", "对比", "区别", "差异", "compare", "versus", "difference"),
    "relationship": ("关系", "影响", "相关", "为什么", "relationship", "effect", "correlation", "why"),
}

DOC_TYPE_BY_ALIAS = {
    "paper": "paper",
    "manual": "manual",
    "recipe": "media_recipe",
    "experiment_data": "experiment_data",
}

SPARSE_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "paper": ("paper", "literature", "论文", "文献"),
    "manual": ("manual", "sop", "protocol", "实验手册", "操作规程"),
    "recipe": ("medium", "recipe", "composition", "培养基", "配方"),
    "experiment_data": ("experiment", "dataset", "实验数据", "数据表"),
    "photobioreactor": ("photobioreactor", "pbr", "光生物反应器"),
    "biomass": ("biomass", "growth", "od750", "生物量", "生长"),
    "tap": ("tap", "tap_medium", "tris", "acetate", "phosphate"),
    "k2hpo4": ("k2hpo4", "phosphate", "磷酸盐"),
}


@dataclass(frozen=True)
class NormalizedRagQuery:
    original: str
    normalized_query: str
    sparse_terms: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    intent: str = "simple"
    exact_terms: list[str] = field(default_factory=list)

    @property
    def normalized(self) -> str:
        return self.normalized_query

    @property
    def query_type(self) -> str:
        return self.intent

    @property
    def sparse_tokens(self) -> list[str]:
        return self.sparse_terms

    @property
    def sparse_query(self) -> str:
        return " ".join(self.sparse_terms)

    @property
    def dense_query(self) -> str:
        return self.normalized_query

    def model_dump(self) -> dict:
        return asdict(self)


def normalize_rag_query(question: str) -> NormalizedRagQuery:
    original = str(question or "").strip()
    normalized = _normalize_text(original)
    concepts = [name for name, aliases in ALIASES.items() if _contains_alias(normalized, aliases)]
    doc_types = _infer_doc_types(concepts)
    exact_terms = _exact_terms(normalized)
    terms = _sparse_tokens(normalized, concepts, exact_terms)
    return NormalizedRagQuery(
        original=original,
        normalized_query=normalized,
        sparse_terms=terms,
        concepts=concepts,
        doc_types=doc_types,
        intent="complex" if _is_complex(normalized, concepts) else "simple",
        exact_terms=exact_terms,
    )


def infer_rag_doc_types(question: str) -> list[str]:
    return normalize_rag_query(question).doc_types


def sparse_query_tokens(question: str, limit: int = 24) -> list[str]:
    return normalize_rag_query(question).sparse_terms[: max(int(limit), 1)]


def is_complex_rag_query(question: str) -> bool:
    return normalize_rag_query(question).intent == "complex"


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).translate(SUBSCRIPT_MAP).translate(SUPERSCRIPT_MAP).casefold()
    replacements = {
        "µ": "u",
        "μ": "u",
        "·": " ",
        "℃": " c ",
        "°c": " c ",
        "−": "-",
        "–": "-",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return re.sub(r"\s+", " ", value).strip()


def _contains_alias(text: str, aliases: tuple[str, ...]) -> bool:
    return any(alias.casefold() in text for alias in aliases)


def _infer_doc_types(concepts: list[str]) -> list[str]:
    result: list[str] = []
    for concept in concepts:
        doc_type = DOC_TYPE_BY_ALIAS.get(concept)
        if doc_type and doc_type not in result:
            result.append(doc_type)
    return result


def _exact_terms(text: str) -> list[str]:
    patterns = (
        r"\b[a-z]{1,4}\d+[a-z0-9-]*\b",
        r"\bod\s*750\b",
        r"\bph\s*\d+(?:\.\d+)?\b",
        r"\b\d+(?:\.\d+)?\s*(?:mg/l|g/l|ml|umol|nm|h|min|c)\b",
    )
    terms: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.I):
            token = re.sub(r"\s+", "", str(match)).casefold()
            if token and token not in terms:
                terms.append(token)
    return terms


def _sparse_tokens(text: str, concepts: list[str], exact_terms: list[str]) -> list[str]:
    raw = re.findall(r"[a-z0-9_+.-]+|[\u3400-\u9fff]{2,}", text)
    tokens: list[str] = []
    for token in [*exact_terms, *raw]:
        cleaned = token.strip("_.-+")
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
        if re.fullmatch(r"[\u3400-\u9fff]{4,}", cleaned):
            for index in range(len(cleaned) - 1):
                gram = cleaned[index : index + 2]
                if gram not in tokens:
                    tokens.append(gram)
    for concept in concepts:
        for expanded in SPARSE_EXPANSIONS.get(concept, ()):
            if expanded not in tokens:
                tokens.append(expanded)
    for trigger, expansions in SPARSE_EXPANSIONS.items():
        if trigger in text:
            for expanded in expansions:
                if expanded not in tokens:
                    tokens.append(expanded)
    return tokens


def _is_complex(text: str, concepts: list[str]) -> bool:
    if any(item in concepts for item in ("comparison", "relationship")):
        return True
    separators = len(re.findall(r"[?？]|以及|并且|同时|\band\b|versus|\bvs\b", text, re.I))
    source_concepts = sum(1 for item in concepts if item in DOC_TYPE_BY_ALIAS)
    return separators >= 2 or source_concepts >= 2
