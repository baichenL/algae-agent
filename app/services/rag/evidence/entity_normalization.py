import re
import unicodedata


FORMULA_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][a-z]?\d*)+(?:[·.]\d*(?:[A-Z][a-z]?\d*)+)*(?![A-Za-z0-9])"
)


def normalize_unicode_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return normalized.replace("•", "·").replace("⋅", "·").replace("∙", "·")


def canonicalize_chemical_formula(text: str) -> str:
    normalized = normalize_unicode_text(text)
    normalized = normalized.replace("·", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[^A-Za-z0-9()+\-]", "", normalized)
    return normalized.casefold()


def extract_formula_candidates(text: str) -> list[str]:
    normalized = normalize_unicode_text(text)
    result = []
    for match in FORMULA_TOKEN_RE.finditer(normalized):
        value = match.group(0)
        if any(char.isdigit() for char in value) or len(value) <= 6:
            if value not in result:
                result.append(value)
    return result


def canonicalize_entity_text(text: str) -> str:
    normalized = normalize_unicode_text(text).casefold()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized)
