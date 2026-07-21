import hashlib
from pathlib import Path


SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".xlsx", ".xls", ".csv", ".md", ".txt"}


def parse_source_metadata(path: str | Path) -> dict:
    source = Path(path).resolve()
    stem = source.stem
    parts = stem.split("__")
    doc_type = _doc_type_from_path(source)
    topic = stem
    version = None
    year = None
    language = None

    if len(parts) >= 4:
        doc_type = parts[0] or doc_type
        topic = parts[1] or topic
        version_or_year = parts[2]
        language = parts[3]
        if version_or_year.isdigit() and len(version_or_year) == 4:
            year = version_or_year
        else:
            version = version_or_year
    elif len(parts) >= 2:
        doc_type = parts[0] or doc_type
        topic = parts[1] or topic

    return {
        "source_path": str(source),
        "file_name": source.name,
        "doc_type": doc_type,
        "topic": topic,
        "version": version,
        "year": year,
        "language": language,
        "extension": source.suffix.lower(),
    }


def compute_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_supported_files(source: str | Path) -> list[Path]:
    root = Path(source)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _doc_type_from_path(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent in {"media_recipes", "media_recipe"}:
        return "media_recipe"
    if parent in {"manual", "manuals"}:
        return "manual"
    if parent in {"experiment_data", "experiments"}:
        return "experiment_data"
    if parent == "papers":
        return "paper"
    return "document"
