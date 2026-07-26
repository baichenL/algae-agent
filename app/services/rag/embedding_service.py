import hashlib
import math
import os
import re
import time
import threading
from dataclasses import dataclass

from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_FAKE_DIMENSION = 64
_probe_lock = threading.Lock()
_probe_status: dict | None = None

FAKE_SYNONYMS = {
    "pbr": ["photobioreactor", "reactor"],
    "photobioreactor": ["pbr", "reactor"],
    "od": ["od750", "optical", "density"],
    "od750": ["od", "optical", "density", "biomass"],
    "biomass": ["growth", "od750"],
    "tap": ["medium", "recipe", "acetate"],
    "sop": ["manual", "protocol", "procedure"],
    "protocol": ["sop", "manual", "procedure"],
}


@dataclass
class EmbeddingConfig:
    enabled: bool
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    vector_backend: str
    batch_size: int
    dimension: int | None = None


def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        enabled=_env_bool("RAG_EMBEDDING_ENABLED", default=True),
        provider=os.getenv("RAG_EMBEDDING_PROVIDER", "openai_compatible"),
        model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        base_url=os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        vector_backend=os.getenv("RAG_VECTOR_BACKEND", "sqlite_vec"),
        batch_size=max(int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "10")), 1),
        dimension=int(os.getenv("RAG_FAKE_EMBEDDING_DIMENSION", str(DEFAULT_FAKE_DIMENSION))),
    )


def embedding_runtime_status() -> dict:
    config = get_embedding_config()
    sqlite_vec_available = _sqlite_vec_available()
    if not config.enabled:
        status = "disabled"
    elif config.provider == "fake":
        status = "enabled_fake"
    elif not config.api_key:
        status = "disabled_missing_api_key"
    else:
        status = "enabled"
    backend = "sqlite_vec" if config.vector_backend == "sqlite_vec" and sqlite_vec_available else "sqlite_blob_fallback"
    degraded = bool(config.vector_backend == "sqlite_vec" and backend != "sqlite_vec")
    runtime = {
        "status": status,
        "provider": config.provider,
        "model": config.model,
        "vector_backend_requested": config.vector_backend,
        "vector_backend_active": backend,
        "sqlite_vec_available": sqlite_vec_available,
        "degraded": degraded,
        "degraded_reason": "sqlite_vec_unavailable_blob_cosine_fallback" if degraded else None,
    }
    if _probe_status:
        runtime.update(_probe_status)
        runtime["degraded"] = bool(runtime.get("degraded") or _probe_status.get("status") != "ready")
    return runtime


def probe_embedding_capability(*, force: bool = False) -> dict:
    global _probe_status
    config = get_embedding_config()
    if not config.enabled:
        return {"status": "disabled", "degraded": False}
    if config.provider == "fake":
        return {"status": "ready", "degraded": False, "probe": "fake"}
    enabled = _env_bool("RAG_EMBEDDING_PROBE_ENABLED", default=True)
    if not enabled:
        return {"status": "configured", "degraded": False, "probe": "disabled"}
    with _probe_lock:
        if _probe_status and not force:
            return dict(_probe_status)
        try:
            vectors = _embed_openai_compatible(["health"], config)
            if not vectors or not vectors[0]:
                raise RuntimeError("embedding_probe_returned_empty_vector")
            _probe_status = {"status": "ready", "degraded": False, "probe": "completed"}
        except Exception as exc:
            lowered = str(exc).casefold()
            code = (
                "embedding_model_configuration_error"
                if "model_not_found" in lowered or "does not exist" in lowered or "invalid_request_error" in lowered
                else "embedding_service_unavailable"
            )
            _probe_status = {
                "status": "degraded",
                "degraded": True,
                "probe": "completed",
                "degraded_reason": code,
                "provider": config.provider,
                "model": config.model,
            }
        return dict(_probe_status)


def reset_embedding_probe_status() -> None:
    global _probe_status
    with _probe_lock:
        _probe_status = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    config = get_embedding_config()
    if not config.enabled:
        return []
    cleaned = [text.strip() for text in texts if str(text or "").strip()]
    if not cleaned:
        return []
    if config.provider == "fake":
        return [_fake_embedding(text, config.dimension or DEFAULT_FAKE_DIMENSION) for text in cleaned]
    if not config.api_key:
        return []
    return _embed_openai_compatible(cleaned, config)


def embed_query(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else []


def build_chunk_embedding_records(chunks: list[dict]) -> list[dict]:
    texts = [_embedding_text(chunk) for chunk in chunks]
    vectors = embed_texts(texts)
    records = []
    for chunk, vector in zip(chunks, vectors):
        if not vector:
            continue
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "content_hash": chunk["content_hash"],
                "vector": vector,
            }
        )
    return records


def active_vector_backend() -> str:
    status = embedding_runtime_status()
    return status["vector_backend_active"]


def _embed_openai_compatible(texts: list[str], config: EmbeddingConfig) -> list[list[float]]:
    timeout_seconds = max(float(os.getenv("RAG_EMBEDDING_TIMEOUT_SECONDS", "20")), 1.0)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout_seconds, max_retries=0)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        for attempt in range(3):
            try:
                response = client.embeddings.create(model=config.model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors.extend([_normalize_vector(list(item.embedding)) for item in ordered])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
    return vectors


def _embedding_text(chunk: dict) -> str:
    if str(chunk.get("index_text") or "").strip():
        return str(chunk["index_text"]).strip()
    metadata = chunk.get("metadata") or {}
    return "\n".join(
        str(item or "")
        for item in [
            chunk.get("file_name"),
            chunk.get("title"),
            chunk.get("section"),
            metadata.get("topic"),
            metadata.get("organism"),
            metadata.get("medium"),
            metadata.get("task_type"),
            metadata.get("equipment"),
            metadata.get("measurement"),
            chunk.get("content"),
        ]
    )


def _fake_embedding(text: str, dimension: int) -> list[float]:
    vector = [0.0] * max(int(dimension or DEFAULT_FAKE_DIMENSION), 8)
    for token in _tokens(text):
        _add_token(vector, token, 1.0)
        for synonym in FAKE_SYNONYMS.get(token, []):
            _add_token(vector, synonym, 0.75)
    return _normalize_vector(vector)


def _add_token(vector: list[float], token: str, weight: float) -> None:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign * weight


def _tokens(text: str) -> list[str]:
    lowered = str(text or "").lower()
    return re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", lowered)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return []
    return [float(item) / norm for item in vector]


def _sqlite_vec_available() -> bool:
    try:
        import sqlite3
        import sqlite_vec  # noqa: F401

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute("CREATE VIRTUAL TABLE vec_health USING vec0(embedding float[2])")
        finally:
            conn.close()
        return True
    except Exception:
        return False


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
