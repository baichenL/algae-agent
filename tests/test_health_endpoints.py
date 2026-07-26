from fastapi.testclient import TestClient

from app import main


def test_liveness_probe_is_independent_of_dependencies():
    response = TestClient(main.app).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_probe_accepts_disabled_reranker(monkeypatch, isolated_sqlite_db):
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "false")

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 200
    checks = response.json()["checks"]
    assert checks["database"] == {"status": "ready"}
    assert checks["reranker"] == {"status": "disabled"}
    assert checks["models"]["status"] == "ready"
    assert checks["embedding"]["status"] in {"ready", "disabled", "degraded", "enabled_fake"}


def test_readiness_probe_rejects_degraded_reranker(monkeypatch, isolated_sqlite_db):
    monkeypatch.setenv("RAG_RERANKER_ENABLED", "true")
    monkeypatch.setattr(
        main,
        "reranker_runtime_status",
        lambda: {"status": "degraded", "degraded": True, "reason": "model_missing"},
    )

    response = TestClient(main.app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["database"] == {"status": "ready"}
