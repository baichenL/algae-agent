from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v2 import router
from app.core.security import ApiPrincipal
from app.core.session_security import require_session_viewer


def _report() -> dict:
    return {
        "schema_version": "1.0",
        "report_id": "eval-test",
        "generated_at": "2026-07-30T00:00:00Z",
        "git_commit": "abc",
        "dataset_versions": {},
        "environment": {"embedding_backend": "fake", "api_key": "secret"},
        "suites": [{"key": "agent", "system_prompt": "private", "metrics": []}],
        "comparisons": [],
    }


def _viewer_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_session_viewer] = lambda: ApiPrincipal(
        name="evaluation-viewer",
        role="viewer",
        source="test",
    )
    return TestClient(app)


def test_viewer_reads_latest_and_download_is_redacted(tmp_path, monkeypatch):
    (tmp_path / "latest.json").write_text(json.dumps(_report()), encoding="utf-8")
    (tmp_path / "eval-test.json").write_text(json.dumps(_report()), encoding="utf-8")
    monkeypatch.setenv("ALGAE_EVAL_REPORT_DIR", str(tmp_path))
    client = _viewer_client()

    latest = client.get("/api/v2/evaluations/latest")
    assert latest.status_code == 200
    rendered = json.dumps(latest.json())
    assert "secret" not in rendered
    assert "private" not in rendered

    download = client.get("/api/v2/evaluations/eval-test/download")
    assert download.status_code == 200
    assert "attachment" in download.headers["content-disposition"]
    assert "secret" not in download.text
    assert "private" not in download.text


def test_missing_invalid_and_empty_reports_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGAE_EVAL_REPORT_DIR", str(tmp_path))
    client = _viewer_client()
    assert client.get("/api/v2/evaluations/latest").status_code == 404
    assert client.get("/api/v2/evaluations/missing").status_code == 404
    assert client.get("/api/v2/evaluations/%2E%2E").status_code in {400, 404}
