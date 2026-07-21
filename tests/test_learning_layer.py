from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import router as api_router
from app.core.db.connection import connect
from app.services.agent_runtime.events import record_run_event, start_run
from app.services.context.context_builder import build_context_snapshot
from app.services.learning.curator import run_learning_curator
from app.services.learning.reviewer import run_post_run_learning_review
from app.services.learning.selector import select_learning_context
from app.services.learning.store import (
    list_agent_learning_reviews,
    list_agent_memories,
    list_agent_skills,
    seed_builtin_skills,
    upsert_memory,
)


def _client():
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


def test_learning_selector_respects_budget_and_selects_builtin_skill(isolated_sqlite_db):
    seed_builtin_skills()
    upsert_memory(
        scope="agent",
        content="Current strain status questions should use DB-first lab query routing before RAG.",
        source_run_id="seed-run",
        confidence=0.9,
    )

    selection = select_learning_context(
        session_id="learn-s1",
        message="show current strain generation status",
        agent_run_id="learn-run-1",
        memory_budget=120,
        skill_budget=500,
    )

    assert selection.selected_agent_memories
    assert sum(len(item["content"]) for item in selection.selected_agent_memories) <= 120
    assert any(item["name"] == "db_first_lab_query" for item in selection.selected_skill_summaries)


def test_context_snapshot_includes_learning_sections(isolated_sqlite_db):
    upsert_memory(
        scope="agent",
        content="Pending approval is not executed lab state.",
        source_run_id="seed-run",
        confidence=0.8,
    )

    snapshot = build_context_snapshot("learn-context-s1")

    data = snapshot.to_dict()
    assert "selected_agent_memories" in data
    assert "selected_skill_summaries" in data
    prompt = snapshot.to_prompt_facts()
    assert "selected_skill_summaries" in prompt or snapshot.selected_skill_summaries == []


def test_learning_review_only_materializes_for_trigger_events(isolated_sqlite_db):
    start_run("learn-s1", "run workflow", "learn-review-run-1")
    skipped = run_post_run_learning_review(agent_run_id="learn-review-run-1", session_id="learn-s1")
    assert skipped["status"] == "skipped"

    record_run_event(
        "learn-review-run-1",
        session_id="learn-s1",
        event_type="agent_action_blocked",
        layer="agent_runtime",
        payload={
            "action": {"action_name": "trigger_hardware"},
            "policy": {"category": "block_forbidden", "reason": "blocked"},
        },
    )

    result = run_post_run_learning_review(agent_run_id="learn-review-run-1", session_id="learn-s1")

    assert result["status"] == "accepted"
    assert result["applied_memory_id"]
    reviews = list_agent_learning_reviews(agent_run_id="learn-review-run-1")
    assert reviews[0]["candidate_type"] == "memory_candidate"
    assert list_agent_memories(scope="agent", status="active")


def test_curator_marks_low_success_skill_stale(isolated_sqlite_db):
    seed_builtin_skills()
    skill = next(item for item in list_agent_skills(status="active") if item["name"] == "db_first_lab_query")
    with connect() as conn:
        conn.execute(
            "UPDATE agent_skills SET usage_count = 5, success_count = 1, failure_count = 4 WHERE id = ?",
            (skill["id"],),
        )
        conn.commit()

    result = run_learning_curator(agent_run_id="curator-run", session_id="curator-s1")

    assert result["decision_count"] >= 1
    stale = list_agent_skills(status="stale")
    assert any(item["name"] == "db_first_lab_query" for item in stale)


def test_learning_management_api_is_read_only_surface(isolated_sqlite_db):
    seed_builtin_skills()
    upsert_memory(
        scope="agent",
        content="Use DB facts for current lab state.",
        source_run_id="seed-run",
        confidence=0.9,
    )

    memories = _client().get("/api/v1/agent/memories")
    skills = _client().get("/api/v1/agent/skills")
    reviews = _client().get("/api/v1/agent/learning/reviews")

    assert memories.status_code == 200
    assert skills.status_code == 200
    assert reviews.status_code == 200
    assert memories.json()["memories"]
    assert skills.json()["skills"]
