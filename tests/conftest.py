import os
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("ALGAE_AUTH_MODE", "test")
os.environ["RAG_SEMANTIC_PARSER"] = "local"
os.environ["RAG_EMBEDDING_PROVIDER"] = "fake"


workflow_stub = types.ModuleType("app.tools.workflow_tool_handlers")


def handle_subculture_tool(function_args):
    from app.services.protocols.pending_payload import build_protocol_response_fields
    from app.services.strains import strain_service

    pending_id = strain_service.create_pending_workflow_subculture(
        function_args.get("strain_id", "Chlorella_01"),
        requester="agent_tool",
        session_id=function_args.get("session_id"),
        agent_run_id=function_args.get("agent_run_id"),
        source_message=function_args.get("source_message"),
    )
    pending = strain_service.get_pending_action(pending_id)
    protocol_fields = build_protocol_response_fields((pending or {}).get("payload") or {})
    return {
        "action": "workflow_subculture",
        "status": "pending",
        "pending_id": pending_id,
        "response_payload": {
            "action": "workflow_subculture",
            "status": "pending",
            "pending_id": pending_id,
            "strain_id": function_args.get("strain_id", "Chlorella_01"),
            "require_confirmation": True,
            "requires_approval": True,
            **protocol_fields,
        },
        "memory_text": "workflow pending stub",
    }


workflow_stub.handle_subculture_tool = handle_subculture_tool
sys.modules.setdefault("app.tools.workflow_tool_handlers", workflow_stub)


def context(strains=None, pending_actions=None):
    return SimpleNamespace(
        session_id="test-session",
        strains=strains or [],
        pending_actions=pending_actions or [],
        to_prompt_facts=lambda: "test context",
    )


@pytest.fixture
def api_client():
    from app.api.chat import router as chat_router

    app = FastAPI()
    app.include_router(chat_router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture
def isolated_pending_form_state(tmp_path, monkeypatch):
    from app.services.chat import pending_form_state as pfs

    state_dir = tmp_path / "pending_form_state"
    monkeypatch.setattr(pfs, "STATE_DIR", str(state_dir))
    return state_dir


@pytest.fixture
def isolated_session_memory(monkeypatch):
    from app.services.chat import chat_service

    memory = {}

    def get_memory(session_id):
        return memory.setdefault(session_id, [{"role": "system", "content": "test"}])

    def save_memory(session_id, conversation_history):
        memory[session_id] = conversation_history

    monkeypatch.setattr(chat_service, "get_session_memory", get_memory)
    monkeypatch.setattr(chat_service, "save_session_memory", save_memory)
    return memory


@pytest.fixture
def isolated_sqlite_db(tmp_path, monkeypatch):
    from app.core.db import connection, control_plane, experiments, pending_actions, rag, reminder_cycles, schema, scientific, strains, workflow_runs
    from app.core import database
    from app.core.workspaces import WorkspaceContext, workspace_scope

    db_path = tmp_path / "algae_test.sqlite3"
    db_path_str = str(db_path)
    monkeypatch.setattr(connection, "DB_PATH", db_path_str)
    monkeypatch.setattr(control_plane, "DB_PATH", db_path_str)
    monkeypatch.setattr(schema, "DB_PATH", db_path_str)
    monkeypatch.setattr(pending_actions, "DB_PATH", db_path_str)
    monkeypatch.setattr(reminder_cycles, "DB_PATH", db_path_str)
    monkeypatch.setattr(rag, "DB_PATH", db_path_str)
    monkeypatch.setattr(experiments, "DB_PATH", db_path_str)
    monkeypatch.setattr(strains, "DB_PATH", db_path_str)
    monkeypatch.setattr(workflow_runs, "DB_PATH", db_path_str)
    monkeypatch.setattr(scientific, "DB_PATH", db_path_str)
    monkeypatch.setattr(database, "DB_PATH", db_path_str)

    # The developer .env may use split control/domain storage.  Unit tests need a
    # single isolated database regardless of that machine-level setting; otherwise
    # schema.init_db() initializes the configured workspace paths while assertions
    # read the temporary path and tests become order/environment dependent.
    test_workspace = WorkspaceContext(
        # Runtime fixtures and safety envelopes intentionally use the production
        # default workspace identity while the physical database remains isolated.
        # Changing the logical identity here would test cross-workspace denial
        # instead of the behavior each unit test declares.
        id="shared",
        name="pytest",
        db_path=db_path_str,
        storage_mode="legacy",
    )
    with workspace_scope(test_workspace):
        schema.init_db()
        yield db_path


@pytest.fixture
def isolated_chat_side_effects(monkeypatch):
    from app.services.chat import chat_service

    monkeypatch.setenv("RAG_SEMANTIC_PARSER", "local")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setattr(chat_service, "append_decision_event", lambda event: None)

    async def fail_llm_path(*args, **kwargs):
        raise AssertionError("LLM should not be called in /api/v1/chat integration tests")

    monkeypatch.setattr(chat_service, "_handle_llm_or_tool_path", fail_llm_path)
