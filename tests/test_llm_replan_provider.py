from types import SimpleNamespace

from app.services.agent_runtime import llm_replan_provider as provider
from app.services.agent_runtime.llm_replan_provider import collect_llm_replan_suggestions
from app.services.agent_runtime.state import AgentObservation, AgentRunState


def _state() -> AgentRunState:
    return AgentRunState(
        agent_run_id="llm-replan-provider-run",
        session_id="llm-replan-provider-session",
        user_message="test",
        conversation_history=[],
    )


def _observation() -> AgentObservation:
    return AgentObservation(
        status="success",
        action="query_due_subculture",
        route_kind="lab_query",
        output={"action": "query_due_subculture", "strains": [{"strain_id": "Chlorella_01"}]},
        natural_reply="one due strain",
    )


def test_llm_replan_provider_returns_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("AGENT_REPLAN_LLM_ENABLED", raising=False)

    result = collect_llm_replan_suggestions(_state(), _observation())

    assert result.assessment is None
    assert result.candidates == ()


def test_llm_replan_provider_filters_forbidden_and_unknown_actions(monkeypatch):
    monkeypatch.setenv("AGENT_REPLAN_LLM_ENABLED", "1")

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="""
                            {
                              "assessment": {
                                "category": "success",
                                "confidence": 0.8,
                                "explanation": "one target"
                              },
                              "candidates": [
                                {
                                  "action_type": "trigger_workflow",
                                  "confidence": 1,
                                  "reason": "forbidden direct execution",
                                  "target_ids": ["Chlorella_01"]
                                },
                                {
                                  "action_type": "modify_database",
                                  "confidence": 1,
                                  "reason": "forbidden database write"
                                },
                                {
                                  "action_type": "propose_workflow_approval",
                                  "confidence": 0.9,
                                  "reason": "approval only",
                                  "target_ids": ["Chlorella_01"]
                                }
                              ]
                            }
                            """
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(provider, "client", fake_client)

    result = collect_llm_replan_suggestions(_state(), _observation())

    assert result.assessment.category == "success"
    assert len(result.candidates) == 1
    assert result.candidates[0].action_type == "propose_workflow_approval"
