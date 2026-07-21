from types import SimpleNamespace

from app.services.intent import llm_candidate_provider as provider
from app.services.intent.llm_candidate_provider import collect_llm_route_candidates


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content

    def create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _fake_client(content: str):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=_FakeCompletions(content),
        )
    )


def test_llm_provider_returns_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("HYBRID_ROUTER_LLM_ENABLED", raising=False)

    result = collect_llm_route_candidates(
        original_text="让 Chlorella_01 进入下一轮培养",
        normalized_text="让 Chlorella_01 进入下一轮培养",
        context_snapshot=SimpleNamespace(strains=[], pending_actions=[]),
    )

    assert result == []


def test_llm_provider_parses_valid_candidate(monkeypatch):
    monkeypatch.setenv("HYBRID_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setattr(
        provider,
        "client",
        _fake_client(
            """
            {
              "candidates": [
                {
                  "route_kind": "workflow",
                  "confidence": 0.84,
                  "speech_act": "command",
                  "risk_level": "high",
                  "evidence": ["open cultivation language implies a subculture workflow candidate"],
                  "missing_fields": [],
                  "mentioned_targets": ["Chlorella_01"]
                }
              ]
            }
            """
        ),
    )

    result = collect_llm_route_candidates(
        original_text="让 Chlorella_01 进入下一轮培养",
        normalized_text="让 Chlorella_01 进入下一轮培养",
        context_snapshot=SimpleNamespace(strains=[{"strain_id": "Chlorella_01"}], pending_actions=[]),
    )

    assert len(result) == 1
    assert result[0].route_kind.value == "workflow"
    assert result[0].source == "llm"
    assert result[0].mentioned_targets == ("Chlorella_01",)


def test_llm_provider_drops_invalid_json(monkeypatch):
    monkeypatch.setenv("HYBRID_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setattr(provider, "client", _fake_client("not json"))

    result = collect_llm_route_candidates(
        original_text="让 Chlorella_01 进入下一轮培养",
        normalized_text="让 Chlorella_01 进入下一轮培养",
        context_snapshot=SimpleNamespace(strains=[], pending_actions=[]),
    )

    assert result == []


def test_llm_provider_drops_unknown_route(monkeypatch):
    monkeypatch.setenv("HYBRID_ROUTER_LLM_ENABLED", "true")
    monkeypatch.setattr(
        provider,
        "client",
        _fake_client('{"candidates": [{"route_kind": "execute_now", "confidence": 0.99}]}'),
    )

    result = collect_llm_route_candidates(
        original_text="让 Chlorella_01 进入下一轮培养",
        normalized_text="让 Chlorella_01 进入下一轮培养",
        context_snapshot=SimpleNamespace(strains=[], pending_actions=[]),
    )

    assert result == []


def test_shadow_mode_records_intent_frame_without_changing_candidates(monkeypatch):
    monkeypatch.setenv("INTENT_ROUTER_MODE", "shadow")
    monkeypatch.setattr(
        provider,
        "client",
        _fake_client(
            '{"schema_version":"intent-frame/v1","goal":"查询品系状态",'
            '"primary_route":"lab_query","alternatives":[],"speech_act":"query",'
            '"requested_effect":"read","entity_mentions":["Chlorella_01"],'
            '"slots":{},"missing_slots":[],"suggested_skill":null,'
            '"suggested_tools":["list_algae_strains"],"confidence":0.92,'
            '"evidence":["状态查询"]}'
        ),
    )
    result = collect_llm_route_candidates(
        original_text="看看 Chlorella_01",
        normalized_text="看看 Chlorella_01",
        context_snapshot=SimpleNamespace(strains=[], pending_actions=[]),
    )
    diagnostics = provider.current_router_diagnostics()
    assert result == []
    assert diagnostics["mode"] == "shadow"
    assert diagnostics["intent_frame"]["requested_effect"] == "read"


def test_intent_frame_rejects_model_supplied_risk(monkeypatch):
    monkeypatch.setenv("INTENT_ROUTER_MODE", "hybrid")
    monkeypatch.setattr(
        provider,
        "client",
        _fake_client(
            '{"schema_version":"intent-frame/v1","goal":"删除品系",'
            '"primary_route":"write_action","alternatives":[],"speech_act":"command",'
            '"requested_effect":"propose","entity_mentions":[],"slots":{},'
            '"missing_slots":[],"suggested_skill":null,"suggested_tools":[],'
            '"confidence":0.99,"evidence":[],"risk_level":"low"}'
        ),
    )
    result = collect_llm_route_candidates(
        original_text="删除它",
        normalized_text="删除它",
        context_snapshot=SimpleNamespace(strains=[], pending_actions=[]),
    )
    assert result == []
    assert provider.current_router_diagnostics()["status"] == "invalid_json"
