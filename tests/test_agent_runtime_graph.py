from app.services.agent_runtime import build_agent_runtime_graph


def test_agent_runtime_graph_exposes_core_orchestration_nodes():
    graph = build_agent_runtime_graph().get_graph()

    for node in (
        "InitializeRun",
        "BeginStep",
        "BuildContext",
        "DecideAction",
        "EvaluatePolicy",
        "ExecuteAction",
        "Observe",
        "Replan",
        "FinishRun",
    ):
        assert node in graph.nodes
