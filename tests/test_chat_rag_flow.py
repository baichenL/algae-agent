from pathlib import Path

import pytest

from app.core import database
from app.services.rag.ingestion.index_store import ingest_file


def _post_chat(client, message, session_id):
    return client.post(
        "/api/v1/chat",
        json={"message": message, "session_id": session_id},
    )


def test_chat_routes_tap_question_to_rag(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    source = tmp_path / "media_recipe__tap_medium__v1__zh.txt"
    source.write_text(
        "TAP 培养基包含 Tris, NH4Cl, MgSO4, CaCl2, phosphate buffer, trace elements.",
        encoding="utf-8",
    )
    ingest_file(source)

    response = _post_chat(api_client, "TAP 培养基包含哪些组分？", "rag-chat-tap")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    assert body["agent_output"]["answer"]["conclusion"]
    assert body["agent_output"]["citations"]
    assert body["agent_output"]["citations"][0]["file_name"] == source.name
    assert "事实：" in body["natural_reply"]
    assert "解释：" in body["natural_reply"]
    assert "建议：" in body["natural_reply"]
    assert "工作液" in body["natural_reply"] or "TAP" in body["natural_reply"]
    assert "结论" in body["natural_reply"]
    assert "依据" in body["natural_reply"]
    assert "不确定性" in body["natural_reply"]


def test_chat_tap_question_prefers_recipe_citations_over_manual_noise(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    recipe = Path("data/raw/media_recipes/media_recipe__tap_medium__v1__zh.docx")
    manual = Path("data/raw/manual/manual__microalgae_laboratory_manual__v1__en.pdf")
    if not recipe.exists() or not manual.exists():
        pytest.skip("local TAP recipe or manual PDF is not available")
    ingest_file(recipe)
    ingest_file(manual)

    response = _post_chat(api_client, "TAP 培养基包含哪些组分？", "rag-chat-tap-real")

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    citations = body["agent_output"]["citations"]
    assert citations
    assert {item["doc_type"] for item in citations} == {"media_recipe"}
    reply = body["natural_reply"]
    assert "工作液配制" in reply
    assert "母液1" in reply
    assert "母液2" in reply
    assert "母液3" in reply
    assert "K₂HPO₄" in reply
    assert "NH₄Cl" in reply
    assert "MgSO₄·7H₂O" in reply


def test_chat_tap_question_without_space_still_retrieves_recipe(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    recipe = Path("data/raw/media_recipes/media_recipe__tap_medium__v1__zh.docx")
    manual = Path("data/raw/manual/manual__microalgae_laboratory_manual__v1__en.pdf")
    if not recipe.exists() or not manual.exists():
        pytest.skip("local TAP recipe or manual PDF is not available")
    ingest_file(recipe)
    ingest_file(manual)

    response = _post_chat(api_client, "TAP培养基成分是什么", "rag-chat-tap-no-space")

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    citations = body["agent_output"]["citations"]
    assert citations
    assert {item["doc_type"] for item in citations} == {"media_recipe"}
    reply = body["natural_reply"]
    assert "TAP（Tris-Acetate-Phosphate）" in reply
    assert "K₂HPO₄" in reply
    assert "当前没有可引用依据" not in reply


def test_chat_tap_phosphate_question_returns_focused_answer(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    recipe = Path("data/raw/media_recipes/media_recipe__tap_medium__v1__zh.docx")
    if not recipe.exists():
        pytest.skip("local TAP recipe is not available")
    ingest_file(recipe)

    response = _post_chat(api_client, "TAP培养基的磷酸盐组分有哪些？", "rag-chat-tap-phosphate")

    assert response.status_code == 200
    body = response.json()
    reply = body["natural_reply"]
    assert "磷酸盐溶液" in reply
    assert "K₂HPO₄" in reply
    assert "KH₂PO₄" in reply
    assert "Hutner" not in body["agent_output"]["answer"]["conclusion"]


def test_chat_paper_question_returns_readable_paper_list(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    source = tmp_path / "paper__machine_learning_approaches_microalgae_cultivation_systems__2024__en.txt"
    source.write_text(
        "Machine learning approaches support microalgae cultivation optimization and growth prediction.",
        encoding="utf-8",
    )
    ingest_file(source)

    response = _post_chat(
        api_client,
        "哪些论文讨论了 microalgae cultivation 的 machine learning 方法？",
        "rag-chat-paper-readable",
    )

    assert response.status_code == 200
    body = response.json()
    reply = body["natural_reply"]
    assert "命中以下与问题相关的论文" in reply
    assert "machine learning approaches microalgae cultivation systems" in reply
    assert "检索到的主要依据是" not in body["agent_output"]["answer"]["conclusion"]


def test_chat_experiment_data_question_reports_biomass_growth_curve(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    data = Path("data/raw/experiment_data/experiment_data__algae_growth_curve__v1__zh.xlsx")
    if not data.exists():
        pytest.skip("local experiment data XLSX is not available")
    ingest_file(data)

    response = _post_chat(
        api_client,
        "实验数据里有没有 OD750 或 growth curve 相关记录？",
        "rag-chat-experiment-data",
    )

    assert response.status_code == 200
    body = response.json()
    reply = body["natural_reply"]
    assert "没有" in reply
    assert "OD750" in reply
    assert "biomass g/L" in reply
    assert "不能等同" in reply


def test_chat_m2_table_schema_missing_ph_does_not_hallucinate_environment_columns(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    import openpyxl

    path = tmp_path / "experiment_data__algae_growth_curve__v1__zh.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append([None, "时间 h", "Iμmol·m−2·s−1", "N mg/L", "P mg/L", "tf", "biomass g/L"])
    sheet.append([0, 0, 26, 0.335, 0.0157, 12, 0.11])
    workbook.save(path)
    ingest_file(path)

    response = _post_chat(api_client, "实验数据里有没有 pH 字段？", "rag-chat-m2-ph")

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    reply = body["natural_reply"]
    assert "没有" in reply
    assert "pH" in reply
    assert "CO2 concentration" not in reply
    assert "temperature (°C)" not in reply


def test_chat_m2_environment_variables_use_true_schema_columns(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    import openpyxl

    path = tmp_path / "experiment_data__algae_growth_curve__v1__zh.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append([None, "时间 h", "Iμmol·m−2·s−1", "N mg/L", "P mg/L", "tf", "biomass g/L"])
    sheet.append([0, 0, 26, 0.335, 0.0157, 12, 0.11])
    workbook.save(path)
    ingest_file(path)

    response = _post_chat(api_client, "实验数据里环境变量是哪些？", "rag-chat-m2-env-vars")

    assert response.status_code == 200
    body = response.json()
    reply = body["natural_reply"]
    assert "Iμmol·m−2·s−1" in reply
    assert "N mg/L" in reply
    assert "P mg/L" in reply
    assert "tf" in reply
    assert "biomass g/L" not in body["agent_output"]["answer"]["conclusion"]
    assert "CO2 concentration" not in reply


def test_chat_rag_blocks_direct_cultivation_plan_modification(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    source = tmp_path / "paper__cultivation_background__2024__en.txt"
    source.write_text("A paper about microalgae cultivation.", encoding="utf-8")
    ingest_file(source)

    response = _post_chat(api_client, "请根据论文直接修改培养方案", "rag-chat-block-modify-plan")

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    assert body["agent_output"]["blocked"] is True
    assert body["agent_output"]["blocked_reason"] == "rag_read_only_boundary"


def test_chat_routes_paper_question_to_local_rag_sources(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    source = tmp_path / "paper__machine_learning_approaches_microalgae_cultivation_systems__2024__en.txt"
    source.write_text(
        "This paper reviews machine learning approaches for microalgae cultivation systems.",
        encoding="utf-8",
    )
    ingest_file(source)

    response = _post_chat(
        api_client,
        "哪些论文讨论了 microalgae cultivation 的 machine learning 方法？",
        "rag-chat-paper",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    assert body["agent_output"]["citations"][0]["doc_type"] == "paper"
    assert source.name in body["natural_reply"]
    assert "未检索到" not in body["natural_reply"]


def test_chat_manual_question_can_return_pdf_page_number(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    manual = Path("data/raw/manual/manual__microalgae_laboratory_manual__v1__en.pdf")
    if not manual.exists():
        pytest.skip("local manual PDF is not available")
    ingest_file(manual)

    response = _post_chat(
        api_client,
        "实验手册中和 TAP 培养基相关的操作有哪些？",
        "rag-chat-manual",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    manual_citations = [
        item
        for item in body["agent_output"]["citations"]
        if item["doc_type"] == "manual"
    ]
    assert manual_citations
    assert any(item.get("page_number") for item in manual_citations)


def test_chat_manual_tap_operation_question_prefers_manual_operations(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    recipe = Path("data/raw/media_recipes/media_recipe__tap_medium__v1__zh.docx")
    manual = Path("data/raw/manual/manual__microalgae_laboratory_manual__v1__en.pdf")
    if not recipe.exists() or not manual.exists():
        pytest.skip("local TAP recipe or manual PDF is not available")
    ingest_file(recipe)
    ingest_file(manual)

    response = _post_chat(
        api_client,
        "实验手册中和 TAP 培养基相关的操作有哪些？",
        "rag-chat-manual-operation",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "rag_answer"
    citations = body["agent_output"]["citations"]
    assert citations
    assert {item["doc_type"] for item in citations} == {"manual"}
    reply = body["natural_reply"]
    assert "根据实验手册" in reply
    assert "相关的操作" in reply
    assert "TAP 培养基由 1 L 工作液和 3 类母液组成" not in reply
    assert "page " in reply


def test_chat_current_generation_uses_fact_layer_not_rag(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
    tmp_path,
):
    source = tmp_path / "paper__generation_number_background__2024__en.txt"
    source.write_text("generation_number is a database fact and not a paper fact.", encoding="utf-8")
    ingest_file(source)

    response = _post_chat(
        api_client,
        "当前 Chlorella_01 的 generation_number 是多少？",
        "rag-chat-fact",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent_output"]["action"] == "query_strain"
    assert body["agent_output"]["data"]["strain_id"] == "Chlorella_01"
    assert "generation_number" in body["natural_reply"]


def test_chat_fact_layer_status_queries_have_specific_replies(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    pending = _post_chat(api_client, "当前 Chlorella_01 的 pending 状态是什么？", "fact-pending-specific")
    assert pending.status_code == 200
    pending_body = pending.json()
    assert pending_body["agent_output"]["action"] == "query_strain"
    assert pending_body["agent_output"]["query_type"] == "pending"
    assert "pending" in pending_body["natural_reply"]

    reminder = _post_chat(api_client, "Chlorella_01 的提醒状态是什么？", "fact-reminder-specific")
    assert reminder.status_code == 200
    reminder_body = reminder.json()
    assert reminder_body["agent_output"]["action"] == "query_strain"
    assert reminder_body["agent_output"]["query_type"] == "reminder_status"
    assert "提醒" in reminder_body["natural_reply"]


def test_chat_workflow_email_and_write_do_not_route_to_rag(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    workflow = _post_chat(api_client, "请执行 Chlorella_01 的传代流程", "rag-chat-workflow")
    assert workflow.status_code == 200
    assert workflow.json()["agent_output"]["action"] != "rag_answer"

    email = _post_chat(api_client, "帮我发送邮件提醒实验员检查 Chlorella_01", "rag-chat-email")
    assert email.status_code == 200
    assert email.json()["agent_output"]["action"] == "email_draft"

    write = _post_chat(api_client, "把 Chlorella_01 的 generation_number 改成 18", "rag-chat-write")
    assert write.status_code == 200
    assert write.json()["agent_output"]["action"] != "rag_answer"
    assert database.list_rag_documents() == []


def test_chat_external_ecology_question_never_answers_with_lab_strain_count(
    api_client,
    isolated_session_memory,
    isolated_sqlite_db,
    isolated_chat_side_effects,
):
    response = _post_chat(
        api_client,
        "我想知道在沈阳市，野外分布有哪些常见藻种",
        "rag-chat-external-ecology",
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["agent_output"]["action"] == "model_fallback"
    assert body["agent_output"]["answer_source"] == "model_prior"
    assert body["agent_output"]["action"] != "list_strains"
    assert "当前数据库中共有" not in body["natural_reply"]
