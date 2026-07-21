from app.services.intent.write_action_parser import (
    parse_add_fields,
    parse_update_fields,
    parse_write_action,
    validate_required_fields,
)

from tests.conftest import context


def test_parse_pipe_add_format():
    fields = parse_add_fields("Spirulina_02 | 螺旋藻 (Spirulina platensis)")

    assert fields["strain_id"] == "Spirulina_02"
    assert fields["name_cn"] == "螺旋藻"
    assert fields["name_en"] == "Spirulina platensis"


def test_add_unknown_algae_is_not_hard_coded_but_has_missing_fields():
    plan = parse_write_action("增加衣藻", context())

    assert plan["operation"] == "add"
    assert plan["tool_name"] == "add_algae_strain"
    assert plan["fields"]["name_cn"] == "衣藻"
    assert set(plan["missing_fields"]) == {"strain_id", "name_en"}


def test_update_uses_database_candidate_as_target_and_extracts_new_id():
    ctx = context(strains=[
        {
            "strain_id": "Chlamydomonas_01",
            "name_cn": "莱茵衣藻",
            "name_en": "Chlamydomonas reinhardtii",
        }
    ])

    plan = parse_write_action("把实验室里的莱茵衣藻的品系改为Chlamydomonas_137AH", ctx)

    assert plan["operation"] == "update"
    assert plan["fields"]["strain_id"] == "Chlamydomonas_01"
    assert plan["fields"]["new_strain_id"] == "Chlamydomonas_137AH"
    assert plan["missing_fields"] == []


def test_validate_required_fields_is_backend_owned():
    complete, missing = validate_required_fields("add", {"name_cn": "螺旋藻"})

    assert complete is False
    assert missing == ["strain_id", "name_en"]


def test_parse_update_generation_field():
    fields = parse_update_fields("代数改为 3")

    assert fields["generation_number"] == 3
