from __future__ import annotations

from app.services.labware.models import ContainerRef, LabwareSpec, WorkspaceLocation


LABWARE_SPECS = {
    "erlenmeyer_flask_250ml": LabwareSpec(
        labware_id="erlenmeyer_flask_250ml",
        display_name="250 mL Erlenmeyer flask",
        category="culture_vessel",
        max_volume_ml=250.0,
        compatible_operations=[
            "dispense_medium",
            "transfer_seed",
            "close_reactor",
            "configure_incubator",
        ],
    ),
    "culture_bottle_250ml": LabwareSpec(
        labware_id="culture_bottle_250ml",
        display_name="250 mL culture bottle",
        category="culture_vessel",
        max_volume_ml=250.0,
        compatible_operations=[
            "dispense_medium",
            "transfer_seed",
            "close_reactor",
            "configure_incubator",
        ],
    ),
    "centrifuge_tube_50ml": LabwareSpec(
        labware_id="centrifuge_tube_50ml",
        display_name="50 mL centrifuge tube",
        category="sample_tube",
        max_volume_ml=50.0,
        compatible_operations=["transfer_seed", "centrifuge", "sample"],
    ),
    "cuvette_3_5ml": LabwareSpec(
        labware_id="cuvette_3_5ml",
        display_name="3.5 mL spectrophotometer cuvette",
        category="measurement_cell",
        max_volume_ml=3.5,
        compatible_operations=["measure_od"],
    ),
    "microplate_96_well": LabwareSpec(
        labware_id="microplate_96_well",
        display_name="96-well microplate",
        category="measurement_plate",
        max_volume_ml=0.35,
        compatible_operations=["measure_od", "transfer_liquid"],
    ),
}


WORKSPACE_LOCATIONS = {
    "biosafety_cabinet": WorkspaceLocation(
        location_id="biosafety_cabinet",
        display_name="Biosafety cabinet / clean bench",
        zone_type="sterile_workspace",
        supported_operations=[
            "sterilize_workspace",
            "check_materials",
            "dispense_medium",
            "transfer_seed",
            "close_reactor",
            "cleanup",
        ],
    ),
    "incubator_1": WorkspaceLocation(
        location_id="incubator_1",
        display_name="Illuminated incubator",
        zone_type="incubator",
        supported_operations=["configure_incubator", "incubate"],
    ),
    "shaker_1": WorkspaceLocation(
        location_id="shaker_1",
        display_name="Constant-temperature shaker",
        zone_type="shaker",
        supported_operations=["incubate", "shake"],
    ),
    "centrifuge_1": WorkspaceLocation(
        location_id="centrifuge_1",
        display_name="Centrifuge",
        zone_type="centrifuge",
        supported_operations=["centrifuge"],
    ),
    "spectrophotometer_1": WorkspaceLocation(
        location_id="spectrophotometer_1",
        display_name="UV-visible spectrophotometer",
        zone_type="measurement",
        supported_operations=["measure_od"],
    ),
    "plate_reader_1": WorkspaceLocation(
        location_id="plate_reader_1",
        display_name="Microplate reader",
        zone_type="measurement",
        supported_operations=["measure_od"],
    ),
    "balance_1": WorkspaceLocation(
        location_id="balance_1",
        display_name="Electronic balance",
        zone_type="weighing",
        supported_operations=["weigh"],
    ),
}


CONTAINER_REFS = {
    "Reactor_A": ContainerRef(
        container_id="Reactor_A",
        labware_id="erlenmeyer_flask_250ml",
        location_id="biosafety_cabinet",
        role="source_culture",
        current_volume_ml=150.0,
    ),
    "Reactor_B": ContainerRef(
        container_id="Reactor_B",
        labware_id="erlenmeyer_flask_250ml",
        location_id="biosafety_cabinet",
        role="target_culture",
        current_volume_ml=0.0,
    ),
    "OD_Cuvette_A": ContainerRef(
        container_id="OD_Cuvette_A",
        labware_id="cuvette_3_5ml",
        location_id="spectrophotometer_1",
        role="measurement_cell",
        current_volume_ml=0.0,
    ),
    "CentrifugeTube_A": ContainerRef(
        container_id="CentrifugeTube_A",
        labware_id="centrifuge_tube_50ml",
        location_id="centrifuge_1",
        role="sample_tube",
        current_volume_ml=0.0,
    ),
    "Microplate_96_A": ContainerRef(
        container_id="Microplate_96_A",
        labware_id="microplate_96_well",
        location_id="plate_reader_1",
        role="measurement_plate",
        current_volume_ml=0.0,
    ),
}


def get_labware_spec(labware_id: str) -> LabwareSpec | None:
    return LABWARE_SPECS.get(labware_id)


def get_workspace_location(location_id: str) -> WorkspaceLocation | None:
    return WORKSPACE_LOCATIONS.get(location_id)


def get_container_ref(container_id: str) -> ContainerRef | None:
    return CONTAINER_REFS.get(container_id)


def list_labware_specs() -> list[LabwareSpec]:
    return list(LABWARE_SPECS.values())


def list_workspace_locations() -> list[WorkspaceLocation]:
    return list(WORKSPACE_LOCATIONS.values())


def list_container_refs() -> list[ContainerRef]:
    return list(CONTAINER_REFS.values())
