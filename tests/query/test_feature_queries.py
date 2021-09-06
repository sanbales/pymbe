import networkx as nx

from pymbe.label import get_label_for_id
from pymbe.query.query import (
    get_features_typed_by_type,
    get_types_for_feature,
    roll_up_lower_multiplicity,
    roll_up_multiplicity_for_type,
    roll_up_upper_multiplicity,
)
from tests.conftest import kerbal_model_loaded_client

ROCKET_BUILDING = "Model::Kerbal::Rocket Building::"
PARTS_LIBRARY = "Model::Kerbal::Parts Library::"
SIMPLE_MODEL = "Model::Simple Parts Model::"


def test_feature_to_type1(kerbal_client, kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    engines_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::engines: Liquid Engine <<PartUsage>>"
    ]
    engine_type_feat = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Engine <<PartDefinition>>"]

    assert get_types_for_feature(kerbal_lpg, engines_feat) == [engine_type_feat]


def test_type_to_feature1(kerbal_client, kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    engines_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::engines: Liquid Engine <<PartUsage>>"
    ]
    engine_type_feat = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Engine <<PartDefinition>>"]

    assert get_features_typed_by_type(kerbal_lpg, engine_type_feat) == [engines_feat]


def test_type_to_feature2(simple_parts_client, simple_parts_lpg, simple_parts_stable_names):
    *_, qualified_name_to_id = simple_parts_stable_names

    port_type_id = qualified_name_to_id[f"Model::Ports::Port <<PortDefinition>>"]
    power_in_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part::Power In: Port <<PortUsage>>"
    ]
    power_out_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power Source: Part::Power Out: Port <<PortUsage>>"
    ]

    print(get_features_typed_by_type(simple_parts_lpg, port_type_id))

    assert power_in_id in get_features_typed_by_type(
        simple_parts_lpg, port_type_id
    ) and power_out_id in get_features_typed_by_type(simple_parts_lpg, port_type_id)


def test_banded_graph_paths1(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    rocket_id = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]
    engines_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::engines: Liquid Engine <<PartUsage>>"
    ]

    all_paths = nx.all_simple_paths(
        kerbal_lpg.get_projection("Expanded Banded"), engines_feat, rocket_id
    )

    path_lists = list(all_paths)

    for path in path_lists:
        path_naming = []
        for item in path:
            path_naming.append(get_label_for_id(item, kerbal_lpg.model))

        print(path_naming)

    assert len(path_lists) == 1


def test_banded_graph_paths2(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names
    rocket_id = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]
    ft200_feat_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::tanks: Fuel Tank Section <<PartUsage>>"
    ]

    all_paths = nx.all_simple_paths(
        kerbal_lpg.get_projection("Expanded Banded"), ft200_feat_id, rocket_id
    )

    path_lists = list(all_paths)

    for path in path_lists:
        path_naming = []
        for item in path:
            path_naming.append(get_label_for_id(item, kerbal_lpg.model))

        print(path_naming)

    assert len(path_lists) == 1


def test_banded_graph_paths3(simple_parts_lpg, simple_parts_stable_names):
    *_, qualified_name_to_id = simple_parts_stable_names

    power_group_id = qualified_name_to_id[f"{SIMPLE_MODEL}Power Group: Part <<PartUsage>>"]

    power_in_port_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part::Power In: Port <<PortUsage>>"
    ]
    power_out_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power Source: Part::Power Out: Port <<PortUsage>>"
    ]

    ebg = simple_parts_lpg.get_projection("Expanded Banded")

    assert power_group_id in list(simple_parts_lpg.nodes)
    assert power_in_port_id in list(simple_parts_lpg.nodes)

    assert power_group_id in list(ebg.nodes)
    assert power_in_port_id in list(ebg.nodes)

    all_paths = nx.all_simple_paths(ebg, power_out_id, power_group_id)

    path_lists = list(all_paths)
    print(path_lists)

    assert len(path_lists) == 1


def test_feature_multiplicity_rollup1(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    engines_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::engines: Liquid Engine <<PartUsage>>"
    ]
    stages_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage <<PartUsage>>"
    ]

    assert engines_feat in list(kerbal_lpg.nodes.keys())
    assert stages_feat in list(kerbal_lpg.nodes.keys())

    engines_lower_mult = roll_up_lower_multiplicity(
        lpg=kerbal_lpg,
        feature=kerbal_lpg.model.elements[engines_feat],
    )

    engines_upper_mult = roll_up_upper_multiplicity(
        lpg=kerbal_lpg,
        feature=kerbal_lpg.model.elements[engines_feat],
    )

    stages_lower_mult = roll_up_lower_multiplicity(
        lpg=kerbal_lpg,
        feature=kerbal_lpg.model.elements[stages_feat],
    )

    stages_upper_mult = roll_up_upper_multiplicity(
        lpg=kerbal_lpg,
        feature=kerbal_lpg.model.elements[stages_feat],
    )

    assert engines_lower_mult == 0
    assert engines_upper_mult == 40

    assert stages_lower_mult == 1
    assert stages_upper_mult == 5


def test_type_multiplicity_rollup1(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    liquid_engine_type = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Engine <<PartDefinition>>"]
    rocket_type = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]

    liquid_engine_ele = kerbal_lpg.model.elements[liquid_engine_type]
    rocket_ele = kerbal_lpg.model.elements[rocket_type]

    liquid_upper = roll_up_multiplicity_for_type(
        kerbal_lpg,
        liquid_engine_ele,
        "upper",
    )

    rocket_upper = roll_up_multiplicity_for_type(
        kerbal_lpg,
        rocket_ele,
        "upper",
    )

    assert liquid_upper == 40
    assert rocket_upper == 0


def test_type_multiplicity_rollup2(simple_parts_lpg, simple_parts_stable_names):
    *_, qualified_name_to_id = simple_parts_stable_names

    power_out_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power Source: Part::Power Out: Port <<PortUsage>>"
    ]
    power_in_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part::Power In: Port <<PortUsage>>"
    ]
    port_type_id = qualified_name_to_id["Model::Ports::Port <<PortDefinition>>"]

    port_type = simple_parts_lpg.model.elements[port_type_id]

    port_upper = roll_up_multiplicity_for_type(
        simple_parts_lpg,
        port_type,
        "upper",
    )

    power_in_mult = roll_up_upper_multiplicity(
        lpg=simple_parts_lpg,
        feature=simple_parts_lpg.model.elements[power_in_id],
    )

    power_out_mult = roll_up_upper_multiplicity(
        lpg=simple_parts_lpg,
        feature=simple_parts_lpg.model.elements[power_out_id],
    )

    assert power_in_mult == 4
    assert power_out_mult == 2

    assert port_upper == 6
