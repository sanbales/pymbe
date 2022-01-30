from pymbe.interpretation.interp_playbooks import (
    build_expression_sequence_templates,
    build_sequence_templates,
)
from pymbe.interpretation.results import pprint_double_id_list

ROCKET_BUILDING = "Model::Kerbal::Rocket Building::"
PARTS_LIBRARY = "Model::Kerbal::Parts Library::"

SIMPLE_MODEL = "Model::Simple Parts Model::"
PARTS_LIBRARY = "Model::Simple Parts Model::Fake Library::"


def test_feature_sequence_templates1(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    seq_templates = build_sequence_templates(kerbal_lpg)

    assert len(seq_templates) == 39

    _solid_booster_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Booster <<PartDefinition>>"]
    _boosters_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Solid Stage::boosters: Solid Booster <<PartUsage>>"
    ]
    liquid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Stage <<PartDefinition>>"]
    engines_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::engines: Liquid Engine <<PartUsage>>"
    ]
    tanks_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::tanks: Fuel Tank Section <<PartUsage>>"
    ]
    krp_mass_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Kerbal Rocket Part::Mass: Real <<AttributeUsage>>"
    ]

    print(pprint_double_id_list(seq_templates, kerbal_lpg.model))

    assert [liquid_stage_id, engines_id] in seq_templates

    assert any(krp_mass_id in seq for seq in seq_templates)

    assert any((engines_id in seq) and (liquid_stage_id in seq) for seq in seq_templates)

    assert any((tanks_id in seq) and (liquid_stage_id in seq) for seq in seq_templates)


def test_feature_sequence_templates2(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    seq_templates = build_sequence_templates(kerbal_lpg)

    coupler_usage_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to "
        f"Carrying Stage: Coupler <<PartUsage>>"
    ]
    rocket_id = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]
    stages_feat = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage <<PartUsage>>"
    ]
    sep_force_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to Carrying Stage: "
        f"Coupler::Separation Force: Real <<AttributeUsage>>"
    ]

    print(pprint_double_id_list(seq_templates, kerbal_lpg.model))

    assert [rocket_id, stages_feat, coupler_usage_id, sep_force_id] in seq_templates


def test_feature_sequence_templates3(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    seq_templates = build_sequence_templates(kerbal_lpg)

    coupler_usage_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to "
        f"Carrying Stage: Coupler <<PartUsage>>"
    ]
    rocket_id = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]
    sep_force_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to Carrying Stage: "
        f"Coupler::Separation Force: Real <<AttributeUsage>>"
    ]

    print(pprint_double_id_list(seq_templates, kerbal_lpg.model))

    assert [rocket_id, coupler_usage_id, sep_force_id] not in seq_templates


def test_feature_sequence_templates4(simple_parts_lpg, simple_parts_stable_names):
    *_, qualified_name_to_id = simple_parts_stable_names

    seq_templates = build_sequence_templates(simple_parts_lpg)

    power_group_id = qualified_name_to_id[f"{SIMPLE_MODEL}Power Group: Part <<PartUsage>>"]
    power_user_part_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part <<PartUsage>>"
    ]
    power_in_port_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part::Power In: Port <<PortUsage>>"
    ]

    print(seq_templates)

    assert [power_group_id, power_user_part_id, power_in_port_id] in seq_templates


def test_expression_sequence_templates(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    top_plus = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass"
        " (FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => "
        "$result <<OperatorExpression>>"
    ]

    fre_1_result = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass "
        "(FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => $result::sum "
        "(tanks.Full Mass (FRE.tanks)) => $result::tanks.Full Mass (FRE.tanks) => "
        "$result::FRE.tanks <<Feature>>"
        # Used to be: "$result::FRE.Full Mass::FRE.Full Mass <<Feature>>"
    ]

    expr_sequences = build_expression_sequence_templates(lpg=kerbal_lpg)
    top_plus_paths = 0
    direct_literals = 0

    fre_result_found = False

    # big roll-up
    for seq in expr_sequences:
        if top_plus in seq:
            top_plus_paths += 1
        if fre_1_result in seq:
            fre_result_found = True

    # individual literal expression value assignments
    for seq in expr_sequences:
        if "Literal" in kerbal_lpg.nodes[seq[0]]["@type"]:
            direct_literals += 1
            assert len(seq) == 2
        # each sequence should end with the result
        assert kerbal_lpg.nodes[seq[len(seq) - 1]]["@type"] == "Feature"

    assert top_plus_paths == 15
    assert direct_literals == 26
    assert len(expr_sequences) == 41

    assert fre_result_found
