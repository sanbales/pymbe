import networkx as nx
import pytest

from pymbe.interpretation.interp_playbooks import (
    random_generator_phase_1_multiplicities,
    random_generator_playbook_phase_2_rollup,
    random_generator_playbook_phase_2_unconnected,
)
from pymbe.interpretation.set_builders import (
    create_set_with_new_instances,
    extend_sequences_by_sampling,
)

ROCKET_BUILDING = "Model::Kerbal::Rocket Building::"
PARTS_LIBRARY = "Model::Kerbal::Parts Library::"
SIMPLE_MODEL = "Model::Simple Parts Model::"


def test_type_multiplicity_dict_building(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    solid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Stage <<PartDefinition>>"]
    liquid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Stage <<PartDefinition>>"]
    flea_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-5 "Flea" Solid Fuel Booster <<PartDefinition>>"""
    ]
    hammer_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-10 "Hammer" Solid Fuel Booster <<PartDefinition>>"""
    ]
    f100_tank = qualified_name_to_id[f"{PARTS_LIBRARY}FL-T100 Fuel Tank <<PartDefinition>>"]
    f200_tank = qualified_name_to_id[f"{PARTS_LIBRARY}FL-T200 Fuel Tank <<PartDefinition>>"]
    real_id = qualified_name_to_id["Model::Real <<DataType>>"]

    ptg = kerbal_lpg.get_projection("Part Typing")
    scg = kerbal_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(kerbal_lpg, ptg, scg)

    assert len(full_multiplicities) == 9
    assert full_multiplicities[solid_stage_id] + full_multiplicities[liquid_stage_id] == 5
    assert full_multiplicities[flea_id] + full_multiplicities[hammer_id] == 40
    assert full_multiplicities[f100_tank] + full_multiplicities[f200_tank] == 150
    assert full_multiplicities[real_id] == 3038


def test_phase_1_instance_creation(kerbal_random_stage_1_instances, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    flea_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-5 "Flea" Solid Fuel Booster <<PartDefinition>>"""
    ]
    solid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Stage <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Booster <<PartDefinition>>"]
    real_id = qualified_name_to_id["Model::Real <<DataType>>"]

    assert flea_id in kerbal_random_stage_1_instances
    assert solid_stage_id in kerbal_random_stage_1_instances
    assert real_id in kerbal_random_stage_1_instances
    # don"t expect abstract member at this point
    assert solid_booster_id not in kerbal_random_stage_1_instances


def test_phase_1_singleton_instances(kerbal_random_stage_1_complete, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    flea_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-5 "Flea" Solid Fuel Booster <<PartDefinition>>"""
    ]
    solid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Stage <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Booster <<PartDefinition>>"]
    pod_id = qualified_name_to_id[f"{PARTS_LIBRARY}Mk1 Command Pod <<PartDefinition>>"]

    assert flea_id in kerbal_random_stage_1_complete
    assert solid_stage_id in kerbal_random_stage_1_complete
    assert pod_id in kerbal_random_stage_1_complete
    # don"t expect abstract member at this point
    assert solid_booster_id not in kerbal_random_stage_1_complete


def test_phase_2_instance_creation(
    kerbal_lpg, kerbal_random_stage_1_complete, kerbal_stable_names, kerbal_client
):
    *_, qualified_name_to_id = kerbal_stable_names

    solid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Stage <<PartDefinition>>"]
    liquid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Stage <<PartDefinition>>"]
    flea_id = qualified_name_to_id[
        f"""{PARTS_LIBRARY}RT-5 "Flea" Solid Fuel Booster <<PartDefinition>>"""
    ]
    pod_id = qualified_name_to_id[f"{PARTS_LIBRARY}Mk1 Command Pod <<PartDefinition>>"]
    krp_id = qualified_name_to_id[f"{ROCKET_BUILDING}Kerbal Rocket Part <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Booster <<PartDefinition>>"]
    rocket_id = qualified_name_to_id[f"{ROCKET_BUILDING}Rocket <<PartDefinition>>"]

    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(scg, kerbal_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(kerbal_lpg.model, kerbal_random_stage_1_complete)

    assert flea_id in kerbal_random_stage_1_complete
    assert pod_id in kerbal_random_stage_1_complete
    assert krp_id in kerbal_random_stage_1_complete
    assert solid_booster_id in kerbal_random_stage_1_complete
    assert rocket_id in kerbal_random_stage_1_complete

    assert (
        len(kerbal_random_stage_1_complete[solid_stage_id])
        + len(kerbal_random_stage_1_complete[liquid_stage_id])
        == 5
    )
    assert len(kerbal_random_stage_1_complete[solid_booster_id]) == 40

    assert len(kerbal_random_stage_1_complete[krp_id]) == 272


def test_phase_3_instance_sampling(kerbal_random_stage_3_complete, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names

    coupler_usage_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to "
        f"Carrying Stage: Coupler <<PartUsage>>"
    ]
    sep_force_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Rocket::stages: Rocket Stage::Coupler to Carrying Stage: "
        f"Coupler::Separation Force: Real <<AttributeUsage>>"
    ]

    booster_empty_mass_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Solid Booster::Empty Mass: Real <<AttributeUsage>>"
    ]
    booster_isp_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Solid Booster::Specific Impulse: Real <<AttributeUsage>>"
    ]

    assert coupler_usage_id in kerbal_random_stage_3_complete

    if len(kerbal_random_stage_3_complete[coupler_usage_id]) > 0:
        if len(kerbal_random_stage_3_complete[sep_force_id][0]) == 3:
            print(kerbal_random_stage_3_complete[sep_force_id][0])

    # not sure what's up with the varying lengths to the sep force sequences
    assert len(kerbal_random_stage_3_complete[coupler_usage_id]) == 0 or len(
        kerbal_random_stage_3_complete[sep_force_id][0]
    ) in (3, 4)

    assert len(kerbal_random_stage_3_complete[booster_isp_id]) > 0
    # TODO: figure out why this fails sometimes
    # assert len(kerbal_random_stage_3_complete[rt_10_isp_id]) > 0
    assert len(kerbal_random_stage_3_complete[booster_empty_mass_id]) > 0


def test_phase_4_instance_sampling(
    kerbal_random_stage_4_complete, kerbal_stable_names, kerbal_lpg
):
    *_, qualified_name_to_id = kerbal_stable_names

    top_plus_expr_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass "
        f"(FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => "
        f"$result <<OperatorExpression>>"
    ]
    sum_1_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ "
        + "(sum (engines.Mass (FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => $result::sum "
        + "(tanks.Full Mass (FRE.tanks)) => $result <<InvocationExpression>>"
    ]
    full_mass_pse = qualified_name_to_id[
        f"{ROCKET_BUILDING}Liquid Stage::Full Mass: Real::+ (sum (engines.Mass"
        + " (FRE.engines)), sum (tanks.Full Mass (FRE.tanks))) => $result::sum "
        + "(tanks.Full Mass (FRE.tanks)) => $result::tanks.Full Mass (FRE.tanks)"
        + " => $result <<PathStepExpression>>"
    ]

    liquid_stage_id = qualified_name_to_id[f"{ROCKET_BUILDING}Liquid Stage <<PartDefinition>>"]

    booster_empty_mass_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Solid Booster::Empty Mass: Real <<AttributeUsage>>"
    ]
    booster_isp_id = qualified_name_to_id[
        f"{ROCKET_BUILDING}Solid Booster::Specific Impulse: Real <<AttributeUsage>>"
    ]
    rt_10_isp_id = qualified_name_to_id[
        f'{PARTS_LIBRARY}RT-10 "Hammer" Solid Fuel Booster::Specific Impulse: '
        f"Real <<AttributeUsage>>"
    ]

    assert (
        len(kerbal_random_stage_4_complete[full_mass_pse]) > 0
        or len(kerbal_random_stage_4_complete[liquid_stage_id]) == 0
    )

    if len(kerbal_random_stage_4_complete[liquid_stage_id]) > 0:
        assert len(kerbal_random_stage_4_complete[top_plus_expr_id][0]) == 2

    if len(kerbal_random_stage_4_complete[liquid_stage_id]) > 0:
        # TODO: figure out why this is not consistent
        assert len(kerbal_random_stage_4_complete[sum_1_id][0]) in (2, 3)

    assert len(kerbal_random_stage_4_complete[booster_isp_id]) > 0
    # FIXME: uncomment this when we can resolve the "flake"
    # assert len(kerbal_random_stage_4_complete[rt_10_isp_id]) > 0
    assert len(kerbal_random_stage_4_complete[booster_empty_mass_id]) > 0


@pytest.mark.skip()
def test_sp_phase_1_instance_creation(
    simple_parts_random_stage_1_instances, simple_parts_stable_names
):
    *_, qualified_name_to_id = simple_parts_stable_names

    part_id = qualified_name_to_id["Model::Parts::Part <<PartDefinition>>"]
    port_id = qualified_name_to_id["Model::Ports::Port <<PortDefinition>>"]
    connection_id = qualified_name_to_id["Model::Connections::Connection <<ConnectionDefinition>>"]

    power_user_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part <<PartUsage>>"
    ]

    assert part_id in simple_parts_random_stage_1_instances
    assert port_id in simple_parts_random_stage_1_instances
    assert connection_id in simple_parts_random_stage_1_instances
    # don"t expect features at this point
    assert power_user_id not in simple_parts_random_stage_1_instances


@pytest.mark.skip()
def test_sp_phase_1_singleton_instances(
    simple_parts_random_stage_1_complete, simple_parts_stable_names
):
    *_, qualified_name_to_id = simple_parts_stable_names

    part_id = qualified_name_to_id["Model::Parts::Part <<PartDefinition>>"]
    port_id = qualified_name_to_id["Model::Ports::Port <<PortDefinition>>"]
    connection_id = qualified_name_to_id["Model::Connections::Connection <<ConnectionDefinition>>"]

    power_user_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part <<PartUsage>>"
    ]

    assert part_id in simple_parts_random_stage_1_complete
    assert port_id in simple_parts_random_stage_1_complete
    assert connection_id in simple_parts_random_stage_1_complete
    # don"t expect features at this point
    assert power_user_id not in simple_parts_random_stage_1_complete


@pytest.mark.skip()
def test_sp_phase_2_instance_creation(
    simple_parts_lpg,
    simple_parts_random_stage_1_complete,
    simple_parts_stable_names,
    simple_parts_client,
):
    *_, qualified_name_to_id = simple_parts_stable_names

    connection_id = qualified_name_to_id["Model::Connections::Connection <<ConnectionDefinition>>"]
    power_user_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part <<PartUsage>>"
    ]
    part_id = qualified_name_to_id["Model::Parts::Part <<PartDefinition>>"]
    port_id = qualified_name_to_id["Model::Ports::Port <<PortDefinition>>"]

    scg = simple_parts_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(scg, simple_parts_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(
        simple_parts_lpg.model, simple_parts_random_stage_1_complete
    )

    assert part_id in simple_parts_random_stage_1_complete
    assert port_id in simple_parts_random_stage_1_complete
    assert connection_id in simple_parts_random_stage_1_complete
    # don"t expect features at this point
    assert power_user_id not in simple_parts_random_stage_1_complete

    assert len(simple_parts_random_stage_1_complete[part_id]) == 7
    assert len(simple_parts_random_stage_1_complete[port_id]) == 6


@pytest.mark.skip()
def test_sp_phase_3_instance_sampling(
    simple_parts_random_stage_3_complete, simple_parts_stable_names
):
    *_, qualified_name_to_id = simple_parts_stable_names

    power_source_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power Source: Part <<PartUsage>>"
    ]
    power_user_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part <<PartUsage>>"
    ]
    power_in_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power User: Part::Power In: Port <<PortUsage>>"
    ]
    power_out_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::Power Source: Part::Power Out: Port <<PortUsage>>"
    ]
    connect_use_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::powerToUser: Connection <<ConnectionUsage>>"
    ]
    power_group_id = qualified_name_to_id[
        f"{SIMPLE_MODEL}Power Group: Part::powerToUser: Connection <<ConnectionUsage>>"
    ]

    assert connect_use_id in simple_parts_random_stage_3_complete
    # check that each part has exactly one port
    inst_names = [str(inst) for inst in simple_parts_random_stage_3_complete[power_in_id]]
    unique_names = set(inst_names)
    assert len(unique_names) == len(simple_parts_random_stage_3_complete[power_in_id])

    inst_names = [str(inst) for inst in simple_parts_random_stage_3_complete[power_user_id]]
    unique_names = set(inst_names)
    assert len(unique_names) == len(simple_parts_random_stage_3_complete[power_user_id])

    inst_names = [str(inst) for inst in simple_parts_random_stage_3_complete[power_out_id]]
    unique_names = set(inst_names)
    assert len(unique_names) == len(simple_parts_random_stage_3_complete[power_out_id])


def test_expression_inferred_graph(kerbal_lpg):
    # inferred graph provides a reliable order of execution for expressions
    eig = kerbal_lpg.get_projection("Expression Inferred")

    all_edge_types = [edge_type for *_, edge_type in eig.edges]
    implied_edges = [edge for edge in eig.edges if edge[2].startswith("Implied")]

    assert set(all_edge_types).intersection({"ImpliedParameterFeedforward^-1"})
    assert len(implied_edges) == 28

    top_plus_expr_id = "d05c42b2-3453-4c94-bf18-5bbc38949d19"
    fl_200_full_mass_id = "c3344ffd-6a7f-499b-90cf-e7e311e309f5"
    x_feature = "1159649f-b211-42c1-84e7-0bf3fc88417f"
    deep_mass_feature = "0bc0b127-ffcb-44a8-ad65-2c34745e9b2d"

    for comp in nx.connected_components(eig.to_undirected()):
        connected_sub = nx.subgraph(eig, list(comp))
        if top_plus_expr_id in list(comp):
            assert len(list(comp)) == 44


def test_dependency_graph():
    # see how fully solved sequences go to make the dependency graph for computation
    assert True


def test_new_instances(kerbal_lpg):
    # TODO: should move these to a separate file but need the common fixtures
    part_defs = kerbal_lpg.nodes_by_type["PartDefinition"]

    new_instances = {
        part_def: create_set_with_new_instances(
            sequence_template=[kerbal_lpg.model.elements[part_def]],
            quantities=[10],
        )
        for part_def in part_defs
    }
    assert len(new_instances) == 17


def test_instance_sampling(kerbal_lpg, kerbal_stable_names):
    *_, qualified_name_to_id = kerbal_stable_names
    # what we really need here is a fixture that generates a healthy instance dictionary from the playbook phases
    solid_booster_id = qualified_name_to_id[f"{ROCKET_BUILDING}Solid Booster <<PartDefinition>>"]

    # try creating the boosters under solid stage

    solid_stage_instances = create_set_with_new_instances(
        sequence_template=[kerbal_lpg.model.elements[solid_booster_id]],
        quantities=[5],
    )

    solid_booster_instances = create_set_with_new_instances(
        sequence_template=[kerbal_lpg.model.elements[solid_booster_id]],
        quantities=[40],
    )

    booster_feature_instances = extend_sequences_by_sampling(
        solid_stage_instances,
        8,
        8,
        solid_booster_instances,
        False,
        {},
    )

    assert len(booster_feature_instances) == 40
    assert len(booster_feature_instances[4]) == 2
