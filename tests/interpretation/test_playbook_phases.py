import networkx as nx

from pymbe.interpretation.interp_playbooks import (
    random_generator_phase_1_multiplicities,
    random_generator_playbook_phase_2_rollup,
    random_generator_playbook_phase_2_unconnected,
)
from pymbe.interpretation.set_builders import (
    create_set_with_new_instances,
    extend_sequences_by_sampling,
)


def test_type_multiplicity_dict_building(kerbal_client, kerbal_lpg, kerbal_stable_names):
    (id_to_qualified_name, qualified_name_to_id) = kerbal_stable_names

    solid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Stage <<PartDefinition>>"]
    liquid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Liquid Stage <<PartDefinition>>"]
    flea_id = qualified_name_to_id["Model::Kerbal::Parts Library::RT-5 \"Flea\" Solid Fuel Booster <<PartDefinition>>"]
    hammer_id = qualified_name_to_id[
        "Model::Kerbal::Parts Library::RT-10 \"Hammer\" Solid Fuel Booster <<PartDefinition>>"]
    f100_tank = qualified_name_to_id["Model::Kerbal::Parts Library::FL-T100 Fuel Tank <<PartDefinition>>"]
    f200_tank = qualified_name_to_id["Model::Kerbal::Parts Library::FL-T200 Fuel Tank <<PartDefinition>>"]
    real_id = qualified_name_to_id["Model::Real <<DataType>>"]

    ptg = kerbal_lpg.get_projection("Part Typing")
    scg = kerbal_lpg.get_projection("Part Definition")

    full_multiplicities = random_generator_phase_1_multiplicities(kerbal_client.elements_by_id, kerbal_lpg, ptg, scg)

    assert len(full_multiplicities) == 9
    assert full_multiplicities[solid_stage_id] + full_multiplicities[liquid_stage_id] == 5
    assert full_multiplicities[flea_id] + full_multiplicities[hammer_id] == 40
    assert full_multiplicities[f100_tank] + full_multiplicities[f200_tank] == 150
    assert full_multiplicities[real_id] == 3038


def test_phase_1_instance_creation(kerbal_random_stage_1_instances, kerbal_stable_names):
    (id_to_qualified_name, qualified_name_to_id) = kerbal_stable_names

    flea_id = qualified_name_to_id["Model::Kerbal::Parts Library::RT-5 \"Flea\" Solid Fuel Booster <<PartDefinition>>"]
    solid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Stage <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Booster <<PartDefinition>>"]
    real_id = qualified_name_to_id["Model::Real <<DataType>>"]

    assert flea_id in kerbal_random_stage_1_instances
    assert solid_stage_id in kerbal_random_stage_1_instances
    assert real_id in kerbal_random_stage_1_instances
    # don"t expect abstract member at this point
    assert solid_booster_id not in kerbal_random_stage_1_instances


def test_phase_1_singleton_instances(kerbal_random_stage_1_complete, kerbal_stable_names):
    (id_to_qualified_name, qualified_name_to_id) = kerbal_stable_names

    flea_id = qualified_name_to_id["Model::Kerbal::Parts Library::RT-5 \"Flea\" Solid Fuel Booster <<PartDefinition>>"]
    solid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Stage <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Booster <<PartDefinition>>"]
    pod_id = qualified_name_to_id["Model::Kerbal::Parts Library::Mk1 Command Pod <<PartDefinition>>"]

    assert flea_id in kerbal_random_stage_1_complete
    assert solid_stage_id in kerbal_random_stage_1_complete
    assert pod_id in kerbal_random_stage_1_complete
    # don"t expect abstract member at this point
    assert solid_booster_id not in kerbal_random_stage_1_complete


def test_phase_2_instance_creation(kerbal_lpg, kerbal_random_stage_1_complete, kerbal_stable_names, kerbal_client):
    (id_to_qualified_name, qualified_name_to_id) = kerbal_stable_names

    solid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Stage <<PartDefinition>>"]
    liquid_stage_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Liquid Stage <<PartDefinition>>"]
    flea_id = qualified_name_to_id["Model::Kerbal::Parts Library::RT-5 \"Flea\" Solid Fuel Booster <<PartDefinition>>"]
    pod_id = qualified_name_to_id["Model::Kerbal::Parts Library::Mk1 Command Pod <<PartDefinition>>"]
    krp_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Kerbal Rocket Part <<PartDefinition>>"]
    solid_booster_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Solid Booster <<PartDefinition>>"]
    rocket_id = qualified_name_to_id["Model::Kerbal::Rocket Building::Rocket <<PartDefinition>>"]

    scg = kerbal_lpg.get_projection("Part Definition")

    random_generator_playbook_phase_2_rollup(kerbal_client.elements_by_id, scg, kerbal_random_stage_1_complete)

    random_generator_playbook_phase_2_unconnected(kerbal_lpg.nodes, kerbal_random_stage_1_complete)

    assert flea_id in kerbal_random_stage_1_complete
    assert pod_id in kerbal_random_stage_1_complete
    assert krp_id in kerbal_random_stage_1_complete
    assert solid_booster_id in kerbal_random_stage_1_complete
    assert rocket_id in kerbal_random_stage_1_complete

    assert len(kerbal_random_stage_1_complete[solid_stage_id]) + \
           len(kerbal_random_stage_1_complete[liquid_stage_id]) == 5
    assert len(kerbal_random_stage_1_complete[solid_booster_id]) == 40

    assert len(kerbal_random_stage_1_complete[krp_id]) == 272


def test_phase_3_instance_sampling(kerbal_random_stage_3_complete):
    coupler_usage_id = "3a609e5a-3e6f-4eb4-97ff-5a32b23122bf"
    sep_force_id = "7f5e38cb-6647-482d-b8fe-5c266d73ab42"

    booster_empty_mass_id = "645ee1b3-3cb3-494e-8cb2-ec32e377c9f6"
    booster_isp_id = "2b1351f4-a0fb-470b-bb22-1b924dde38f7"
    rt_10_isp_id = "eb09ff1c-1791-4571-8016-c0534906faa4"

    assert coupler_usage_id in kerbal_random_stage_3_complete

    if len(kerbal_random_stage_3_complete[coupler_usage_id]) > 0:
        if len(kerbal_random_stage_3_complete[sep_force_id][0]) == 3:
            print(kerbal_random_stage_3_complete[sep_force_id][0])

    # not sure what's up with the varying lengths to the sep force sequences
    assert len(kerbal_random_stage_3_complete[coupler_usage_id]) == 0 or \
           len(kerbal_random_stage_3_complete[sep_force_id][0]) in (3, 4)

    assert len(kerbal_random_stage_3_complete[booster_isp_id]) > 0
    assert len(kerbal_random_stage_3_complete[rt_10_isp_id]) > 0
    assert len(kerbal_random_stage_3_complete[booster_empty_mass_id]) > 0


def test_phase_4_instance_sampling(kerbal_random_stage_4_complete):
    top_plus_expr_id = "b51bb349-e210-4be8-be64-e749ea4e563b"
    sum_1_id = "700d97d1-410a-459c-ad09-8792c27e2803"
    collect_1_id = "d6644a0a-6eef-49c1-a770-60886073554c"
    collect_1_result = "31f8c4bd-9700-4bc3-9970-3eb5451f0203"

    liquid_stage_id = "e6c22f19-e5e0-4a4b-9a3f-af2f01382465"

    booster_empty_mass_id = "645ee1b3-3cb3-494e-8cb2-ec32e377c9f6"
    booster_isp_id = "2b1351f4-a0fb-470b-bb22-1b924dde38f7"
    rt_10_isp_id = "eb09ff1c-1791-4571-8016-c0534906faa4"

    assert len(kerbal_random_stage_4_complete[collect_1_id]) > 0 or \
        len(kerbal_random_stage_4_complete[liquid_stage_id]) == 0

    assert len(kerbal_random_stage_4_complete[collect_1_result]) > 0 or \
           len(kerbal_random_stage_4_complete[liquid_stage_id]) == 0

    if len(kerbal_random_stage_4_complete[liquid_stage_id]) > 0:
        assert len(kerbal_random_stage_4_complete[top_plus_expr_id][0]) == 2

    if len(kerbal_random_stage_4_complete[liquid_stage_id]) > 0:
        assert len(kerbal_random_stage_4_complete[sum_1_id][0]) == 3

    assert len(kerbal_random_stage_4_complete[booster_isp_id]) > 0
    assert len(kerbal_random_stage_4_complete[rt_10_isp_id]) > 0
    assert len(kerbal_random_stage_4_complete[booster_empty_mass_id]) > 0


def test_expression_inferred_graph(kerbal_lpg):
    # inferred graph provides a reliable order of execution for expressions

    all_edge_keys = list(kerbal_lpg.edges.keys())
    all_edge_types = [edg[2] for edg in all_edge_keys]
    implied_edges = [edg for edg in all_edge_keys if edg[2] == "ImpliedParameterFeedforward"]

    assert any([typ == "ImpliedParameterFeedforward" for typ in all_edge_types])
    assert len(implied_edges) == 30

    eig = kerbal_lpg.get_projection("Expression Inferred")

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
            sequence_template=[kerbal_lpg.nodes[part_def]],
            quantities=[10],
        )
        for part_def in part_defs
    }
    assert len(new_instances) == 17


def test_instance_sampling(kerbal_lpg):
    # what we really need here is a fixture that generates a healthy instance dictionary from the playbook phases
    solid_booster_id = "24a0a10e-77ba-4bfa-9618-f2525a8a7042"

    # try creating the boosters under solid stage

    solid_stage_instances = create_set_with_new_instances(
        sequence_template=[kerbal_lpg.nodes[solid_booster_id]],
        quantities=[5],
    )

    solid_booster_instances = create_set_with_new_instances(
        sequence_template=[kerbal_lpg.nodes[solid_booster_id]],
        quantities=[40],
    )

    booster_feature_instances = extend_sequences_by_sampling(
        solid_stage_instances,
        8,
        8,
        solid_booster_instances,
        False,
        {},
        {},
    )

    assert len(booster_feature_instances) == 40
    assert len(booster_feature_instances[4]) == 2
