import networkx as nx
import pytest


def test_graph_load(kerbal_lpg):
    assert len(kerbal_lpg.nodes) == 152
    assert (
        len(kerbal_lpg.edges) == 380 - 152
    )  # nodes + edges should be all elements from the client


@pytest.mark.skip("Need to refactor tests, after upgrades")
def test_graph_projection_part_def_node_filter(kerbal_lpg, kerbal_ids_by_type):
    pdg = kerbal_lpg.get_projection("Part Definition")

    # we actually lose a node on this one ...
    pdg_set = set(pdg.nodes.keys())
    kerbal_set = set(kerbal_ids_by_type["PartDefinition"])

    dropped_nodes = list(kerbal_set - pdg_set)
    test_node = dropped_nodes[0]

    assert kerbal_lpg.nodes[test_node]["name"] == "Rocket"

    assert len(pdg.nodes) == len(kerbal_ids_by_type["PartDefinition"]) - 1


@pytest.mark.skip("Need to refactor tests, after upgrades")
def test_graph_projection_part_def_components(kerbal_lpg):
    pdg = kerbal_lpg.get_projection("Part Definition")

    assert len(list(nx.connected_components(pdg.to_undirected()))) == 2


def test_graph_projection_part_def_components_are_dags(kerbal_lpg):
    pdg = kerbal_lpg.get_projection("Part Definition")

    for comp in nx.connected_components(pdg.to_undirected()):
        connected_sub = nx.subgraph(pdg, list(comp))
        assert nx.is_directed_acyclic_graph(connected_sub)


def test_graph_projection_part_def_edges(kerbal_lpg):
    # check that nothing but Superclassing edges are allowed by the filter
    pdg = kerbal_lpg.get_projection("Part Definition")

    assert all(edge[2] == "Superclassing^-1" for edge in pdg.edges)
