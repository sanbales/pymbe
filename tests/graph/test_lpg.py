import warnings

import pytest

from pymbe.graph import SysML2LabeledPropertyGraph

SEED_NODE_NAMES = ("FL-T100 Fuel Tank", """RT-5 "Flea" Solid Fuel Booster""")


def test_lpg(kerbal_model):
    lpg = SysML2LabeledPropertyGraph(model=kerbal_model)
    assert len(lpg.nodes) == len(kerbal_model.all_non_relationships)
    assert len(lpg.edges) == len(kerbal_model.all_relationships)

    banded = lpg.get_projection("Banded")
    assert len(banded.nodes) > 0
    assert len(banded.nodes) < len(lpg.nodes)
    assert len(banded.edges) > 0
    assert len(banded.edges) < len(lpg.edges)


def test_spanning_graph(kerbal_model):
    lpg = SysML2LabeledPropertyGraph(model=kerbal_model)
    nodes = [id_ for id_, data in lpg.nodes.items() if data.get("name") in SEED_NODE_NAMES]
    small_directed = lpg.get_spanning_graph(
        graph=lpg.graph,
        seeds=nodes,
        max_distance=1,
        enforce_directionality=True,
    )
    assert len(small_directed.nodes) > 0
    assert len(small_directed.nodes) < len(lpg.nodes)
    assert len(small_directed.edges) > 0
    assert len(small_directed.edges) < len(lpg.edges)

    small_undirected = lpg.get_spanning_graph(
        graph=lpg.graph,
        seeds=nodes,
        max_distance=1,
        enforce_directionality=False,
    )
    assert len(small_undirected.nodes) > len(small_directed.nodes)
    assert len(small_undirected.nodes) < len(lpg.nodes)
    assert len(small_undirected.edges) > len(small_directed.edges)
    assert len(small_undirected.edges) < len(lpg.edges)

    large_directed = lpg.get_spanning_graph(
        graph=lpg.graph,
        seeds=nodes,
        max_distance=2,
        enforce_directionality=True,
    )
    assert len(large_directed.nodes) > len(small_directed.nodes)
    assert len(large_directed.nodes) < len(lpg.nodes)
    assert len(large_directed.edges) > len(small_directed.edges)
    assert len(large_directed.edges) < len(lpg.edges)

    large_undirected = lpg.get_spanning_graph(
        graph=lpg.graph,
        seeds=nodes,
        max_distance=2,
        enforce_directionality=False,
    )
    assert len(large_undirected.nodes) > len(small_undirected.nodes)
    assert len(large_undirected.nodes) < len(lpg.nodes)
    assert len(large_undirected.edges) > len(small_undirected.edges)
    assert len(large_undirected.edges) < len(lpg.edges)

    small_one_seed = lpg.get_spanning_graph(
        graph=lpg.graph,
        seeds=[nodes[0]],
        max_distance=1,
        enforce_directionality=True,
    )
    assert len(small_one_seed.nodes) > 0
    assert len(small_one_seed.nodes) < len(small_directed.nodes)
    assert len(small_one_seed.edges) > 0
    assert len(small_one_seed.edges) < len(small_directed.edges)


def test_path_graph(kerbal_model):
    lpg = SysML2LabeledPropertyGraph(model=kerbal_model)
    source, target = [
        id_ for id_, data in lpg.nodes.items() if data.get("name") in SEED_NODE_NAMES
    ]
    path_graph = lpg.get_path_graph(
        lpg.graph,
        source=source,
        target=target,
        enforce_directionality=False,
    )
    for node in (target, source):
        assert node in path_graph
    assert len(path_graph.nodes) > 0
    assert len(path_graph.nodes) < len(lpg.nodes)
    assert len(path_graph.edges) > 0
    assert len(path_graph.edges) < len(lpg.edges)

    with pytest.warns(UserWarning, match="networkx.exception.NetworkXNoPath"):
        warnings.warn("networkx.exception.NetworkXNoPath", UserWarning)

    path_graph = lpg.get_path_graph(
        lpg.graph,
        source=source,
        target=target,
        enforce_directionality=True,
    )
    assert len(path_graph.nodes) == 0
