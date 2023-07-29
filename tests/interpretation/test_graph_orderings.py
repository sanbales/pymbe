import networkx as nx


def test_part_def_graph_ordering(kerbal_lpg):
    pdg = kerbal_lpg.get_projection("Part Definition")
    rocket_part = "63f5c455-261b-4a80-9a3b-5a9bef2361da"
    flea = "68f08797-0e68-47b1-bad5-9e734af2742f"
    hammer = "628929a4-1dc2-4c34-aefb-2653faaa46fe"
    fuel_tank_section = "b51e6a60-fdf3-401c-b267-4d3d6aeaa19d"

    for comp in nx.connected_components(pdg.to_undirected()):
        connected_sub = nx.subgraph(pdg, list(comp))
        sorted_sub = list(nx.topological_sort(connected_sub))

        if rocket_part in sorted_sub:
            assert flea in sorted_sub
            assert hammer in sorted_sub
            assert fuel_tank_section in sorted_sub

            assert sorted_sub.index(flea) > sorted_sub.index(rocket_part)
            assert sorted_sub.index(hammer) > sorted_sub.index(rocket_part)
            assert sorted_sub.index(fuel_tank_section) > sorted_sub.index(rocket_part)
