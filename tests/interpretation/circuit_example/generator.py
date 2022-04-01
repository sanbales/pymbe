import traceback
import typing as ty

import networkx as nx

import pymbe.api as pm
from pymbe.interpretation.interp_playbooks import random_generator_playbook
from pymbe.model import Element

from .openmdao import execute_interpretation

__all__ = ("get_circuit_data",)


def get_unique_connections(instances, feature):
    source_feat, target_feat = feature
    m0_connector_ends = [
        (tuple(source), tuple(target))
        for source, target in zip(instances[source_feat._id], instances[target_feat._id])
    ]

    m0_connector_ends = tuple(
        {(source, target) for source, target in m0_connector_ends if source[:-1] != target[:-1]}
    )

    unique_connections = {
        (source[-2:], target[-2:]): (source, target) for source, target in m0_connector_ends
    }
    return tuple((source, target) for source, target in unique_connections.values())
    # return tuple((source, target) for source, target in m0_connector_ends)


def get_circuit_data(interpretation: dict, connector_feature: Element):
    """
    A very rudimentary filter for simple circuit connectors.

    Filters out self-connections and duplicate connectors between the same m0 pins.
    """
    unique_connections = get_unique_connections(interpretation, connector_feature)
    circuit_data = dict(interpretation=interpretation)

    def get_el_name(element):
        name = element.name
        name, num = name.split("#")
        name = name if name == "EMF" else name[0]
        return name + num

    def legal_predecessors(graph, node, valids):
        all_pred = set(graph.predecessors(node))
        return {pred for pred in all_pred if pred in valids}

    def legal_successors(graph, node, valids):
        all_suc = set(graph.successors(node))
        return {pred for pred in all_suc if pred in valids}

    graph = nx.DiGraph()
    edges = [
        ((get_el_name(source[-2]), "Pos"), (get_el_name(target[-2]), "Neg"))
        for source, target in unique_connections
    ]
    nodes = {node for node, _ in sum(map(list, edges), [])}
    edges += [((node, "Neg"), (node, "Pos")) for node in nodes if not node.startswith("EMF")]
    graph.add_edges_from(edges)

    emf_nodes = {node for node in nodes if node.upper().startswith("EMF")}
    if not emf_nodes:
        raise RuntimeError("There are not EMF nodes in the graph!")

    if len(emf_nodes) > 1:
        raise RuntimeError(
            f"Found {len(emf_nodes)} EMF nodes ({emf_nodes})!  There should only be one!"
        )
    emf_node = emf_nodes.pop()
    emf_pos_name = (emf_node, "Pos")
    emf_neg_name = (emf_node, "Neg")

    # find all paths in the graph from EMF positive side to EMF negative side

    flows = list(nx.all_simple_paths(graph, emf_pos_name, emf_neg_name))
    flow_edges = list(nx.all_simple_edge_paths(graph, emf_pos_name, emf_neg_name))

    flowed_nodes = {node for flow in flows for node in flow}

    circuit_data["valid_edges"] = set(sum(flow_edges, []))

    valid_graph = nx.DiGraph()
    valid_graph.add_edges_from(circuit_data["valid_edges"])

    # look at junctions in the graph

    # split junctions - where a positive pin has multiple outputs
    circuit_data["split_junctions"] = {
        node: legal_successors(valid_graph, node, flowed_nodes)
        for node in graph.nodes
        if len(legal_successors(valid_graph, node, flowed_nodes)) > 1
    }

    # join junctions - where a negative pin has multiple inputs
    circuit_data["join_junctions"] = {
        node: legal_predecessors(valid_graph, node, flowed_nodes)
        for node in graph.nodes
        if len(legal_predecessors(valid_graph, node, flowed_nodes)) > 1
    }

    circuit_data["both_ways_junctions"] = {
        node
        for node in graph.nodes
        if len(legal_successors(valid_graph, node, flowed_nodes)) > 1
        if node in sum(map(list, circuit_data["join_junctions"].values()), [])
    }

    circuit_data["number_loops"] = len(flows)
    circuit_data["valid_nodes"] = flowed_nodes

    circuit_data["loop_edges"] = flow_edges

    circuit_data["cleaned_graph"] = valid_graph
    circuit_data["plain_graph"] = graph

    # EMF positive is always in loop #1
    circuit_data["emf_pins"] = {"+": emf_pos_name, "-": emf_neg_name}

    # compute the independent loops on the graph
    circuit_data["loop_unique_edges"] = []
    if circuit_data["number_loops"] > 1:
        encountered_edges = set()
        for indx in range(circuit_data["number_loops"]):
            new_edges = []
            for edg in flow_edges[indx]:
                if edg not in encountered_edges:
                    new_edges.append(edg)
                    encountered_edges.add(edg)
            circuit_data["loop_unique_edges"].append(new_edges)
    else:
        circuit_data["loop_unique_edges"] = list(flow_edges)

    return circuit_data


def make_circuit_interpretations(
    m1_model: pm.Model,
    required_interpretations: int = 1,
    max_tries: int = 20,
    print_exceptions: bool = True,
    on_attempt: ty.Callable = None,
) -> tuple:
    """Returns a tuple of data on valid randomly generated M0 interpretations."""
    num_tries = 0
    interpretations = []
    circuit_pkg = m1_model.ownedElement["Circuit Builder"]
    connector_feature = circuit_pkg.ownedElement["Circuit"].ownedMember["Part to Part"].endFeature
    while len(interpretations) < required_interpretations and num_tries < max_tries:
        num_tries += 1
        if on_attempt is not None:
            on_attempt()
        try:
            interpretation = random_generator_playbook(
                m1=m1_model,
                filtered_feat_packages=[circuit_pkg],
            )
            interpretation_data = get_circuit_data(interpretation, connector_feature)
            graph: nx.DiGraph = interpretation_data["cleaned_graph"]
            assert len(graph.nodes) > 0, "Graph has no nodes!"
            assert len(graph.edges) > 0, "Graph has no edges!"
            interpretation_data["problem"] = execute_interpretation(
                interpretation_data=interpretation_data,
                print_exceptions=print_exceptions,
            )
            interpretations += [interpretation_data]
        except:  # pylint: disable=bare-except  # noqa: E722
            if print_exceptions:
                print(
                    f">>> Failed to generate interpretation try {num_tries}!\n",
                    f"\n\tTRACEBACK:\n\t{traceback.format_exc()}",
                )
    return tuple(interpretations)
