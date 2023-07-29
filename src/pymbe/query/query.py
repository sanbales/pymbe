# a set of queries to run on Labeled Property Graphs
import math
from typing import List
from warnings import warn

import networkx as nx

from ..graph.lpg import SysML2LabeledPropertyGraph
from ..label import get_label
from ..model import Element
from .metamodel_navigator import feature_multiplicity


def roll_up_lower_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: Element,
) -> int:
    return roll_up_multiplicity(
        lpg=lpg,
        feature=feature,
        bound="lower",
    )


def roll_up_upper_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: Element,
) -> int:
    return roll_up_multiplicity(
        lpg=lpg,
        feature=feature,
        bound="upper",
    )


def roll_up_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: Element,
    bound: str,
) -> int:
    max_multiplicity = lpg.model.max_multiplicity

    banded_featuring_graph = lpg.get_projection("Expanded Banded")

    # Need to check that we are dealing with a DAG, otherwise take care with cycles
    if not nx.is_directed_acyclic_graph(banded_featuring_graph):
        warn("Banded featuring graph is not an acyclic digraph!!!")

    banded_roots = [
        node
        for node in banded_featuring_graph.nodes
        if banded_featuring_graph.out_degree(node) < 1
    ]

    model = lpg.model
    total_mult = 0
    feature_id = feature._id
    for part_tree_root in banded_roots:
        # case where the usage is actually top of a nesting set
        if feature_id == part_tree_root:
            total_mult = 1

        try:
            part_paths = nx.all_simple_paths(
                banded_featuring_graph,
                source=feature_id,
                target=part_tree_root,
            )
            for part_path in part_paths:
                # TODO: check that the path actually exists
                corrected_mult = math.prod(
                    [
                        min(
                            feature_multiplicity(model.elements[element_id], bound),
                            max_multiplicity,
                        )
                        for element_id in part_path
                    ]
                )
                total_mult += corrected_mult
        except nx.NetworkXNoPath:
            print("Found no path when rolling up multiplicity.")
        except nx.NodeNotFound:
            # nothing to roll up, so just use own multiplicity
            total_mult = min(feature_multiplicity(feature, bound), max_multiplicity)

    return total_mult


def roll_up_multiplicity_for_type(
    lpg: SysML2LabeledPropertyGraph,
    element: Element,
    bound: str,
) -> int:
    rdg = lpg.get_projection("Redefinition and Subsetting")
    cug = lpg.get_projection("Connection")

    running_total = 0
    ptg = lpg.get_projection("Part Typing")

    all_elements = lpg.model.elements
    if element._id in ptg.nodes:
        feat_ids = get_features_typed_by_type(lpg, element._id)
        for feat_id in feat_ids:
            if lpg.model.elements[feat_id].isAbstract:
                continue
            running_total += roll_up_multiplicity(
                lpg,
                all_elements[feat_id],
                bound,
            )
            if feat_id in rdg and feat_id not in cug:
                redef_ids = rdg.predecessors(feat_id)
                for redef_id in redef_ids:
                    running_total += roll_up_multiplicity(
                        lpg,
                        all_elements[redef_id],
                        bound,
                    )
        return running_total
    return 0


def get_types_for_feature(
    lpg: SysML2LabeledPropertyGraph,
    feature_id: str,
) -> List[str]:
    ptg = lpg.get_projection("Part Typing")
    rdg = lpg.get_projection("Redefinition and Subsetting")

    # approach is to see source and target ends of feature typing and then see
    # if there is a redefinition path back to the requested feature

    types = []
    if feature_id in list(ptg.nodes):
        types = list(ptg.successors(feature_id))
    elif feature_id in list(rdg.nodes):
        feature_neighbors = list(nx.dfs_preorder_nodes(rdg, feature_id))
        for neighbor in feature_neighbors:
            # if the redefinitions along the way point to types, include those types
            if neighbor in list(ptg.nodes):
                types += list(ptg.successors(neighbor))

    return types


def get_features_typed_by_type(
    lpg: SysML2LabeledPropertyGraph,
    type_id: str,
) -> list:
    ptg = lpg.get_projection("Part Typing")
    rdg = lpg.get_projection("Redefinition and Subsetting")

    # approach is to look at clusters of redefinitions and see if the most redefined
    # connects to the type

    features = []

    if type_id in list(ptg.nodes):
        features = list(ptg.predecessors(type_id))
    elif type_id in list(rdg.nodes):
        for comp in nx.connected_components(rdg.to_undirected()):
            connected_sub = nx.subgraph(rdg, list(comp))
            roots = [node for node in connected_sub.nodes if connected_sub.out_degree(node) == 0]
            for root in roots:
                if root in list(ptg.nodes):
                    for item in list(comp):
                        if item not in features:
                            features += item

    return features


def build_element_owner_sequence(element: Element, seq: List[Element] = None) -> list:
    seq = seq or []
    if element.owner is None:
        return seq
    seq.append(element)

    return build_element_owner_sequence(
        element.owner,
        seq,
    )


def calculate_signature(element: Element) -> str:
    owned_sequence = build_element_owner_sequence(element)

    sig_seq = []
    for item in owned_sequence:
        name = item.get("name")
        if name:
            sig_seq.append(name)
        else:
            sig_seq.append(get_label(item))

    sig_seq.reverse()

    return "::".join(sig_seq)
