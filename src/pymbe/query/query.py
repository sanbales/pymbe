# a set of queries to run on Labeled Property Graphs
import networkx as nx
import numpy as np

from ..graph.lpg import SysML2LabeledPropertyGraph
from .metamodel_navigator import *
from ..interpretation.results import *


def roll_up_lower_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: dict,
) -> int:
    return roll_up_multiplicity(
        lpg=lpg,
        feature=feature,
        bound="lower",
    )


def roll_up_upper_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: dict,
) -> int:
    return roll_up_multiplicity(
        lpg=lpg,
        feature=feature,
        bound="upper",
    )


def roll_up_multiplicity(
    lpg: SysML2LabeledPropertyGraph,
    feature: dict,
    bound: str,
) -> int:
    banded_featuring_graph = lpg.get_projection("Expanded Banded Graph")

    # FIXME: Need projections to work correctly

    to_remove = []

    for edg in banded_featuring_graph.edges:
        if edg[2] == 'ImpliedParameterFeedforward':
            to_remove.append(edg)

    for remover in to_remove:
        banded_featuring_graph.remove_edge(remover[0], remover[1])

    banded_roots = [
        node
        for node in banded_featuring_graph.nodes
        if banded_featuring_graph.out_degree(node) < 1
    ]

    all_elements = lpg.nodes

    corrected_mult = 1
    for part_tree_root in banded_roots:
        try:
            part_path = nx.shortest_path(
                banded_featuring_graph,
                source=feature['@id'],
                target=part_tree_root,
            )
            # TODO: check that the path actually exists
            corrected_mult = np.prod([
                feature_multiplicity(all_elements[node], all_elements, bound)
                for node in part_path
            ])
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            # nothing to roll up, so just use own multiplicity
            corrected_mult = feature_multiplicity(feature, all_elements, bound)

    return corrected_mult

def roll_up_multiplicity_for_type(
    lpg: SysML2LabeledPropertyGraph,
    typ: dict,
    bound: str
) -> int:

    rdg = lpg.get_projection("Redefinition Graph")
    # FIXME: Need projections to work correctly

    to_remove = []

    for edg in rdg.edges:
        if edg[2] == 'ImpliedParameterFeedforward':
            to_remove.append(edg)

    for remover in to_remove:
        rdg.remove_edge(remover[0], remover[1])

    running_total = 0
    ptg = lpg.get_projection("Part Typing Graph")

    if typ['@id'] in ptg.nodes:
        feat_ids = ptg.predecessors(typ['@id'])
        for feat_id in feat_ids:
            running_total += roll_up_multiplicity(
                lpg,
                lpg.nodes[feat_id],
                bound
            )
            if feat_id in rdg:
                redef_ids = rdg.predecessors(feat_id)
                for redef_id in redef_ids:
                    running_total += roll_up_multiplicity(
                        lpg,
                        lpg.nodes[redef_id],
                        bound
                )
        return running_total
    else:
        return 0
