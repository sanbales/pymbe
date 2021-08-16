# The playbooks here build up random sets of instances from a given model using
# "set building" step for classifiers and features
# NOTE: This playbook is an initial attempt to randomly generate sequences to fill
#       in sets that are interpretations of the user model

import logging
import traceback
from random import randint, sample

import networkx as nx

from ..graph.lpg import SysML2LabeledPropertyGraph
from ..label import get_label, get_label_for_id
from ..model import Element, Model
from ..query.metamodel_navigator import safe_feature_data
from ..query.query import feature_multiplicity, roll_up_multiplicity_for_type
from .results import pprint_dict_keys
from .set_builders import (
    create_set_with_new_instances,
    extend_sequences_by_sampling,
    extend_sequences_with_new_expr,
    extend_sequences_with_new_value_holder,
)

logger = logging.getLogger(__name__)


TYPES_FOR_FEATURING = (
    "ActionUsage",
    "AttributeUsage",
    "ConnectionUsage",
    "InterfaceUsage",
    "PartUsage",
    "PortUsage",
)

TYPES_FOR_ROLL_UP_MULTIPLICITY = (
    "ActionDefinition",
    "AttributeDefinition",
    "ConnectionDefinition",
    "DataType",
    "InterfaceDefinition",
    "PartDefinition",
    "PortDefinition",
)


def random_generator_playbook(lpg: SysML2LabeledPropertyGraph, name_hints: dict = None) -> dict:
    """
    Main routine to execute a playbook to randomly generate sequences as an interpretation
    of a SysML v2 model

    :param lpg: Labeled propery graph of the M1 model
    :param name_hints: A dictionary to make labeling instances more clean
    :return: A dictionary of sequences keyed by the id of a given M1 type
    """

    all_elements = lpg.model.elements
    name_hints = name_hints or {}
    can_interpret = validate_working_data(lpg)

    if not can_interpret:
        return {}

    # PHASE 1: Create a set of instances for part definitions based on usage multiplicities

    # work from part definitions to establish how many definitions are needed

    ptg = lpg.get_projection("Part Typing")
    scg = lpg.get_projection("Part Definition")

    feature_sequences = build_sequence_templates(lpg=lpg)

    full_multiplicities = random_generator_phase_1_multiplicities(lpg, ptg, scg)

    instances_dict = {
        type_id: create_set_with_new_instances(
            sequence_template=[all_elements[type_id]],
            quantities=[number],
            name_hints=name_hints,
        )
        for type_id, number in full_multiplicities.items()
    }

    # pick up the definitions that aren't matched to a usage yet

    random_generator_playbook_phase_1_singletons(lpg.model, scg, instances_dict)

    # PHASE 2: Combine sets of instances into sets that are marked as more general in the
    #          user model

    # "Roll up" the graph through a breadth-first search from the most general classifier down to
    # the most specific and then move in reverse order from the specific (subset) to
    # the general (superset)

    random_generator_playbook_phase_2_rollup(scg, instances_dict)

    # Fill in any part definitions that still don't have instances yet (because they get filtered
    # out by the Part Definition pre-defined graph (neither typed nor subclassed))

    random_generator_playbook_phase_2_unconnected(lpg.model, instances_dict)

    # PHASE 3: Expand the dictionaries out into feature sequences by pulling from instances
    #          developed here

    random_generator_playbook_phase_3(lpg.model, feature_sequences, instances_dict)

    # PHASE 4: Expand sequences to support computations

    expr_sequences = build_expression_sequence_templates(lpg=lpg)

    # for indx, seq in enumerate(expr_sequences):
    #    print("Sequence number " + str(indx))
    #    for item in seq:
    #        print(get_label(all_elements[item], all_elements) + ", id = " + item)

    # Move through existing sequences and then start to pave further with new steps

    random_generator_playbook_phase_4(lpg.model, expr_sequences, instances_dict)

    # attached connector ends to sequences(

    random_generator_playbook_phase_5(lpg, lpg.get_projection("Connection"), instances_dict)

    return instances_dict


def random_generator_phase_1_multiplicities(
    lpg: SysML2LabeledPropertyGraph,
    ptg: nx.DiGraph,
    scg: nx.DiGraph,
) -> dict:
    """
    Calculates the multiplicities for classifiers in the considered model
    to support initial generation.

    :param lpg: Active SysML graph
    :param ptg: Part Typing Graph projection from the LPG
    :param scg: Subclassing Graph projection from the LPG
    :return: dictionary of multiplicities for instance generation, indexed by classifier ID
    """
    elements = lpg.model.elements
    abstracts = [node_id for node_id in ptg.nodes if elements[node_id]._is_abstract]

    # find the maximal amount of types directly based on instances
    type_multiplicities = {
        node_id: roll_up_multiplicity_for_type(
            lpg,
            elements[node_id],
            "upper",
        )
        for node_id in ptg.nodes
        if elements[node_id]._metatype in TYPES_FOR_ROLL_UP_MULTIPLICITY
    }

    full_multiplicities = {}
    # look at all the types in the feature sequences
    for typ, mult in type_multiplicities.items():
        if typ in abstracts:
            specifics = list(scg.successors(typ))
            taken = 0
            no_splits = len(specifics)
            # need to sub-divide the abstract quantities
            for index, specific in enumerate(specifics):
                if index < (no_splits - 1):
                    draw = randint(0, mult)
                    taken = taken + draw
                else:
                    draw = mult - taken
                full_multiplicities[specific] = draw
        else:
            full_multiplicities[typ] = mult

    return full_multiplicities


def random_generator_playbook_phase_1_singletons(
    model: Model,
    scg: nx.DiGraph,
    instances_dict: dict,
) -> None:
    """
    Calculates instances for classifiers that aren't directly typed (but may have
    members or be superclasses for model elements that have sequences generated for them).

    :param lpg: Active SysML graph
    :param scg: Subclassing Graph projection from the LPG
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    # need to generate single instances at leaves that don't match types
    leaves = [node for node in scg.nodes if scg.out_degree(node) == 0]

    new_instances = {
        leaf: create_set_with_new_instances(
            sequence_template=[model.elements[leaf]],
            quantities=[1],
        )
        for leaf in leaves
        if leaf not in instances_dict
    }
    instances_dict.update(new_instances)


def random_generator_playbook_phase_2_rollup(
    scg: nx.DiGraph,
    instances_dict: dict,
) -> None:
    """
    Build up set of sequences for classifiers by taking the union of sequences
    already generated for the classifier subclasses.

    :param lpg: Active SysML graph
    :param scg: Subclassing Graph projection from the LPG
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    roots = [node for node in scg.nodes if scg.in_degree(node) == 0]

    for root in roots:
        bfs_dict = dict(nx.bfs_successors(scg, root))
        bfs_list = list(bfs_dict.keys())
        bfs_list.reverse()

        for gen in bfs_list:
            new_superset = []
            # use the BFS dictionary to be assured that everything is covered
            # update_dict = generate_superset_instances(scg, gen, visited_nodes, instances_dict)

            for subset_node in bfs_dict[gen]:
                new_superset.extend(instances_dict[subset_node])

            instances_dict[gen] = new_superset


def random_generator_playbook_phase_2_unconnected(
    model: Model,
    instances_dict: dict,
) -> None:
    """
    Final pass to generate sequences for classifiers that haven't been given sequences yet.

    :param all_elements: Full dictionary of elements in the working memory
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    finishing_list = [
        element
        for element in model.elements.values()
        if element._metatype == "PartDefinition" and element._id not in instances_dict
    ]
    new_instances = {
        element._data["@id"]: create_set_with_new_instances(
            sequence_template=[element],
            quantities=[1],
        )
        for element in finishing_list
    }
    instances_dict.update(new_instances)


def random_generator_playbook_phase_3(
    model: Model,
    feature_sequences: list,
    instances_dict: dict,
) -> list:
    """
    Begin generating interpreting sequences for Features in the model by extending
    classifier sequences with randomly selected instances of classifiers that type
    nested features.

    :param feature_sequences: Sequences that represent the nesting structure of the features
    :param all_elements: Full dictionary of elements in the working memory
    :param lpg: Active SysML graph
    :param ptg: Part Typing Graph projection from the LPG
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: (Temporarily return a trace of actions) None - side effect is addition of new
        instances to the instances dictionary
    """
    logger.debug("Starting things up")
    already_drawn = {}
    for feature_sequence in feature_sequences:
        new_sequences = []
        for index, feature_id in enumerate(feature_sequence):
            if feature_id in instances_dict and index > 0:
                # don't repeat draws if you encounter the same feature again
                continue
            # sample set will be the last element in the sequence for classifiers
            feature = model.elements[feature_id]
            metatype = feature._metatype
            if metatype in TYPES_FOR_FEATURING:
                types = safe_feature_data(feature, "type")
                if isinstance(types, Element):
                    typ = (types._id or [])
                else:
                    if not types:
                        raise NotImplementedError(
                            "Cannot handle untyped features! Tried on "
                            f"{get_label_for_id(feature_id, model)}, "
                            f"id = {feature_id}"
                        )
                    if len(types) > 1:
                        raise NotImplementedError(
                            "Cannot handle features with multiple types yet!"
                        )
                    typ = types[0]
            else:
                typ = feature_id

            if index == 0:
                if metatype in TYPES_FOR_FEATURING:
                    # hack for usage at top
                    new_sequences = [instances_dict[typ][0]]
                    if typ not in already_drawn:
                        already_drawn[typ] = list(new_sequences[0])
                else:
                    new_sequences = instances_dict[typ]
            else:
                if typ in already_drawn:
                    remaining = [
                        item
                        for seq in instances_dict[typ]
                        for item in seq
                        if item not in already_drawn[typ]
                    ]
                else:
                    remaining = [item for seq in instances_dict[typ] for item in seq]

                logger.info("About to extend sequences.")
                logger.info("New sequences is currently %s", new_sequences)
                logger.info("Already drawn is currently %s", already_drawn)

                new_sequences = extend_sequences_by_sampling(
                    new_sequences,
                    feature_multiplicity(feature, "lower"),
                    feature_multiplicity(feature, "upper"),
                    remaining,
                    False,
                    {},
                )

                logger.info("Sequences extended.")
                logger.info("New sequences is currently %s", new_sequences)

                freshly_drawn = [seq[-1] for seq in new_sequences]
                if typ in already_drawn:
                    already_drawn[typ] += freshly_drawn
                else:
                    already_drawn[typ] = freshly_drawn

                logger.info(
                    "Already drawn is currently %s", pprint_dict_keys(already_drawn, model)
                )

            instances_dict[feature_id] = new_sequences


def random_generator_playbook_phase_4(
    model: Model,
    expr_sequences: list,
    instances_dict: dict,
) -> None:
    """
    Generate interpreting sequences for Expressions in the model

    :param expr_sequences: Sequences that represent the membership structure for expressions
        in the model and the features to which expressions provide values
    :param lpg: Active SysML graph
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    all_elements = model.elements
    for expr_seq in expr_sequences:
        new_sequences = []
        # get the featuring type of the first expression
        seq_featuring_type = safe_feature_data(all_elements[expr_seq[0]], "featuringType")
        # FIXME: I don't know what it means for binding connectors to own these expressions,
        #        but need to figure out eventually
        if seq_featuring_type["@type"] == "BindingConnector":
            continue
        new_sequences = instances_dict[seq_featuring_type["@id"]._id]

        for feature_id in expr_seq:
            # sample set will be the last element in the sequence for classifiers
            feature = all_elements[feature_id]
            feature_metatype = feature._metatype
            if feature_id in instances_dict:
                new_sequences = instances_dict[feature_id]
            else:
                if any(key in feature_metatype for key in ("Expression", "Literal")):
                    # Get the element type(s)
                    types: list = feature._data.get("type") or []
                    if isinstance(types, dict):
                        types = [types]
                    type_names = [
                        all_elements[type_["@id"]]._data.get("name")
                        for type_ in types
                        if type_ and "@id" in type_
                    ]
                    type_names = [str(type_name) for type_name in type_names if type_name]

                    new_sequences = extend_sequences_with_new_expr(
                        new_sequences,
                        get_label(feature),
                        feature._data,
                    )
                elif feature_metatype == "Feature":
                    new_sequences = extend_sequences_with_new_value_holder(
                        new_sequences,
                        feature.name,
                        feature._data,
                    )
                else:
                    new_sequences = extend_sequences_by_sampling(
                        new_sequences,
                        1,
                        1,
                        [],
                        True,
                        feature._data,
                    )
                instances_dict[feature_id] = new_sequences


def random_generator_playbook_phase_5(
    lpg: SysML2LabeledPropertyGraph, cug: nx.DiGraph, instances_dict: dict
):
    """
    Generate instances for connector usages and their specializations and randomly
    connect ends to legal sources and targets
    :param lpg: Active SysML graph
    :param cug: A connector usage graph projection to see where their source/targets are linked
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """

    # Generate sequences for connection and interface ends
    for node_id in list(cug.nodes):
        node = lpg.nodes[node_id]
        if node["@type"] in ("ConnectionUsage", "InterfaceUsage", "SuccessionUsage"):

            connector_ends = node["connectorEnd"]

            connector_id = node["@id"]

            source_feat_id = node["source"][0]["@id"]
            target_feat_id = node["target"][0]["@id"]

            source_sequences = instances_dict[source_feat_id]
            target_sequences = instances_dict[target_feat_id]

            connectors = instances_dict[connector_id]

            extended_source_sequences = []
            extended_target_sequences = []

            min_side = min(len(source_sequences), len(target_sequences))
            max_side = max(len(source_sequences), len(target_sequences))

            if len(source_sequences) <= len(target_sequences):
                source_indices = list(range(0, min_side))
                other_steps = sample(range(0, min_side), (max_side - min_side))
                source_indices.extend(other_steps)
                target_indices = sample(range(0, max_side), max_side)
            else:
                source_indices = sample(range(0, max_side), max_side)
                other_steps = sample(range(0, min_side), (max_side - min_side))
                target_indices = list(range(0, min_side))
                target_indices.extend(other_steps)

            # taking over indx to wrap around
            indx = 0

            for seq in connectors:
                new_source_seq = []
                new_target_seq = []

                for item in seq:
                    new_source_seq.append(item)
                    new_target_seq.append(item)

                for jndx, item in enumerate(source_sequences[source_indices[indx]]):
                    if jndx > 0:
                        new_source_seq.append(item)

                for jndx, item in enumerate(target_sequences[target_indices[indx]]):
                    if jndx > 0:
                        new_target_seq.append(item)

                extended_source_sequences.append(new_source_seq)
                extended_target_sequences.append(new_target_seq)

                if indx >= len(target_indices) - 1:
                    indx = 0
                else:
                    indx = indx + 1

            instances_dict[connector_ends[0]["@id"]] = extended_source_sequences
            instances_dict[connector_ends[1]["@id"]] = extended_target_sequences


def build_sequence_templates(lpg: SysML2LabeledPropertyGraph) -> list:
    part_featuring_graph = lpg.get_projection("Part Featuring")
    sorted_feature_groups = []
    for comp in nx.connected_components(part_featuring_graph.to_undirected()):
        connected_sub = nx.subgraph(part_featuring_graph, list(comp))
        leaves = [node for node in connected_sub.nodes if connected_sub.in_degree(node) == 0]
        roots = [node for node in connected_sub.nodes if connected_sub.out_degree(node) == 0]
        for leaf in leaves:
            for root in roots:
                try:
                    leaf_path = nx.shortest_path(connected_sub, leaf, root)
                    sorted_feature_groups.append(leaf_path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    logger.debug("Could not find path: %s", traceback.format_exc())

        # TODO: Look into adding the topologically sorted connected subcomponents
        # sorted_feature_groups.append(
        #     [node for node in nx.topological_sort(connected_sub)]
        # )

    return sorted_feature_groups


def generate_superset_instances(
    part_def_graph: nx.MultiDiGraph,
    superset_node: str,
    visited_nodes: set,
    instances_dict: dict,
) -> dict:
    """
    Take specific classifiers and push the calculated instances to more general classifiers

    :return:
    """
    new_superset = []
    subset_nodes = part_def_graph.predecessors(superset_node)
    if all(subset_node in visited_nodes for subset_node in subset_nodes):
        for subset_node in part_def_graph.predecessors(superset_node):
            new_superset.extend(instances_dict[subset_node])
    else:
        return {}

    return {superset_node: new_superset}


def build_expression_sequence_templates(lpg: SysML2LabeledPropertyGraph) -> list:
    evg = lpg.get_projection("Expression Value")

    # FIXME: Need projections to work correctly
    # TODO: @Bjorn: should we remove all the implied edges? We could add a key to them
    to_remove = [edge for edge in evg.edges if edge[2] == "ImpliedParameterFeedforward"]

    for source, target, *_ in to_remove:
        evg.remove_edge(source, target)

    sorted_feature_groups = []
    for comp in nx.connected_components(evg.to_undirected()):
        connected_sub = nx.subgraph(evg, list(comp))
        leaves = [node for node in connected_sub.nodes if connected_sub.out_degree(node) == 0]
        roots = [node for node in connected_sub.nodes if connected_sub.in_degree(node) == 0]
        for root in roots:
            for leaf in leaves:
                try:
                    leaf_path = nx.shortest_path(connected_sub, root, leaf)
                    sorted_feature_groups.append(leaf_path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    logger.debug("Could not find path: %s", traceback.format_exc())

    return sorted_feature_groups


def validate_working_data(lpg: SysML2LabeledPropertyGraph) -> bool:
    """
    Helper method to check that the user model is valid for instance generation

    :return: A Boolean indicating that the user model is ready to be interpreted
    """
    # FIXME: Convert to the element-model style for better accuracy

    # check that all the elements of the graph are in fact proper model elements
    for id_, non_relation in lpg.model.all_non_relationships.items():
        try:
            non_relation["@type"]
        except KeyError:
            print(f"No metatype found in {non_relation}, id = '{id_}', name = {(lpg.model.elements[id_].name or '')}")
            return False
        except TypeError:
            print(f"Expecting dict of model element data, got = {non_relation}")
    return True
