# The playbooks here build up random sets of instances from a given model using
# "set building" step for classifiers and features
# NOTE: This playbook is an initial attempt to randomly generate sequences to fill
#       in sets that are interpretations of the user model

import logging
import traceback
from random import randint, sample
from typing import Dict, List

import networkx as nx

from ..graph.lpg import SysML2LabeledPropertyGraph
from ..label import get_label
from ..model import Element, InstanceDictType, Model
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


def random_generator_playbook(
    lpg: SysML2LabeledPropertyGraph,
    name_hints: Dict[str, str] = None,
    filtered_feat_packages: List[Element] = None,
) -> InstanceDictType:
    """
    Main routine to execute a playbook to randomly generate sequences as an interpretation
    of a SysML v2 model

    :param lpg: Labeled propery graph of the M1 model
    :param name_hints: A dictionary to make labeling instances more clean
    :param filtered_feat_packages: A list of packages by which to down filter feature and
        expression sequence templates
    :return: A dictionary of sequences keyed by the id of a given M1 type
    """

    all_elements = lpg.model.elements
    name_hints = name_hints or {}
    filtered_feat_packages = filtered_feat_packages or []

    # PHASE 1: Create a set of instances for part definitions based on usage multiplicities

    # work from part definitions to establish how many definitions are needed

    ptg = lpg.get_projection("Part Typing")
    scg = lpg.get_projection("Part Definition")

    expression_sequences = build_expression_sequence_templates(lpg=lpg)
    feature_sequences = build_sequence_templates(lpg=lpg)

    if filtered_feat_packages:
        expression_sequences = [
            seq
            for seq in expression_sequences
            if all_elements[seq[-1]].owning_package in filtered_feat_packages
        ]
        feature_sequences = [
            seq
            for seq in feature_sequences
            if all_elements[seq[-1]].owning_package in filtered_feat_packages
        ]

    validate_working_data(
        expression_sequences=expression_sequences,
        feature_sequences=feature_sequences,
        model=lpg.model,
    )

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
    # Move through existing sequences and then start to pave further with new steps
    random_generator_playbook_phase_4(lpg.model, expression_sequences, instances_dict)

    # PHASE 5: Interpret connection usages and map ConnectionEnds at M0
    random_generator_playbook_phase_5(lpg, lpg.get_projection("Connection"), instances_dict)

    return instances_dict


def random_generator_phase_1_multiplicities(
    lpg: SysML2LabeledPropertyGraph,
    ptg: nx.DiGraph,
    scg: nx.DiGraph,
) -> Dict[str, int]:
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
            if typ not in scg.nodes:
                full_multiplicities[typ] = mult
            try:
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
            except nx.NetworkXError:
                continue
        else:
            full_multiplicities[typ] = mult

    return full_multiplicities


def random_generator_playbook_phase_1_singletons(
    model: Model,
    scg: nx.DiGraph,
    instances_dict: InstanceDictType,
):
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
    instances_dict: InstanceDictType,
):
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
            for subset_node in bfs_dict[gen]:
                new_superset.extend(instances_dict[subset_node])

            instances_dict[gen] = new_superset


def random_generator_playbook_phase_2_unconnected(
    model: Model,
    instances_dict: InstanceDictType,
):
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
    feature_sequences: List[List[str]],
    instances_dict: InstanceDictType,
):
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
        # skip if the feature is abstract or its owning type is
        last_item = model.elements[feature_sequence[-1]]
        if last_item.isAbstract:
            print(f"Skipped sequence ending in {last_item}")
            continue

        new_sequences = []
        first_type = True
        for type_id in feature_sequence:
            if type_id in instances_dict and not first_type:
                # Don't repeat draws if you encounter the same feature again
                continue
            # sample set will be the last element in the sequence for classifiers
            type_ = model.elements[type_id]
            metatype = type_._metatype
            if metatype in TYPES_FOR_FEATURING:
                feature_types = type_.get("type")
                if isinstance(feature_types, Element):
                    classifier_id = feature_types._id or []
                elif len(feature_types) > 1:
                    raise NotImplementedError("Cannot handle features with multiple types yet!")
                else:
                    classifier_id = feature_types[0]._id
            else:
                classifier_id = type_id

            if first_type:
                if metatype in TYPES_FOR_FEATURING:
                    # TODO: refactor this "hack" for usage at top
                    new_sequences = [instances_dict[classifier_id][0]]
                    if classifier_id not in already_drawn:
                        already_drawn[classifier_id] = list(new_sequences[0])
                else:
                    new_sequences = instances_dict[classifier_id]
            else:
                # We know every type after the first must be a feature in SysML v2
                new_sequences = add_nested_features(
                    already_drawn=already_drawn,
                    classifier_id=classifier_id,
                    feature=type_,
                    instances_dict=instances_dict,
                    new_sequences=new_sequences,
                    model=model,
                )

            first_type = False
            instances_dict[type_id] = new_sequences


def add_nested_features(
    already_drawn: Dict[str, List[str]],
    classifier_id: str,
    feature: Element,
    instances_dict: InstanceDictType,
    new_sequences: List[List[str]],
    model: Model,
):
    """Interpret nested features within a given type."""
    try:
        if classifier_id in already_drawn:
            remaining = [
                item
                for seq in instances_dict[classifier_id]
                for item in seq
                if item not in already_drawn[classifier_id]
            ]
        else:
            remaining = [item for seq in instances_dict[classifier_id] for item in seq]
    except KeyError as exc:
        raise KeyError(
            f"Cannot find type {model.elements[classifier_id]}, id {classifier_id} "
            "in instances dict made so far!"
        ) from exc

    logger.debug("About to extend sequences.")
    logger.debug("New sequences is currently %s", new_sequences)
    logger.debug("Already drawn is currently %s", already_drawn)

    lower_mult = feature_multiplicity(feature, "lower")
    upper_mult = min(feature_multiplicity(feature, "upper"), model.max_multiplicity)

    new_sequences = extend_sequences_by_sampling(
        new_sequences,
        lower_mult,
        upper_mult,
        remaining,
        False,
        {},
    )

    logger.debug("Sequences extended.")
    logger.debug("New sequences is currently %s", new_sequences)

    freshly_drawn = [seq[-1] for seq in new_sequences]
    if classifier_id in already_drawn:
        already_drawn[classifier_id] += freshly_drawn
    else:
        already_drawn[classifier_id] = freshly_drawn

    logger.debug("Already drawn is currently %s", pprint_dict_keys(already_drawn, model))
    return new_sequences


def random_generator_playbook_phase_4(
    model: Model,
    expr_sequences: List[List[str]],
    instances_dict: InstanceDictType,
):
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
        if isinstance(seq_featuring_type, list):
            continue
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
    lpg: SysML2LabeledPropertyGraph,
    cug: nx.DiGraph,
    instances_dict: InstanceDictType,
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
        node = lpg.model.elements[node_id]
        if node._metatype in ("ConnectionUsage", "InterfaceUsage", "SuccessionUsage"):

            connector_ends = node.connectorEnd

            connector_id = node._id

            source_feat_id = node.source[0].chainingFeature[-1]._id
            target_feat_id = node.target[0].chainingFeature[-1]._id

            print(node.source)

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

            instances_dict[connector_ends[0]._id] = extended_source_sequences
            instances_dict[connector_ends[1]._id] = extended_target_sequences


def build_sequence_templates(lpg: SysML2LabeledPropertyGraph) -> List[List[str]]:
    """
    Compute minimal length sequence of M1 types for all features in the model.

    :return: list of lists of Element IDs (as strings) representing feature nesting.
    """
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
    return sorted_feature_groups


def build_expression_sequence_templates(lpg: SysML2LabeledPropertyGraph) -> List[List[str]]:
    """
    Compute minimal length sequence of M1 types for all expression steps in the model.

    :return: list of lists of Element IDs (as strings) representing feature nesting.
    """
    evg = lpg.get_projection("Expression Value")

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


def validate_working_data(
    expression_sequences: List[List[str]],
    feature_sequences: List[List[str]],
    model: Model,
):
    """
    Helper method to check that the user model is valid for instance generation by ensuring
    that all the feature sequence templates of the model have types and multiplicities.
    """
    for sequence in expression_sequences:
        for element_id in sequence:
            # TODO: Add checks for expression sequences
            pass

    for sequence in feature_sequences:
        for element_id in sequence:
            element = model.elements[element_id]
            if element._metatype not in TYPES_FOR_FEATURING:
                continue
            if not element.get("type"):
                raise ValueError(f"Feature {element}, ({element_id}) does not have a type!")
            try:
                for bound in ("upper", "lower"):
                    feature_multiplicity(element, bound)
            except Exception as exc:
                raise ValueError(
                    f"Failed to get '{bound}' multiplicity for {element} ({element_id})"
                ) from exc
