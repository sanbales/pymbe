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
from ..label import get_label, get_label_for_id
from ..model import Element, InstanceDictType, Model
from ..query.query import feature_multiplicity, roll_up_multiplicity_for_type
from .results import pprint_dict_keys
from .set_builders import (
    create_set_with_new_instances,
    extend_sequences_by_sampling,
    extend_sequences_with_new_expr,
    extend_sequences_with_new_instance,
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
    "StateDefinition",
)

TYPES_FOR_CONNECTOR_INSTANCES = ("ConnectionUsage", "InterfaceUsage", "SuccessionUsage")


def random_generator_playbook(
    lpg: SysML2LabeledPropertyGraph,
    name_hints: Dict[str, str] = None,
    filtered_feat_packages: List[Element] = None,
    phase_limit: int = 10,
) -> InstanceDictType:
    """
    Main routine to execute a playbook to randomly generate sequences as an interpretation
    of a SysML v2 model

    :param lpg: Labeled property graph of the M1 model
    :param name_hints: A dictionary to make labeling instances more clean
    :param filtered_feat_packages: A list of packages by which to down filter feature and
        expression sequence templates
    :param phase_limit: Stop process before running the specified phase
    :return: A dictionary of sequences keyed by the id of a given M1 type
    """

    all_elements = lpg.model.elements
    name_hints = name_hints or {}
    filtered_feat_packages = filtered_feat_packages or []

    # all_feature_sequences = build_sequence_templates(lpg=lpg)

    expression_sequences = build_expression_sequence_templates(lpg=lpg)
    feature_sequences = build_banded_sequence_templates(
        lpg=lpg, package_filter=filtered_feat_packages
    )

    if filtered_feat_packages:
        expression_sequences = [
            seq
            for seq in expression_sequences
            if all_elements[seq[-1]].owning_package in filtered_feat_packages
        ]

    validate_working_data(
        expression_sequences=expression_sequences,
        feature_sequences=feature_sequences,
        model=lpg.model,
    )

    instances_dict = {}

    # PHASE 3: Expand the dictionaries out into feature sequences by pulling from instances
    #          developed here

    if phase_limit < 3:
        return instances_dict

    random_generator_playbook_phase_3_new_instances(lpg.model, feature_sequences, instances_dict)

    # do one rollup for features (subsetting and redefinition)
    # and one for classifiers (subclassificaiton and typing)

    random_generator_playbook_phase_3_rollup(
        lpg.model, lpg.get_projection("Redefinition and Subsetting"), instances_dict
    )

    random_generator_playbook_phase_3_rollup(
        lpg.model, lpg.get_projection("Generalization"), instances_dict
    )

    if phase_limit < 4:
        return instances_dict

    # PHASE 4: Expand sequences to support computations
    # Move through existing sequences and then start to pave further with new steps
    random_generator_playbook_phase_4(lpg.model, expression_sequences, instances_dict)

    if phase_limit < 5:
        return instances_dict

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

    :param model: A SysML v2 model
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

    :param model: A SysML v2 Model
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

    :param model: The SysML v2 Model
    :param feature_sequences: Sequences that represent the nesting structure of the features
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
            # print(f"Skipped sequence ending in {last_item}")
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


def random_generator_playbook_phase_3_new_instances(
    model: Model,
    feature_sequences: list,
    instances_dict: dict,
) -> list:
    """
    Begin generating interpreting sequences for Features in the model by extending
    classifier sequences with newly generated instances of classifiers that type
    nested features.

    :param model: A pointer to the active model for which sequences are being generated
    :param feature_sequences: Sequences that represent the nesting structure of the features
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: (Temporarily return a trace of actions) None - side effect is addition of new
        instances to the instances dictionary
    """

    # FIXME: This approach does not currently support moving from abstract classifiers to
    #        non-abstract classifiers but it is hard to know how to do this correctly with
    #        the SysML v2 libraries and implicit use of core definition types like Part or Port

    logger.debug("Starting things up")
    for feature_sequence in feature_sequences:
        # skip if the feature is abstract or its owning type is
        last_item = model.elements[feature_sequence[-1]]
        if last_item.isAbstract or last_item.owner.isAbstract:
            # print(f"Skipped sequnce ending in {last_item}")
            continue

        new_sequences = []
        for index, feature_id in enumerate(feature_sequence):
            # if you've seen this feature before, move deeper into the sequence
            if feature_id in instances_dict:
                new_sequences = instances_dict[feature_id]
                continue

            # sample set will be the last element in the sequence for classifiers
            feature = model.elements[feature_id]
            metatype = feature._metatype
            if metatype in TYPES_FOR_FEATURING:
                types = feature.type
                if isinstance(types, Element):
                    typ_id = types._id or []
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
                    typ_id = types[0]
            else:
                typ_id = feature_id

            lower_mult = feature_multiplicity(feature, "lower")
            upper_mult = min(feature_multiplicity(feature, "upper"), model.MAX_MULTIPLICITY)

            new_sequences = extend_sequences_with_new_instance(
                new_sequences,
                lower_mult,
                upper_mult,
                model.elements[typ_id],
                index == 0,  # set first step true on first time in the loop
            )
            # only do interpretation for usages, not types (they will get sequences from rollup)
            if metatype in TYPES_FOR_FEATURING or index == 0:
                instances_dict[feature_id] = new_sequences


def random_generator_playbook_phase_3_rollup(
    model: Model,
    scg: nx.DiGraph,
    instances_dict: dict,
) -> None:
    """
    Build up set of sequences for classifiers by taking the union of sequences
    already generated for the classifier subclasses.

    :param model: Active SysML model
    :param scg: Generalization graph projection from the LPG
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    roots = [node for node in scg.nodes if scg.in_degree(node) == 0]

    # FIXME: Generalization graph has multiple paths to elements (can be redefine and type
    #        for example) which may lead to some elements being skipped for rollup

    for root in roots:
        bfs_dict = dict(nx.bfs_successors(scg, root))
        bfs_list = list(bfs_dict.keys())
        bfs_list.reverse()

        for gen in bfs_list:
            new_superset = []
            # use the BFS dictionary to be assured that everything is covered
            # update_dict = generate_superset_instances(scg, gen, visited_nodes, instances_dict)

            for subset_node in bfs_dict[gen]:
                if subset_node in instances_dict:
                    try:
                        new_superset.extend(instances_dict[subset_node])
                    except KeyError as exc:
                        raise KeyError(
                            f"Cannot find {model.elements[subset_node]} in instances_dict!"
                        ) from exc

            instances_dict[gen] = new_superset


def random_generator_playbook_phase_4(
    model: Model,
    expr_sequences: List[List[str]],
    instances_dict: InstanceDictType,
):
    """
    Generate interpreting sequences for Expressions in the model

    :param model: The SysML v2 Model
    :param expr_sequences: Sequences that represent the membership structure for expressions
        in the model and the features to which expressions provide values
    :param instances_dict: Working dictionary of interpreted sequences for the model
    :return: None - side effect is addition of new instances to the instances dictionary
    """
    all_elements = model.elements
    for expr_seq in expr_sequences:
        new_sequences = []
        # get the featuring type of the first expression
        # seq_featuring_type = safe_feature_data(all_elements[expr_seq[0]], "featuringType")
        seq_featuring_type = model.elements[expr_seq[0]].featuringType
        # FIXME: I don't know what it means for binding connectors to own these expressions,
        #        but need to figure out eventually
        if isinstance(seq_featuring_type, list):
            continue
        if seq_featuring_type._metatype == "BindingConnector":
            continue
        try:
            new_sequences = instances_dict[seq_featuring_type._id]
        except KeyError as exc:
            raise KeyError(f"Could not find {seq_featuring_type}") from exc

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
    elements = lpg.model.elements
    # Generate sequences for connection and interface ends
    for connector_id in cug.nodes:
        connector = elements[connector_id]
        if connector._metatype not in TYPES_FOR_CONNECTOR_INSTANCES:
            continue

        connector_ends = connector.connectorEnd
        if len(connector_ends) != 2:
            raise NotImplementedError(
                f"Cannot process connector {connector} because it"
                f" it has {len(connector_ends) or 'no'} connector end(s)"
            )

        source, *other_sources = connector.source
        target, *other_targets = connector.target
        if other_sources or other_targets:
            raise NotImplementedError(
                f"Cannot process connector {connector} because it"
                " has multiple sources and/or targets"
            )

        try:
            source_sequences = instances_dict[source.chainingFeature[-1]._id]
        except IndexError as exc:
            raise IndexError(f"{source} has no chainingFeatures!") from exc
        except KeyError as exc:
            raise KeyError(f"Cannot find chaining feature sequences for {source}!") from exc

        try:
            target_sequences = instances_dict[target.chainingFeature[-1]._id]
        except IndexError as exc:
            raise IndexError(f"{target} has no chainingFeatures!") from exc
        except KeyError as exc:
            raise KeyError(f"Cannot find chaining feature sequences for {target}!") from exc

        # Get sequence sizes for each side
        sizes = len(source_sequences), len(target_sequences)
        min_side, max_side = min(sizes), max(sizes)

        # if one side has no connections, there's nothing to connect
        if min_side < 1:
            for connector_end in connector_ends:
                instances_dict[connector_end._id] = []
            continue

        try:
            indeces = list(range(min_side)) + [
                randint(0, min_side - 1) for _ in range(max_side - min_side)
            ]
            if len(source_sequences) <= len(target_sequences):
                connector_ends_indeces = indeces, sample(range(max_side), max_side)
            else:
                connector_ends_indeces = sample(range(max_side), max_side), indeces
        except ValueError as exc:
            raise ValueError(
                "Sample larger than population or is negative. Sizes are:"
                f" {min_side} and {max_side}."
            ) from exc

        connector_ends_sequences = (source_sequences, target_sequences)
        connector_sequences = instances_dict[connector_id]
        for connector_end, connector_end_indeces, connector_end_sequences in zip(
            connector_ends, connector_ends_indeces, connector_ends_sequences
        ):
            instances_dict[connector_end._id] = [
                [
                    sequence
                    + connector_end_sequences[
                        connector_end_indeces[idx % len(connector_end_sequences)]
                    ][1:]
                ]
                for idx, sequence in enumerate(connector_sequences)
            ]


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


def build_banded_sequence_templates(
    lpg: SysML2LabeledPropertyGraph,
    package_filter: List[Element] = None,
) -> list:
    part_featuring_graph = lpg.get_projection("Banded")
    sorted_feature_groups = []
    for comp in nx.connected_components(part_featuring_graph.to_undirected()):
        connected_sub = nx.subgraph(part_featuring_graph, list(comp))
        leaves = [node for node in connected_sub.nodes if connected_sub.in_degree(node) == 0]
        roots = [node for node in connected_sub.nodes if connected_sub.out_degree(node) == 0]

        for leaf_id in leaves:
            # FIXME: Filter by package here - if the leaf is not in
            #        the target package(s) skip looking for paths
            leaf = lpg.model.elements[leaf_id]
            if "Literal" in leaf._metatype:
                continue
            if package_filter and lpg.model.elements[leaf_id].owning_package not in package_filter:
                continue
            for root_id in roots:
                try:
                    leaf_paths = nx.all_simple_paths(connected_sub, leaf_id, root_id)

                    for leaf_path in leaf_paths:
                        leaf_path.reverse()
                        first, *nested = leaf_path
                        filtered_path = [first] + [
                            item_id
                            for item_id in nested
                            if lpg.model.elements[item_id]._metatype in TYPES_FOR_FEATURING
                        ]

                        sorted_feature_groups.append(filtered_path)
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
