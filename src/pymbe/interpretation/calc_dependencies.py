import networkx as nx

from ..graph import SysML2LabeledPropertyGraph
from ..label import get_label_for_id
from ..model import Model


def generate_execution_order(lpg: SysML2LabeledPropertyGraph) -> list:
    """
    Generate an ordered list that relies the structure of computations to be
    applied to the M0 instance of the model when it is instantiated
    :return:
    """

    all_elements = lpg.model.elements
    eig = lpg.get_projection("Expression Inferred")

    execution_pairs = []
    execution_contexts = {}

    # use the BFS rollup method from the playbook phase 2

    roots = [node for node in eig.nodes if eig.in_degree(node) == 0]

    for root in roots:
        try:
            no_types = len(all_elements[root].featuringType)

            if no_types == 0:
                continue
            context = all_elements[root].featuringType[0].get("@id")
            execution_contexts[context] = []

            calc_order = list(nx.edge_bfs(eig, root))
            calc_order.reverse()
            # print(calc_order)

            for edg in calc_order:
                node_child = edg[1]
                node = edg[0]
                kind = ""

                if (
                    all_elements[node_child].get("@type") == "Feature"
                    and all_elements[node].get("@type") == "Feature"
                ):
                    kind = "Assignment"
                elif (
                    all_elements[node_child].get("@type") == "AttributeUsage"
                    and all_elements[node].get("@type") == "AttributeUsage"
                ):
                    relevant_edge_types = [
                        edg[2] for edg in eig.edges if edg[0] == node and edg[1] == node_child
                    ]
                    if "Redefinition^-1" in relevant_edge_types:
                        kind = "Redefinition"
                    else:
                        kind = "Assignment"
                elif all_elements[node].get("@type") == "FeatureReferenceExpression":
                    kind = "SelectionQuery"
                elif (
                    all_elements[node_child].get("@type") == "Feature"
                    and all_elements[node].get("@type") == "AttributeUsage"
                ):
                    kind = "ValueBinding"
                elif (node_child, node, "ReturnParameterMembership") in lpg.edges_by_type[
                    "ReturnParameterMembership"
                ]:
                    kind = "Output"
                elif (node, node_child, "ParameterMembership") in lpg.edges_by_type.get(
                    "ParameterMembership", []
                ):
                    kind = "Input"

                execution_pairs.append([node_child, node, kind])
                # bfs_check.append([node_child, node])

                execution_contexts[context].append(node_child)

        except AttributeError:
            continue

    return execution_pairs


def generate_parameter_signature_map(model: Model, execution_order: list):
    # use the execution order to find better parameter names
    naming_map = {}

    for pair in execution_order:
        if pair[2] == "Assignment":
            try:
                naming_map.update({pair[1]: naming_map[pair[0]]})
            except KeyError:
                # raise KeyError(
                #   f"{pair[0]} <<{model.elements[pair[1]]._metatype}>> not found in naming map!"
                # )
                continue
        elif pair[2] == "Output":
            left_side = ""
            if pair[0] in naming_map:
                left_side = naming_map[pair[0]]
            else:
                left_side = get_label_for_id(pair[0], model)
            # push the expression signature up to the result side
            if " => " in left_side:
                naming_map.update({pair[1]: left_side.split(" =>")[0]})
            else:
                naming_map.update({pair[1]: left_side})
        elif pair[2] == "Input":
            left_side = ""
            if pair[0] in naming_map:
                left_side = naming_map[pair[0]]
            else:
                left_side = get_label_for_id(pair[0], model)
            right_side = ""
            if pair[1] in naming_map:
                right_side = naming_map[pair[1]]
            else:
                right_side = get_label_for_id(pair[1], model)
            if pair[0] in naming_map:
                new_expr = right_side.replace(
                    get_label_for_id(pair[0], model), naming_map[pair[0]]
                )
                naming_map.update({pair[1]: new_expr})
            else:
                naming_map.update({pair[1]: get_label_for_id(pair[1], model)})

    return naming_map
