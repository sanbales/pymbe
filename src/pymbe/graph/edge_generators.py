from typing import List, Tuple
from uuid import uuid4
from warnings import warn

from ..model import Element
from .lpg import SysML2LabeledPropertyGraph

MultiEdge = Tuple[str, str, str, dict]


def make_nx_multi_edge(source, target, metatype, **data) -> MultiEdge:
    return (
        source,  # source
        target,  # target
        metatype,  # edge type
        {  # edge data
            "@id": f"_{uuid4()}",
            "@type": metatype,
            "implied": True,
            "label": metatype,
            "relatedElement": [
                {"@id": source},
                {"@id": target},
            ],
            "source": [{"@id": source}],
            "target": [{"@id": target}],
            **data,
        },
    )


def get_elements_from_lpg_edges(lpg: SysML2LabeledPropertyGraph) -> List[Element]:
    """Get the elements based on the edges of a SysML2 LPG"""
    elements = lpg.model.elements

    return {
        elements[edge_data["@id"]]
        for edge_data in lpg.edges.values()
        # TODO: Determine if this should work for implied edges too
        if edge_data["@id"] in elements
    }


def make_lpg_edges(*edges) -> List[MultiEdge]:
    """Make networkx multiedges compatible with the LPG"""
    return [make_nx_multi_edge(source, target, metatype) for source, target, metatype in edges]


def get_implied_parameter_feedforward(lpg: SysML2LabeledPropertyGraph) -> List[MultiEdge]:
    implied_parameter_feedforward_edges = (
        (edge.value.result._id, edge.get_owner()._id, "ImpliedParameterFeedforward")
        for edge in get_elements_from_lpg_edges(lpg)
        if edge._metatype == "FeatureValue" and edge.value.result is not None
    )
    return make_lpg_edges(*implied_parameter_feedforward_edges)


def get_implied_feature_typings(lpg: SysML2LabeledPropertyGraph) -> List[MultiEdge]:
    """
    Set up to fill in for cases where typing, definition are in attributes rather than
    with explicit FeatureTyping edges from the API
    :param lpg:
    :return:
    """
    # TODO: Remove this when the API and spec are fixed to have types as multiple

    def get_types(element: Element) -> List[Element]:
        types = getattr(element, "type", None) or []
        if isinstance(types, Element):
            types = [types]

        return types

    all_ft = [(src, tgt) for src, tgt, _ in lpg.edges_by_type["FeatureTyping"]]
    implied_typing_edges = (
        (element._id, type_._id, "ImpliedFeatureTyping")
        for element in lpg.model.elements.values()
        for type_ in get_types(element)
        if (element._id, type_._id) not in all_ft
    )
    return make_lpg_edges(*implied_typing_edges)


def get_implied_feedforward_edges(lpg: SysML2LabeledPropertyGraph) -> List[MultiEdge]:
    elements = lpg.model.elements

    return_parameter_memberships = {
        edge
        for edge in get_elements_from_lpg_edges(lpg)
        if edge._metatype == "ReturnParameterMembership"
    }

    eeg = lpg.get_projection("Expression Evaluation")

    implied_edges = []
    for membership in return_parameter_memberships:
        for result_feeder_id in eeg.predecessors(membership.memberElement._id):
            result_feeder = elements[result_feeder_id]
            rf_metatype = result_feeder._metatype

            # we only want Expressions that have at least one input parameter
            if "Expression" not in rf_metatype or rf_metatype == "FeatureReferenceExpression":
                if rf_metatype == "FeatureReferenceExpression":
                    implied_edges += [
                        (
                            result_feeder.referent,
                            result_feeder,
                            "ImpliedReferentFeed",
                        )
                    ]
                continue

            # Path Step Expressions need results fed into them, so add edges to order this
            if rf_metatype == "PathStepExpression":
                implied_edges += [
                    (argument.result, result_feeder, "ImpliedPathArgumentFeedforward")
                    # Skipping first element to avoid a cyclic dependency
                    for argument in result_feeder.argument[1:]
                ]

            expr_results, para_members = [], []
            # assume that the members of an expression that are themselves members are
            # referenced in the same order as parameters - results of an expression
            # should feed into the input parameter owned by its owner

            # NOTE: There is a special case for when there is a ResultExpressionMembership:
            # A ResultExpressionMembership is a FeatureMembership that indicates that the
            # ownedResultExpression provides the result values for the Function or Expression
            # that owns it. The owning Function or Expression must contain a BindingConnector
            # between the result parameter of the ownedResultExpression and the result
            # parameter of the Function or Expression.
            for relationship in result_feeder.ownedMembership:
                relationship_metatype = relationship._metatype
                edge_member = relationship.memberElement
                if "Parameter" in relationship_metatype:
                    if "ReturnParameter" not in relationship_metatype:
                        para_members.append(edge_member)
                elif "Result" in relationship_metatype:
                    rem_owned_element = relationship.ownedMemberElement
                    expr_results = [rem_owned_element.result]
                    para_members = [relationship.owningType.result]
                    break
                elif "Membership" in relationship_metatype:
                    result = relationship.memberElement.get("result")
                    if result:
                        expr_results.append(result)
            num_expr_results, num_para_members = len(expr_results), len(para_members)
            if num_expr_results != num_para_members:
                warn(
                    f"Found {num_expr_results} Expression Results"
                    f" and {num_para_members} Return Parameter Members!"
                )
            num_matches = min(num_expr_results, num_para_members)
            implied_edges += [
                (expr_result, para_member, "ImpliedParameterFeedforward")
                for expr_result, para_member in list(zip(expr_results, para_members))[:num_matches]
            ]

    implied_edges = [
        (source._id, target._id, metatype) for source, target, metatype in implied_edges
    ]
    return make_lpg_edges(*implied_edges)


IMPLIED_GENERATORS = dict(
    ImpliedFeedforwardEdges=get_implied_feedforward_edges,
    ImpliedParameterFeedforward=get_implied_parameter_feedforward,
    ImpliedFeatureTyping=get_implied_feature_typings,
)
