import copy
from typing import Any, Dict, List
from uuid import uuid4

from pymbe.model import Element, Model
from pymbe.query.metamodel_navigator import (
    get_effective_basic_name,
    get_most_specific_feature_type,
)


def create_element_data_dictionary(
    name: str, metaclass: str, model: Model, specific_fields: Dict[str, Any]
):
    """
    Creates a Python dictionary with data for a new KerML/SysML element based on templates from
    base Ecore definitions.
    """

    new_id = str(uuid4())

    new_element_data = copy.deepcopy(model.metamodel.pre_made_dicts[metaclass])

    new_element_data["declaredName"] = name
    new_element_data["name"] = name
    for specific_update in specific_fields.keys():
        new_element_data[specific_update] = specific_fields[specific_update]

    new_element_data.update({"@type": metaclass})
    new_element_data.update({"@id": new_id})

    return new_element_data


def build_from_classifier_pattern(
    owner: Element,
    name: str,
    model: Model,
    metatype: str,
    superclasses: List[Element],
    specific_fields: Dict[str, Any],
):
    """
    Creates a new element using a classifier-style pattern that assumes:
    - The new element will need an owner
    - New element has a name of interest
    - There are no end features to consider
    """

    classifier_dict = create_element_data_dictionary(
        name=name, metaclass=metatype, model=model, specific_fields=specific_fields
    )

    new_ele = Element.new(data=classifier_dict, model=model)

    new_element_ownership_pattern(
        owner=owner, ele=new_ele, model=model, member_kind="OwningMembership"
    )

    for supr in superclasses:
        if supr is not None:
            build_from_binary_relationship_pattern(
                source=new_ele,
                target=supr,
                model=model,
                metatype="Subclassification",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields=specific_fields,
            )

    return new_ele


def build_from_binary_relationship_pattern(
    source: Element,
    target: Element,
    model: Model,
    metatype: str,
    owned_by_source: bool,
    owns_target: bool,
    alternative_owner: Element,
    specific_fields: Dict[str, Any],
):
    """
    Creates a new element using a graph relationship-style pattern that assumes:
    - The new element may be owned by its source or some other element
    - There are source and target elements, each of multiplicity 1
    """

    owned_related_element = []
    owning_related_element = []
    owner = None
    if owns_target:
        owned_related_element = [{"@id": target._id}]
    if owned_by_source:
        owning_related_element = {"@id": source._id}
    if alternative_owner is not None:
        # TODO: Have reasonable exception if this and owned_by_source are both true
        owner = {"@id": alternative_owner._id}

    rel_specific = {
        "source": [{"@id": source._id}],
        "ownedRelatedElement": owned_related_element,
        "target": [{"@id": target._id}],
        "owningRelatedElement": owning_related_element,
        "owner": owner,
    } | specific_fields

    relationship_dict = create_element_data_dictionary(
        name="", metaclass=metatype, model=model, specific_fields=rel_specific
    )

    new_rel = Element.new(data=relationship_dict, model=model)

    model._add_relationship(new_rel)

    return new_rel


def build_from_feature_pattern(
    owner: Element,
    name: str,
    model: Model,
    specific_fields: Dict[str, Any],
    feature_type: Element,
    direction: str = "",
    metatype: str = "Feature",
    connector_end: bool = False,
):
    """
    Creates a new element using a feature-style pattern that assumes:
    - The Feature will have some special kind of membership connecting it to the owner
    - The Feature may have a multiplicity
    - The Feature may have a type
    """

    typing_snippet = {}
    direction_snippet = {}
    member_kind = ""

    if feature_type is not None:
        typing_snippet = {"type": {"@id": feature_type}}

    if direction != "":
        direction_snippet = {"direction": direction}

    specific_fields = typing_snippet | direction_snippet

    feature_dict = create_element_data_dictionary(
        name=name, metaclass=metatype, model=model, specific_fields=specific_fields
    )

    new_ele = Element.new(data=feature_dict, model=model)

    # TODO: Add more cases here
    if metatype in {"Feature", "Connector", "Succession", "Step"} or "Usage" in metatype:
        if connector_end:
            member_kind = "EndFeatureMembership"
        else:
            member_kind = "FeatureMembership"

    new_element_ownership_pattern(owner=owner, ele=new_ele, model=model, member_kind=member_kind)

    if feature_type is not None:
        build_from_binary_relationship_pattern(
            source=new_ele,
            target=feature_type,
            model=model,
            metatype="FeatureTyping",
            owned_by_source=True,
            owns_target=False,
            alternative_owner=None,
            specific_fields={},
        )

    return new_ele


def build_from_parameter_pattern(
    name: str,
    model: Model,
    specific_fields: Dict[str, Any],
    feature_type: Element,
    direction: str = "",
    metatype: str = "Feature",
    returning_parameter: bool = False,
):

    """
    Creates a new element using a feature-style pattern that assumes:
    - The Feature will have some special kind of membership connecting it to the owner
    - The Feature may have a multiplicity
    - The Feature may have a type
    """

    typing_snippet = {}
    direction_snippet = {}
    member_kind = ""

    if feature_type is not None:
        typing_snippet = {"type": {"@id": feature_type}}

    if direction != "":
        direction_snippet = {"direction": direction}

    specific_fields = typing_snippet | direction_snippet

    feature_dict = create_element_data_dictionary(
        name=name, metaclass=metatype, model=model, specific_fields=specific_fields
    )

    new_ele = Element.new(data=feature_dict, model=model)

    # TODO: Add more cases here
    if returning_parameter:
        member_kind = "ReturnParameterMembership"
    else:
        member_kind = "ParameterMembership"

    # new_element_ownership_pattern(owner=owner, ele=new_ele, model=model, member_kind=member_kind)

    if feature_type is not None:
        build_from_binary_relationship_pattern(
            source=new_ele,
            target=feature_type,
            model=model,
            metatype="FeatureTyping",
            owned_by_source=True,
            owns_target=False,
            alternative_owner=None,
            specific_fields={},
        )

    return new_ele


def build_from_binary_assoc_pattern(
    name: str,
    source_role_name: str,
    target_role_name: str,
    source_type: Element,
    target_type: Element,
    model: Model,
    metatype: str,
    owner: Element,
    superclasses: List[Element],
    specific_fields: Dict[str, Any],
):
    """
    Creates a series of new elements using an association-style pattern that assumes:
    - The association is binary (only one source, one target)
    - The association is owned by some non-connected element such as a Package or a Type
    - Association has two end features that each have a type
    """

    specific_fields = {
        "source": [{"@id": source_type._id}],
        "target": [{"@id": target_type._id}],
    } | specific_fields

    new_ele = build_from_classifier_pattern(
        owner=owner,
        name=name,
        model=model,
        metatype=metatype,
        specific_fields=specific_fields,
        superclasses=[],
    )

    source_feat_ele = build_from_feature_pattern(
        owner=new_ele,
        name=source_role_name,
        model=model,
        specific_fields={},
        feature_type=source_type,
        direction="",
        metatype="Feature",
        connector_end=True,
    )

    target_feat_ele = build_from_feature_pattern(
        owner=new_ele,
        name=target_role_name,
        model=model,
        specific_fields={},
        feature_type=target_type,
        direction="",
        metatype="Feature",
        connector_end=True,
    )

    for supr in superclasses:
        if supr is not None:

            # fix this later to become end feature memberships

            try:
                source_redefined = supr.throughEndFeatureMembership[0]
            except AttributeError:
                source_redefined = supr.throughFeatureMembership[0]
            except IndexError:
                source_redefined = supr.throughFeatureMembership[0]

            try:
                target_redefined = supr.throughEndFeatureMembership[1]
            except AttributeError:
                target_redefined = supr.throughFeatureMembership[1]
            except IndexError:
                target_redefined = supr.throughFeatureMembership[1]

            build_from_binary_relationship_pattern(
                source=new_ele,
                target=supr,
                model=model,
                metatype="Subclassification",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={},
            )

            build_from_binary_relationship_pattern(
                source=source_feat_ele,
                target=source_redefined,
                model=model,
                metatype="Redefinition",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={},
            )

            build_from_binary_relationship_pattern(
                source=target_feat_ele,
                target=target_redefined,
                model=model,
                metatype="Redefinition",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={},
            )

    return new_ele


def build_from_binary_connector_pattern(
    name: str,
    source_role_name: str,
    target_role_name: str,
    source: Element,
    target: Element,
    model: Model,
    metatype: str,
    owner: Element,
    specific_fields: Dict[str, Any],
):
    """
    Creates a series of new elements using a connector-style pattern that assumes:
    - The connector is binary (only one source, one target)
    - The connector is owned by some non-connected element such as a Package or a Type
    - Connector has two end features that point to the source and the target
    """

    connector_dict = {"source": [{"@id": source._id}], "target": [{"@id": target._id}]}

    conn_fields = specific_fields | connector_dict

    new_ele = build_from_classifier_pattern(
        owner=owner,
        name=name,
        model=model,
        metatype=metatype,
        specific_fields=conn_fields,
        superclasses=[],
    )

    source_end = build_from_feature_pattern(
        owner=new_ele,
        name=source_role_name,
        model=model,
        specific_fields={},
        feature_type=None,
        direction="",
        metatype="Feature",
        connector_end=True,
    )

    target_end = build_from_feature_pattern(
        owner=new_ele,
        name=target_role_name,
        model=model,
        specific_fields={},
        feature_type=None,
        direction="",
        metatype="Feature",
        connector_end=True,
    )

    build_from_binary_relationship_pattern(
        source=source_end,
        target=source,
        model=model,
        metatype="ReferenceSubsetting",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    build_from_binary_relationship_pattern(
        source=target_end,
        target=target,
        model=model,
        metatype="ReferenceSubsetting",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    return new_ele


def new_element_ownership_pattern(
    owner: Element, ele: Element, model: Model, member_kind: str = "OwningMembership"
):
    """
    Common helper to link new elements to their owners.
    """

    member_name = ""
    if "declaredName" in ele._data:
        member_name = ele.declaredName

    om_added_data = {
        "memberName": member_name,
        "memberElement": {"@id": ele._id},
    }

    new_om = build_from_binary_relationship_pattern(
        source=owner,
        target=ele,
        model=model,
        metatype=member_kind,
        owned_by_source=True,
        owns_target=True,
        alternative_owner=None,
        specific_fields=om_added_data,
    )

    # should make this more automatic in core code, but add new_om to owner's ownedRelationship
    # a lot of these entailments will be a pain and need to manage them actively

    owner._data["ownedRelationship"].append({"@id": new_om._id})
    ele._data["owningRelationship"] = {"@id": new_om._id}

    return new_om


# TODO: Move to dedicated reasoning file under interpretation area
def build_unioning_superset_classifier(
    classes: List[Element],
    super_name: str,
    owner: Element,
    model: Model,
    added_fields: Dict[str, Any],
    unioned: bool = False,
):
    """
    Take in a list of classifiers and generate a larger set from them. The larger
    set will have some properties:
    - All classes will have Subclassification relationship to the larger class
    - The larger class will be derived as a union of the given list of classes
    """
    new_super = build_from_classifier_pattern(
        owner=owner,
        name=super_name,
        model=model,
        specific_fields=added_fields,
        metatype=classes[0]._metatype,
        superclasses=[],
    )

    for clz in classes:
        subclass_added_data = {
            "specific": {"@id": clz._id},
            "general": {"@id": new_super._id},
            "subclassifier": {"@id": clz._id},
            "superclassifier": {"@id": new_super._id},
        }

        if unioned:
            build_from_binary_relationship_pattern(
                source=new_super,
                target=clz,
                model=model,
                metatype="Unioning",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={"unioningType": {"@id": clz._id}},
            )

        else:
            build_from_binary_relationship_pattern(
                source=clz,
                target=new_super,
                model=model,
                metatype="Subclassification",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields=subclass_added_data,
            )

    return new_super


def apply_covered_feature_pattern(
    one_member_classifiers: List[Element],
    feature_to_cover: Element,
    type_to_apply_pattern_on: Element,
    model: Model,
    new_types_owner: Element,
    covering_classifier_prefix: str = "Class to Cover ",
    covering_classifier_suffix: str = "",
    redefining_feature_prefix: str = "",
    redefining_feature_suffix: str = " (Closed)",
):
    """
    Execute a pattern described in KerML Appendix A to capture a list of specific results
    for a given generated model instance (or trace):
    - A series of classifiers with multiplicity 1 (given by user)
    - A superset classifier to represent all of these classifiers at once
    - Redefining the covered feature with a feature that uses the superset as the type,
        multiplicity set to number of identified specific classifiers
    """

    covering_type = None
    redefined_feature = None

    feature_to_cover_name = get_effective_basic_name(feature_to_cover)

    if len(one_member_classifiers) > 1:
        covering_type = build_unioning_superset_classifier(
            classes=one_member_classifiers,
            super_name=covering_classifier_prefix
            + feature_to_cover_name
            + covering_classifier_suffix,
            owner=new_types_owner,
            model=model,
            added_fields={},
            unioned=True,
        )

        redefined_feature = build_from_feature_pattern(
            owner=type_to_apply_pattern_on,
            name=redefining_feature_prefix + feature_to_cover_name + redefining_feature_suffix,
            model=model,
            specific_fields={},
            feature_type=covering_type,
            connector_end=False,
            metatype="Feature",
        )

    elif len(one_member_classifiers) == 1:
        redefined_feature = build_from_feature_pattern(
            owner=type_to_apply_pattern_on,
            name=redefining_feature_prefix + feature_to_cover_name + redefining_feature_suffix,
            model=model,
            specific_fields={},
            feature_type=one_member_classifiers[0],
            connector_end=False,
            metatype="Feature",
        )

    return redefined_feature


def apply_covered_connector_pattern(
    one_member_classifiers: List[Element],
    feature_to_cover: Element,
    type_to_apply_pattern_on: Element,
    model: Model,
    new_types_owner: Element,
    covering_classifier_prefix: str = "Class to Cover ",
    covering_classifier_suffix: str = "",
    redefining_feature_prefix: str = "",
    redefining_feature_suffix: str = " (Closed)",
    metatype: str = "Connector",
    separate_connectors: bool = False,
):
    """
    Execute a pattern described in KerML Appendix A to capture a list of specific results
    for a given generated model instance (or trace):
    - A series of classifiers with multiplicity 1 (given by user)
    - A superset classifier to represent all of these classifiers at once
    - Redefining the covered feature with a feature that uses the superset as the type,
        multiplicity set to number of identified specific classifiers
    """

    covering_type = None
    redefined_feature = None

    connector_covering_name = get_effective_basic_name(feature_to_cover)

    if separate_connectors:
        if connector_covering_name == "":
            connector_covering_name = (
                f"Connector between "
                + f"{get_effective_basic_name(feature_to_cover.source[0])} and "
                + f"{get_effective_basic_name(feature_to_cover.target[0])}"
            )

        for i, omc in enumerate(one_member_classifiers):
            build_from_feature_pattern(
                owner=type_to_apply_pattern_on,
                name=redefining_feature_prefix
                + connector_covering_name
                + redefining_feature_suffix
                + " "
                + str(i),
                model=model,
                specific_fields={},
                feature_type=omc,
                connector_end=False,
                metatype=metatype,
            )

    else:

        if len(one_member_classifiers) > 1:
            covering_type = build_unioning_superset_classifier(
                classes=one_member_classifiers,
                super_name=covering_classifier_prefix
                + feature_to_cover.basic_name
                + covering_classifier_suffix,
                owner=new_types_owner,
                model=model,
                added_fields={},
                unioned=True,
            )

            redefined_feature = build_from_feature_pattern(
                owner=type_to_apply_pattern_on,
                name=redefining_feature_prefix
                + feature_to_cover.basic_name
                + redefining_feature_suffix,
                model=model,
                specific_fields={},
                feature_type=covering_type,
                connector_end=False,
                metatype=metatype,
            )

        elif len(one_member_classifiers) == 1:
            redefined_feature = build_from_feature_pattern(
                owner=type_to_apply_pattern_on,
                name=redefining_feature_prefix
                + feature_to_cover.basic_name
                + redefining_feature_suffix,
                model=model,
                specific_fields={},
                feature_type=one_member_classifiers[0],
                connector_end=False,
                metatype=metatype,
            )

    return redefined_feature


def build_from_portion_pattern(
    owner: Element,
    name_extension: str,
    model: Model,
    classifier_to_be_portioned: None,
    feature_to_be_set: List[Element],
    feature_values: List[Any],
    specific_fields: Dict[str, Any],
):

    """
    Execute a pattern to create a portion of a classifier

    """

    metatype = classifier_to_be_portioned._metatype

    classifier_dict = create_element_data_dictionary(
        name=classifier_to_be_portioned.declaredName + name_extension,
        metaclass=metatype,
        model=model,
        specific_fields=specific_fields,
    )

    new_ele = Element.new(data=classifier_dict, model=model)

    new_element_ownership_pattern(
        owner=owner, ele=new_ele, model=model, member_kind="OwningMembership"
    )

    if hasattr(classifier_to_be_portioned, "throughFeatureMembership"):
        if len(feature_to_be_set) == 0:
            for feat in classifier_to_be_portioned.throughFeatureMembership:

                redefed_feature = build_from_feature_pattern(
                    owner=new_ele,
                    name=feat.declaredName,
                    model=model,
                    specific_fields={},
                    feature_type=feat.throughFeatureTyping[0]
                    if hasattr(feat, "throughFeatureTyping")
                    else None,
                    direction="",
                    metatype=feat._metatype,
                    connector_end=False,
                )

                build_from_binary_relationship_pattern(
                    source=redefed_feature,
                    target=feat,
                    model=model,
                    metatype="Redefinition",
                    owned_by_source=True,
                    owns_target=False,
                    alternative_owner=None,
                    specific_fields={},
                )
        for feat_set in feature_to_be_set:
            redefed_feature = build_from_feature_pattern(
                owner=new_ele,
                name=feat_set.declaredName,
                model=model,
                specific_fields={},
                feature_type=feat_set.throughFeatureTyping[0]
                if hasattr(feat_set, "throughFeatureTyping")
                else None,
                direction="",
                metatype=feat_set._metatype,
                connector_end=False,
            )

            build_from_binary_relationship_pattern(
                source=redefed_feature,
                target=feat_set,
                model=model,
                metatype="Redefinition",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={},
            )

    return new_ele


def apply_chained_feature_assignment_pattern(
    feature_path_to_chain: List[Element],
    type_to_apply_pattern_on: Element,
    model: Model,
    chained_feature_prefix: str = "",
    chained_feature_suffix: str = " (Closed)",
):
    """
    Execute a pattern that will create a feature that is linked to a
    feature chain that can been used to set values on deeply nested features
    """

    new_feature_for_chain = None

    feature_to_chain_name = get_effective_basic_name(feature_path_to_chain[-1])

    new_feature_for_chain = build_from_feature_pattern(
        owner=type_to_apply_pattern_on,
        name=feature_to_chain_name,
        model=model,
        specific_fields={},
        feature_type=get_most_specific_feature_type(feature_path_to_chain[-1]),
        direction="",
        metatype=feature_path_to_chain[-1]._metatype,
        connector_end=False,
    )

    # create the parameters

    new_in_para_1 = build_from_parameter_pattern(
        name="source",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="in",
        metatype="Feature",
        returning_parameter=False,
    )

    new_result_para_1 = build_from_parameter_pattern(
        name="result",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="out",
        metatype="Feature",
        returning_parameter=True,
    )

    # create the feature chain expression

    new_fce = build_from_expression_pattern(
        owner=new_feature_for_chain,
        model=model,
        specific_fields={},
        metatype="FeatureChainExpression",
        in_paras=[new_in_para_1],
        return_para=new_result_para_1,
    )

    # start the feature chain
    build_from_binary_relationship_pattern(
        source=new_feature_for_chain,
        target=new_fce,
        model=model,
        metatype="FeatureChaining",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    # FeatureReferenceExpression to the first item in the path

    new_fre_in_para_1 = build_from_parameter_pattern(
        name="source",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="in",
        metatype="Feature",
        returning_parameter=False,
    )

    new_fre_result_para_1 = build_from_parameter_pattern(
        name="result",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="out",
        metatype="Feature",
        returning_parameter=True,
    )

    # need reference to the first element in the path
    new_fre = build_from_feature_ref_expression_pattern(
        owner=new_fce,
        model=model,
        specific_fields={},
        metatype="FeatureReferenceExpression",
        in_paras=[new_fre_in_para_1],
        return_para=new_fre_result_para_1,
        referred_feature=feature_path_to_chain[0],
    )

    build_from_binary_relationship_pattern(
        source=new_in_para_1,
        target=new_fre,
        model=model,
        metatype="FeatureValue",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    # will need to do the same for the next ones

    if len(feature_path_to_chain) > 1:
        feature_dict = create_element_data_dictionary(
            name="placeholder", metaclass="Feature", model=model, specific_fields={}
        )

        new_fce_feat = Element.new(data=feature_dict, model=model)

        new_element_ownership_pattern(
            owner=new_fce, ele=new_fce_feat, model=model, member_kind="OwningMembership"
        )

        for path_ele in feature_path_to_chain[1:]:
            build_from_binary_relationship_pattern(
                source=new_fce_feat,
                target=path_ele,
                model=model,
                metatype="FeatureChaining",
                owned_by_source=True,
                owns_target=False,
                alternative_owner=None,
                specific_fields={},
            )
    # force resolution in name once everything is in place
    model._add_labels(new_fce)

    return new_feature_for_chain


def build_from_expression_pattern(
    owner: Element,
    model: Model,
    specific_fields: dict,
    metatype: str = "Expression",
    in_paras: List[Element] = [],
    return_para: Element = None,
):

    typing_snippet = {}
    direction_snippet = {}

    member_kind = "FeatureMembership"

    specific_fields = typing_snippet | direction_snippet | specific_fields

    feature_dict = create_element_data_dictionary(
        name="", metaclass=metatype, model=model, specific_fields=specific_fields
    )

    # hacking to keep label from crashing before the expression has parameters
    model._initializing = True

    new_ele = Element.new(data=feature_dict, model=model)

    model._add_element(new_ele)

    new_rpm = new_element_ownership_pattern(
        owner=new_ele, ele=return_para, model=model, member_kind="ReturnParameterMembership"
    )
    new_pms = []

    for in_para in in_paras:
        new_pm = new_element_ownership_pattern(
            owner=new_ele, ele=in_para, model=model, member_kind="ParameterMembership"
        )
        new_pms.append(new_pm)

    model._initializing = False

    new_ele.resolve()

    model._add_labels(new_ele)

    for new_pm in new_pms:
        new_pm.resolve()
        model._add_labels(new_pm)
    new_rpm.resolve()
    model._add_labels(new_rpm)

    # ownership of expression
    new_element_ownership_pattern(owner=owner, ele=new_ele, model=model, member_kind=member_kind)

    return new_ele


def build_from_feature_ref_expression_pattern(
    owner: Element,
    model: Model,
    specific_fields: dict,
    metatype: str = "FeatureReferenceExpression",
    in_paras: List[Element] = [],
    return_para: Element = None,
    referred_feature: Element = None,
):

    typing_snippet = {}
    direction_snippet = {}

    member_kind = "FeatureMembership"

    specific_fields = typing_snippet | direction_snippet | specific_fields

    feature_dict = create_element_data_dictionary(
        name="", metaclass=metatype, model=model, specific_fields=specific_fields
    )

    # hacking to keep label from crashing before the expression has parameters
    model._initializing = True

    new_ele = Element.new(data=feature_dict, model=model)

    model._add_element(new_ele)

    new_rpm = new_element_ownership_pattern(
        owner=new_ele, ele=return_para, model=model, member_kind="ReturnParameterMembership"
    )
    new_pms = []

    for in_para in in_paras:
        new_pm = new_element_ownership_pattern(
            owner=new_ele, ele=in_para, model=model, member_kind="ParameterMembership"
        )
        new_pms.append(new_pm)

    new_membership = build_from_binary_relationship_pattern(
        source=new_ele,
        target=referred_feature,
        model=model,
        metatype="Membership",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    model._initializing = False

    new_ele.resolve()

    model._add_labels(new_ele)

    for new_pm in new_pms:
        new_pm.resolve()
        model._add_labels(new_pm)
    new_rpm.resolve()
    model._add_labels(new_rpm)

    new_membership.resolve()
    model._add_labels(new_membership)

    # ownership of expression
    new_element_ownership_pattern(owner=owner, ele=new_ele, model=model, member_kind=member_kind)

    return new_ele


def assign_value_by_literal_expression(
    target_feature: Element, value_to_assign: Any, model: Model
):

    """
    Generate the binding connector, assign as a Feature Value and create a Literal Expression
    as a way to assign a value to a Feature.
    """

    # create the parameters

    new_in_para_1 = build_from_parameter_pattern(
        name="source",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="in",
        metatype="Feature",
        returning_parameter=False,
    )

    new_result_para_1 = build_from_parameter_pattern(
        name="result",
        model=model,
        specific_fields={},
        feature_type=None,
        direction="out",
        metatype="Feature",
        returning_parameter=True,
    )

    # create the literal expression

    le_meta = ""
    if isinstance(value_to_assign, bool):
        le_meta = "LiteralBoolean"
    elif isinstance(value_to_assign, int):
        le_meta = "LiteralInteger"
    elif isinstance(value_to_assign, float):
        le_meta = "LiteralReal"
    else:
        le_meta = "LiteralString"

    new_le = build_from_expression_pattern(
        owner=target_feature,
        model=model,
        specific_fields={"value": value_to_assign},
        metatype=le_meta,
        in_paras=[new_in_para_1],
        return_para=new_result_para_1,
    )

    new_valuing = build_from_binary_relationship_pattern(
        source=target_feature,
        target=new_le,
        model=model,
        metatype="FeatureValue",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    return new_le


def assign_value_by_fre():

    """
    Generate the binding connector, assign as a Feature Value and create a
    FeatureReferenceExpression as a way to assign a value to a Feature.
    """

    pass
