import copy
from typing import Any, Dict, List
from uuid import uuid4

from .model import Element, Model


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
    for specific_update in specific_fields.keys():
        new_element_data[specific_update] = specific_fields[specific_update]

    new_element_data.update({"@type": metaclass})
    new_element_data.update({"@id": new_id})

    return new_element_data


def build_from_classifier_pattern(
    owner: Element, name: str, model: Model, metatype: str, specific_fields: Dict[str, Any]
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
    if metatype == "Feature":
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


def build_from_binary_assoc_pattern(
    name: str,
    source_role_name: str,
    target_role_name: str,
    source_type: Element,
    target_type: Element,
    model: Model,
    metatype: str,
    owner: Element,
    specific_fields: Dict[str, Any],
):
    """
    Creates a series of new elements using an association-style pattern that assumes:
    - The association is binary (only one source, one target)
    - The association is owned by some non-connected element such as a Package or a Type
    - Association has two end features that each have a type
    """

    new_ele = build_from_classifier_pattern(
        owner=owner, name=name, model=model, metatype=metatype, specific_fields=specific_fields
    )

    build_from_feature_pattern(
        owner=new_ele,
        name=source_role_name,
        model=model,
        specific_fields={},
        feature_type=source_type,
        direction="",
        metatype="Feature",
        connector_end=True,
    )

    build_from_feature_pattern(
        owner=new_ele,
        name=target_role_name,
        model=model,
        specific_fields={},
        feature_type=target_type,
        direction="",
        metatype="Feature",
        connector_end=True,
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
        owner=owner, name=name, model=model, metatype=metatype, specific_fields=conn_fields
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

    model._add_relationship(new_om)

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
    )

    for clz in classes:
        subclass_added_data = {
            "specific": {"@id": clz._id},
            "general": {"@id": new_super._id},
            "subclassifier": {"@id": clz._id},
            "superclassifier": {"@id": new_super._id},
        }

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

    if len(one_member_classifiers) > 1:
        covering_type = build_unioning_superset_classifier(
            classes=one_member_classifiers,
            super_name=covering_classifier_prefix
            + feature_to_cover.declaredName
            + covering_classifier_suffix,
            owner=new_types_owner,
            model=model,
            added_fields={},
            unioned=True,
        )

        redefined_feature = build_from_feature_pattern(
            owner=type_to_apply_pattern_on,
            name=redefining_feature_prefix
            + feature_to_cover.declaredName
            + redefining_feature_suffix,
            model=model,
            specific_fields={},
            feature_type=covering_type,
            connector_end=False,
            metatype="Feature",
        )

    elif len(one_member_classifiers) == 1:
        redefined_feature = build_from_feature_pattern(
            owner=type_to_apply_pattern_on,
            name=redefining_feature_prefix
            + feature_to_cover.declaredName
            + redefining_feature_suffix,
            model=model,
            specific_fields={},
            feature_type=one_member_classifiers[0],
            connector_end=False,
            metatype="Feature",
        )

    return redefined_feature


def build_covering_classifier_for_connector():
    pass


def build_snapshot_for_classifier():
    pass
