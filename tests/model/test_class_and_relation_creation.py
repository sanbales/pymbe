from uuid import uuid4

import pymbe.api as pm
from pymbe.model import Element, Model
from pymbe.model_modification import (
    build_from_binary_relationship_pattern,
    build_from_classifier_pattern,
)


def test_create_classifier_against_library(load_kerml_library):

    """
    Try the creation of new classifiers that specialize library elements.

    """

    # find Performance in libraries

    peform_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Performances"
    ][0]

    perform_eles = peform_ns.throughOwningMembership[0].throughOwningMembership

    performance = None

    for perform_ele in perform_eles:
        if perform_ele._metatype in ("Behavior"):
            if hasattr(perform_ele, "declaredName"):
                if perform_ele.declaredName == "Performance":
                    performance = perform_ele

    empty_model = pm.Model(elements={})

    package_model_namespace_data = {
        "aliasIds": [],
        "isImpliedIncluded": False,
        "@type": "Namespace",
        "@id": str(uuid4()),
        "ownedRelationship": [],
    }

    package_model_data = {
        "name": "User Process Model",
        "isLibraryElement": False,
        "filterCondition": [],
        "ownedElement": [],
        "owner": {},
        "@type": "Package",
        "@id": str(uuid4()),
        "ownedRelationship": [],
    }

    new_ns = Element.new(data=package_model_namespace_data, model=empty_model)

    new_package = Element.new(data=package_model_data, model=empty_model)

    empty_model.reference_other_model(load_kerml_library)

    new_performance = build_from_classifier_pattern(
        owner=new_package,
        name="New Process",
        model=empty_model,
        metatype="Behavior",
        superclasses=[performance],
        specific_fields={"ownedRelationship": []},
    )

    # check that the ownership is correct

    assert new_performance.reverseOwningMembership[0] == new_package

    # check that the subclassification has been done correctly

    assert new_performance.throughSubclassification[0] == performance

    # check that the new name is applied correctly

    assert new_performance.declaredName == "New Process"


def test_create_disjoint_objects(load_kerml_library):

    """
    Try the creation of new classifiers that specialize library elements.

    """

    # find Object in libraries

    object_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Objects"
    ][0]

    object_eles = object_ns.throughOwningMembership[0].throughOwningMembership

    obj = None

    for object_ele in object_eles:
        if object_ele._metatype in ("Structure"):
            if hasattr(object_ele, "declaredName"):
                if object_ele.declaredName == "Object":
                    obj = object_ele

    empty_model = pm.Model(elements={})

    package_model_namespace_data = {
        "aliasIds": [],
        "isImpliedIncluded": False,
        "@type": "Namespace",
        "@id": str(uuid4()),
        "ownedRelationship": [],
    }

    package_model_data = {
        "name": "User Process Model",
        "isLibraryElement": False,
        "filterCondition": [],
        "ownedElement": [],
        "owner": {},
        "@type": "Package",
        "@id": str(uuid4()),
        "ownedRelationship": [],
    }

    new_ns = Element.new(data=package_model_namespace_data, model=empty_model)

    new_package = Element.new(data=package_model_data, model=empty_model)

    empty_model.reference_other_model(load_kerml_library)

    new_object_1 = build_from_classifier_pattern(
        owner=new_package,
        name="Structure Kind 1",
        model=empty_model,
        metatype="Structure",
        superclasses=[obj],
        specific_fields={"ownedRelationship": []},
    )

    new_object_2 = build_from_classifier_pattern(
        owner=new_package,
        name="Structure Kind 2",
        model=empty_model,
        metatype="Structure",
        superclasses=[obj],
        specific_fields={"ownedRelationship": []},
    )

    # add a relation to assure the two Objects are considered disjoint

    build_from_binary_relationship_pattern(
        source=new_object_1,
        target=new_object_2,
        model=empty_model,
        metatype="Disjoining",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={
            "typeDisjoined": {"@id": new_object_1._id},
            "disjoiningType": {"@id": new_object_2._id},
        },
    )

    build_from_binary_relationship_pattern(
        source=new_object_2,
        target=new_object_1,
        model=empty_model,
        metatype="Disjoining",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={
            "typeDisjoined": {"@id": new_object_2._id},
            "disjoiningType": {"@id": new_object_1._id},
        },
    )

    assert new_object_1.throughDisjoining[0] == new_object_2
    assert new_object_1.reverseDisjoining[0] == new_object_2
