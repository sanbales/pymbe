from uuid import uuid4

import pytest

import pymbe.api as pm
from pymbe.interpretation.working_maps import FeatureTypeWorkingMap
from pymbe.model import Element, Model
from pymbe.model_modification import (
    build_from_binary_relationship_pattern,
    build_from_classifier_pattern,
    build_from_feature_pattern,
)
from pymbe.query.metamodel_navigator import (
    get_effective_basic_name,
    get_effective_lower_multiplicity,
    get_effective_upper_multiplicity,
    get_most_specific_feature_type,
)


def test_create_map():

    """
    Test the creation of a basic working map.

    """

    empty_model = pm.Model(elements={})

    package_model_namespace_data = {
        "aliasIds": [],
        "isImpliedIncluded": False,
        "@type": "Namespace",
        "@id": str(uuid4()),
        "ownedRelationship": [],
    }

    package_model_data = {
        "name": "Map Trial Model",
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

    # create a new classifier with a couple of nested features

    # create some atoms to be the example data

    new_type_instance = build_from_classifier_pattern(
        owner=new_package,
        name="Type Instance Example",
        model=empty_model,
        metatype="Classifier",
        superclasses=[],
        specific_fields={"ownedRelationship": []},
    )

    new_feature_type_1 = build_from_classifier_pattern(
        owner=new_package,
        name="Feature 1 Type",
        model=empty_model,
        metatype="Classifier",
        superclasses=[],
        specific_fields={"ownedRelationship": []},
    )

    new_feature_type_2 = build_from_classifier_pattern(
        owner=new_package,
        name="Feature 2 Type",
        model=empty_model,
        metatype="Classifier",
        superclasses=[],
        specific_fields={"ownedRelationship": []},
    )

    new_feature_1 = build_from_feature_pattern(
        owner=new_type_instance,
        name="Feature Level 1",
        model=empty_model,
        specific_fields={},
        feature_type=new_feature_type_1,
        direction="",
        metatype="Feature",
        connector_end=False,
    )

    new_feature_2 = build_from_feature_pattern(
        owner=new_feature_1,
        name="Feature Level 1.1",
        model=empty_model,
        specific_fields={},
        feature_type=new_feature_type_2,
        direction="",
        metatype="Feature",
        connector_end=False,
    )

    assert (
        get_effective_basic_name(new_type_instance.throughFeatureMembership[0])
        == "Feature Level 1"
    )

    # Create and add some values to the Feature Type working map

    new_map = FeatureTypeWorkingMap(empty_model)

    new_map._add_type_instance_to_map(new_type_instance)
    new_map._add_feature_to_type_instance(new_type_instance, [new_feature_1])
    new_map._add_feature_to_type_instance(new_type_instance, [new_feature_1, new_feature_2])

    new_feature_value_1 = build_from_classifier_pattern(
        owner=new_package,
        name="Feature 1 Type Instance",
        model=empty_model,
        metatype="Classifier",
        superclasses=[],
        specific_fields={"ownedRelationship": []},
    )

    new_feature_value_2 = build_from_classifier_pattern(
        owner=new_package,
        name="Feature 2 Type Instance",
        model=empty_model,
        metatype="Classifier",
        superclasses=[],
        specific_fields={"ownedRelationship": []},
    )

    new_map._add_atom_value_to_feature(new_type_instance, [new_feature_1], new_feature_value_1)
    new_map._add_atom_value_to_feature(
        new_type_instance, [new_feature_1, new_feature_2], new_feature_value_2
    )

    assert new_map._get_atom_values_for_feature(
        new_type_instance, [new_feature_1, new_feature_2]
    ) == [new_feature_value_2]

    assert new_map._get_atom_values_for_feature(new_type_instance, [new_feature_1]) == [
        new_feature_value_1
    ]
