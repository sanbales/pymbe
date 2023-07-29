from pymbe.model_modification import *
from pymbe.query.metamodel_navigator import *


def test_feature_mult_check(basic_load_files):

    # Tests the feature multiplicity of features

    annex_a = basic_load_files["AnnexA"]

    finite_type = get_finite_multiplicity_types(annex_a)

    assert finite_type[0].declaredName == "rollsOn"

    assert get_lower_multiplicty(finite_type[0]) == 2
    assert get_upper_multiplicty(finite_type[0]) == 2

    # The example works from Annex A A.3.3 - One-to-One Connectors


def test_classifier_special(basic_load_files):

    # Tests the production of supersetting classifiers

    annex_a = basic_load_files["AnnexA"]

    annex_a_classifiers = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier"
    ]

    annex_a_assocs = [
        assoc for assoc in list(annex_a.elements.values()) if assoc._metatype == "Association"
    ]

    classifiers_with_features = [
        test_classifier
        for test_classifier in annex_a_classifiers
        if "throughFeatureMembership" in test_classifier._derived
    ]

    assert classifiers_with_features[0].declaredName == "Bicycle"
    assert len(classifiers_with_features) == 1

    base_namespace = classifiers_with_features[0].owningRelationship.owningRelatedElement

    # The example works from Annex A A.3.3 - One-to-One Connectors

    # Do instanitation step 1 - create new atom of the classifier to instantiate

    bicycle_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="Bicycle Atom",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    # specialize from original

    build_from_binary_relationship_pattern(
        source=bicycle_atom,
        target=classifiers_with_features[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    assert bicycle_atom.throughSubclassification[0] == classifiers_with_features[0]


def test_quasi_association_special(basic_load_files):

    # Tests the production of supersetting associations as classifier and feature

    annex_a = basic_load_files["AnnexA"]

    annex_a_assocs = [
        assoc
        for assoc in list(annex_a.elements.values())
        if assoc._metatype == "Association" and assoc.declaredName == "BikeWheelFixed"
    ]

    annex_a_wheel = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier" and classifier.declaredName == "Wheel"
    ]

    assert len(annex_a_assocs) == 1

    base_namespace = annex_a_assocs[0].owningRelationship.owningRelatedElement

    # Do part of instanitation step 5 - create new atom of the association to instantiate and link into covering of connector feature

    bicycle_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="BikeWheelFixed",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    # specialize from original

    build_from_binary_relationship_pattern(
        source=bicycle_atom,
        target=annex_a_assocs[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    # redefine a feature so they can be retyped (next test)

    wheel_end_f = build_from_feature_pattern(
        owner=bicycle_atom,
        name="wheel redef",
        model=annex_a,
        specific_fields={},
        feature_type=annex_a_wheel[0],
        direction="",
        metatype="Feature",
        connector_end=True,
    )


def test_association_pattern(basic_load_files):

    # Tests the production of supersetting associations

    annex_a = basic_load_files["AnnexA"]

    annex_a_assocs = [
        assoc
        for assoc in list(annex_a.elements.values())
        if assoc._metatype == "Association" and assoc.declaredName == "BikeWheelFixed"
    ]

    annex_a_wheel = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier" and classifier.declaredName == "Wheel"
    ]

    annex_a_fork = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier" and classifier.declaredName == "BikeFork"
    ]

    assert len(annex_a_assocs) == 1

    base_namespace = annex_a_assocs[0].owningRelationship.owningRelatedElement

    # Do part of instanitation step 5 - create new atom of the associations (although not with specialized types yet)

    fixed_wheel_1 = build_from_binary_assoc_pattern(
        owner=base_namespace,
        source_role_name="wheel redef",
        target_role_name="fixedTo redef",
        source_type=annex_a_wheel[0],
        target_type=annex_a_fork[0],
        model=annex_a,
        metatype="Association",
        specific_fields={},
        name="BikeWheelFixed Special",
    )

    # specialize from original

    build_from_binary_relationship_pattern(
        source=fixed_wheel_1,
        target=annex_a_assocs[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )


def test_connector_covering(basic_load_files):

    # Tests the production of supersetting associations and covering of connector

    annex_a = basic_load_files["AnnexA"]

    annex_a_classifiers = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier"
    ]

    annex_a_assocs = [
        assoc
        for assoc in list(annex_a.elements.values())
        if assoc._metatype == "Association" and assoc.declaredName == "BikeWheelFixed"
    ]

    annex_a_connects = [
        assoc
        for assoc in list(annex_a.elements.values())
        if assoc._metatype == "Connector" and assoc.declaredName == "fixWheel"
    ]

    annex_a_wheel = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier" and classifier.declaredName == "Wheel"
    ]

    annex_a_fork = [
        classifier
        for classifier in list(annex_a.elements.values())
        if classifier._metatype == "Classifier" and classifier.declaredName == "BikeFork"
    ]

    assert len(annex_a_assocs) == 1
    assert len(annex_a_connects) == 1

    classifiers_with_features = [
        test_classifier
        for test_classifier in annex_a_classifiers
        if "throughFeatureMembership" in test_classifier._derived
    ]

    base_namespace = annex_a_assocs[0].owningRelationship.owningRelatedElement

    # Specialize the bicycle

    bike_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="MyBike",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    # Specialize the 2 wheels and bike forks needed for the rest of the pattern

    wheel_1_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="MyWheel1",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    build_from_binary_relationship_pattern(
        source=wheel_1_atom,
        target=classifiers_with_features[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    wheel_2_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="MyWheel2",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    build_from_binary_relationship_pattern(
        source=wheel_2_atom,
        target=classifiers_with_features[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    fork_1_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="MyBikeFork1",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    build_from_binary_relationship_pattern(
        source=fork_1_atom,
        target=classifiers_with_features[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    fork_2_atom = build_from_classifier_pattern(
        owner=base_namespace,
        name="MyBikeFork2",
        model=annex_a,
        metatype="Classifier",
        specific_fields={},
    )

    build_from_binary_relationship_pattern(
        source=fork_2_atom,
        target=classifiers_with_features[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    # Do part of instanitation step 5 - create new atom of the associations (although not with specialized types yet)

    bike_wheel_fixed_1 = build_from_binary_assoc_pattern(
        owner=base_namespace,
        source_role_name="wheel redef",
        target_role_name="fixedTo redef",
        source_type=wheel_1_atom,
        target_type=fork_1_atom,
        model=annex_a,
        metatype="Association",
        specific_fields={},
        name="BikeWheelFixed1",
    )

    build_from_binary_relationship_pattern(
        source=bike_wheel_fixed_1,
        target=annex_a_assocs[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    bike_wheel_fixed_2 = build_from_binary_assoc_pattern(
        owner=base_namespace,
        source_role_name="wheel redef",
        target_role_name="fixedTo redef",
        source_type=wheel_2_atom,
        target_type=fork_2_atom,
        model=annex_a,
        metatype="Association",
        specific_fields={},
        name="BikeWheelFixed2",
    )

    build_from_binary_relationship_pattern(
        source=bike_wheel_fixed_2,
        target=annex_a_assocs[0],
        model=annex_a,
        metatype="Subclassification",
        owned_by_source=True,
        owns_target=False,
        alternative_owner=None,
        specific_fields={},
    )

    # with the two associations in hand, create a covering for the connector

    apply_covered_feature_pattern(
        one_member_classifiers=[bike_wheel_fixed_1, bike_wheel_fixed_2],
        feature_to_cover=annex_a_connects[0],
        type_to_apply_pattern_on=bike_atom,
        model=annex_a,
        new_types_owner=base_namespace,
        covering_classifier_prefix="",
        covering_classifier_suffix=" total",
        redefining_feature_prefix="",
        redefining_feature_suffix=" (covered)",
    )
