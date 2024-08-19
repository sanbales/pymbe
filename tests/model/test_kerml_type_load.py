from pathlib import Path
from uuid import uuid4

import pytest

from pymbe.model import Element, Model


def test_namespaces_load(load_kerml_library):
    """
    Check that the expected namespaces from the KerML library are indeed loaded.
    """

    found_names = [
        library_model_ns.throughOwningMembership[0].declaredName
        for library_model_ns in load_kerml_library.ownedElement
    ]

    expected_names = [
        "Transfers",
        "CollectionFunctions",
        "BaseFunctions",
        "IntegerFunctions",
        "SequenceFunctions",
        "StringFunctions",
        "VectorValues",
        "TrigFunctions",
        "ScalarFunctions",
        "VectorFunctions",
        "ControlFunctions",
        "Base",
        "Observation",
        "RationalFunctions",
        "Links",
        "Performances",
        "NumericalFunctions",
        "DataFunctions",
        "ControlPerformances",
        "ComplexFunctions",
        "Metaobjects",
        "ScalarValues",
        "StatePerformances",
        "TransitionPerformances",
        "RealFunctions",
        "NaturalFunctions",
        "Objects",
        "Triggers",
        "FeatureReferencingPerformances",
        "Collections",
        "BooleanFunctions",
        "Occurrences",
        "SpatialFrames",
        "Clocks",
        "OccurrenceFunctions",
        "KerML",
    ]

    # check on equality regardless of order
    for fn in found_names:
        assert fn in expected_names

    for en in expected_names:
        assert en in found_names

    return True


def test_base_classifiers(load_kerml_library):

    """
    Check that the base classifiers have loaded and the expected names are all there.
    """

    base_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Base"
    ][0]

    # there are only DataValue and Anything in the base library

    base_root_eles = base_ns.throughOwningMembership[0].throughOwningMembership

    base_classifiers = []

    for base_ele in base_root_eles:
        if base_ele._metatype in ("Classifier", "DataType"):
            assert hasattr(base_ele, "declaredName")
            base_classifiers.append(base_ele)

    assert len(base_classifiers) == 2

    base_classifier_names = [bc.declaredName for bc in base_classifiers]

    assert "DataValue" in base_classifier_names
    assert "Anything" in base_classifier_names


def test_base_features(load_kerml_library):

    """
    Check that the base features have loaded and the expected names are all there.
    """

    base_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Base"
    ][0]

    base_root_eles = base_ns.throughOwningMembership[0].throughOwningMembership

    base_features = []

    for base_ele in base_root_eles:
        if base_ele._metatype in ("Feature"):
            assert hasattr(base_ele, "declaredName")
            base_features.append(base_ele)

    assert len(base_features) == 3

    base_feature_names = [bc.declaredName for bc in base_features]

    assert "dataValues" in base_feature_names
    assert "things" in base_feature_names
    assert "naturals" in base_feature_names


def test_base_multiplicity(load_kerml_library):

    """
    Check that the multiplicity ranges with names in Base library are structured as expected
    """

    base_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Base"
    ][0]

    base_root_eles = base_ns.throughOwningMembership[0].throughOwningMembership

    zero_or_one = None

    for base_ele in base_root_eles:
        if base_ele._metatype in ("MultiplicityRange"):
            if hasattr(base_ele, "declaredName"):
                if base_ele.declaredName == "zeroOrOne":
                    zero_or_one = base_ele

    # we expect that the multiplicity range has two owned literal integers for the 0 and 1

    assert zero_or_one is not None

    zero_or_one_literals = [
        owned
        for owned in zero_or_one.throughOwningMembership
        if owned._metatype == "LiteralInteger"
    ]

    zero_or_one_values = [literal.value for literal in zero_or_one_literals]

    assert 0 in zero_or_one_values
    assert 1 in zero_or_one_values

    assert len(zero_or_one_literals[0].throughReturnParameterMembership) == 1

    # for the literal integers, look at the membership for a name, parameter feature for a direction

    assert hasattr(zero_or_one_literals[0].ownedRelationship[0], "memberName")

    assert zero_or_one_literals[0].ownedRelationship[0].memberName == "result"

    assert hasattr(zero_or_one_literals[0].throughReturnParameterMembership[0], "direction")

    assert zero_or_one_literals[0].throughReturnParameterMembership[0].direction == "out"
