from pathlib import Path
from uuid import uuid4

import pytest


def test_classifier_load(basic_load_files):
    """
    Test that the file loaders can get KerML Classifiers and identify them by name and metatype
    """

    level1 = basic_load_files["Level1"]
    level2 = basic_load_files["Level2"]
    level3 = basic_load_files["Level3"]

    level1_classes = [
        classifier
        for classifier in list(level1.elements.values())
        if classifier._metatype == "Classifier"
    ]

    # Level 1 check
    # find My Bare Class
    # find My Class with Two features
    # find Class with Typed Features
    # find Specialized Class

    assert "My Bare Class" in [classifier.declaredName for classifier in level1_classes]

    # Level 2 check
    # find Context

    level2_classes = [
        classifier
        for classifier in list(level2.elements.values())
        if classifier._metatype == "Classifier"
    ]

    assert "Context" in [classifier.declaredName for classifier in level2_classes]

    # Level 3 check
    # find Adding Machine

    level3_classes = [
        classifier
        for classifier in list(level3.elements.values())
        if classifier._metatype == "Classifier"
    ]

    assert "Adding Machine" in [classifier.declaredName for classifier in level3_classes]


def test_feature_load(basic_load_files):
    """
    Test that the file loaders can get KerML Features and identify them by name and metatype
    """

    level1 = basic_load_files["Level1"]
    level2 = basic_load_files["Level2"]
    level3 = basic_load_files["Level3"]

    # Level 1 check

    level1_features = [
        feature for feature in list(level1.elements.values()) if feature._metatype == "Feature"
    ]

    # find Test Feature 1
    assert "Test Feature 1" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]
    # find Test Feature 2
    assert "Test Feature 2" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]
    # find Bare Feature
    assert "Bare Feature" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]
    # find Bare Classed Feature
    assert "Bare Classed Feature" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]
    # find Typed Feature 1
    assert "Typed Feature 1" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]
    # find Typed Feature 2
    assert "Typed Feature 2" in [
        feature.declaredName for feature in level1_features if hasattr(feature, "declaredName")
    ]

    # Level 2 check
    level2_features = [
        feature for feature in list(level2.elements.values()) if feature._metatype == "Feature"
    ]
    # find Side 1
    assert "Side 1" in [
        feature.declaredName for feature in level2_features if hasattr(feature, "declaredName")
    ]
    # find two 'Value' features
    assert "Value" in [
        feature.declaredName for feature in level2_features if hasattr(feature, "declaredName")
    ]
    assert (
        len(
            [
                feature
                for feature in level2_features
                if hasattr(feature, "declaredName") and feature.declaredName == "Value"
            ]
        )
        == 2
    )
    # find Side 2
    assert "Side 2" in [
        feature.declaredName for feature in level2_features if hasattr(feature, "declaredName")
    ]

    # Level 3 check
    level3_features = [
        feature for feature in list(level3.elements.values()) if feature._metatype == "Feature"
    ]
    # find Register 1
    assert "Register 1" in [
        feature.declaredName for feature in level3_features if hasattr(feature, "declaredName")
    ]
    # find Register 2
    assert "Register 2" in [
        feature.declaredName for feature in level3_features if hasattr(feature, "declaredName")
    ]
    # find Register 3
    assert "Register 3" in [
        feature.declaredName for feature in level3_features if hasattr(feature, "declaredName")
    ]


def test_memberships_load(basic_load_files):
    """
    Test that the file loaders can get KerML Memberships (owning and feature) and identify them by metatype and names of ends
    """

    level1 = basic_load_files["Level1"]

    # Level 1 check

    level1_membs = [
        memb
        for memb in list(level1.elements.values())
        if memb._metatype == "OwningMembership" or memb._metatype == "FeatureMembership"
    ]

    level1_membs_single = [memb for memb in level1_membs if len(memb.ownedRelatedElement) == 1]

    level1_memb_names = [
        (memb.owningRelatedElement.declaredName, memb.ownedRelatedElement[0].declaredName)
        for memb in level1_membs_single
        if hasattr(memb.ownedRelatedElement[0], "declaredName")
        and hasattr(memb.owningRelatedElement, "declaredName")
    ]

    assert ("Model Loader Test Level 1", "My Bare Class") in level1_memb_names
    assert ("Model Loader Test Level 1", "My Class with Two Features") in level1_memb_names
    assert ("Model Loader Test Level 1", "Bare Feature") in level1_memb_names
    assert ("Model Loader Test Level 1", "Bare Classed Feature") in level1_memb_names
    assert ("Model Loader Test Level 1", "Class with Typed Features") in level1_memb_names
    assert ("Model Loader Test Level 1", "Specialized Class") in level1_memb_names

    assert ("My Class with Two Features", "Test Feature 1") in level1_memb_names
    assert ("My Class with Two Features", "Test Feature 2") in level1_memb_names

    assert ("Class with Typed Features", "Typed Feature 1") in level1_memb_names
    assert ("Class with Typed Features", "Typed Feature 2") in level1_memb_names

    # Level 3 check on invariant


def test_invariant_load(basic_load_files):
    """
    Test proper loading of a simple expression tree
    """
    level3 = basic_load_files["Level3"]

    key_invar = [
        invariant
        for invariant in list(level3.elements.values())
        if invariant._metatype == "Invariant"
    ][0]

    # invariant of interest in model is 'inv {'Register 1' == 'Register 2' + 'Register 3'}'

    # Expression tree to pull apart is

    #               inv {'Register 1' == 'Register 2' + 'Register 3'}  (OperatorExpression with op '==')
    #                          /                                            \
    #                         (x)                                             (y)
    #                         /
    #                "Register 1" (via                                         \
    #        Feature-FeatureValue-FeatureReferenceExpression-                  +
    #          Membership-Feature)                                          /       \
    #

    assert hasattr(key_invar, "ownedRelationship")

    assert len(key_invar.ownedRelationship) == 2
    assert key_invar.ownedRelationship[0]._metatype == "ResultExpressionMembership"
    assert key_invar.ownedRelationship[1]._metatype == "ReturnParameterMembership"

    assert key_invar.ownedRelationship[0].ownedRelatedElement[0].operator == "=="

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        ._metatype
        == "FeatureValue"
    )

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .memberElement.declaredName
        == "Register 1"
    )

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[1]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .operator
        == "+"
    )

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[1]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .memberElement.declaredName
        == "Register 2"
    )

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[1]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[1]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[0]
        .memberElement.declaredName
        == "Register 3"
    )

    # check that path with relationship-specific names also work

    assert (
        key_invar.ownedRelationship[0]
        .ownedRelatedElement[0]
        .ownedRelationship[1]
        .memberElement.ownedRelationship[0]
        .target[0]
        .ownedRelationship[1]
        .memberElement.ownedRelationship[0]
        .target[0]
        .ownedRelationship[0]
        .memberElement.declaredName
        == "Register 3"
    )
