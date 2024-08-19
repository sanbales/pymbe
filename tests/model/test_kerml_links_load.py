from pathlib import Path
from uuid import uuid4

import pytest

from pymbe.model import Element, Model


def test_links_associations(load_kerml_library):

    """
    Check that the main links in Links library have loaded and the expected names are all there.
    """

    links_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Links"
    ][0]

    link_eles = links_ns.throughOwningMembership[0].throughOwningMembership

    link_assocs = []

    for link_ele in link_eles:
        if link_ele._metatype in ("Association"):
            assert hasattr(link_ele, "declaredName")
            link_assocs.append(link_ele)

    assert len(link_assocs) == 3

    link_assoc_names = [bc.declaredName for bc in link_assocs]

    assert "Link" in link_assoc_names
    assert "BinaryLink" in link_assoc_names
    assert "SelfLink" in link_assoc_names

    return None


def test_binarylink_has_features(load_kerml_library):

    """
    Check that the features participant, source, and target are loaded under BinaryLink
    """

    links_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Links"
    ][0]

    link_eles = links_ns.throughOwningMembership[0].throughOwningMembership

    binarylink_assoc = None

    for link_ele in link_eles:
        if link_ele._metatype in ("Association"):
            if hasattr(link_ele, "declaredName"):
                if link_ele.declaredName == "BinaryLink":
                    binarylink_assoc = link_ele

    assert len(binarylink_assoc.throughFeatureMembership) == 3

    return None


def test_binarylink_feature_details(load_kerml_library):

    """
    Check that the features participant, source, and target have appropriate subsetting, redefinition,
    and other metadata
    """

    links_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "Links"
    ][0]

    # there are only DataValue and Anything in the base library

    link_eles = links_ns.throughOwningMembership[0].throughOwningMembership

    binarylink_assoc = None

    for link_ele in link_eles:
        if link_ele._metatype in ("Association"):
            if hasattr(link_ele, "declaredName"):
                if link_ele.declaredName == "BinaryLink":
                    binarylink_assoc = link_ele

    # check on the Feature Memberships under BinaryLink

    binary_owned_rels = [
        owned_rel
        for owned_rel in binarylink_assoc.ownedRelationship
        if owned_rel._metatype == "FeatureMembership"
    ]

    assert len(binary_owned_rels) == 3

    binary_features = [feature for feature in binarylink_assoc.throughFeatureMembership]

    end_feature_count = 0
    participant_feature_count = 0

    for bf in binary_features:
        if bf.isEnd:
            end_feature_count = end_feature_count + 1
        else:
            participant_feature_count = participant_feature_count + 1

    assert end_feature_count == 2
    assert participant_feature_count == 1

    # check features for subsetting and redefintion

    participant_feature = None

    source_feature = None
    target_feature = None

    for feat_ele in binary_features:
        if hasattr(feat_ele, "declaredName"):
            if feat_ele.declaredName == "participant":
                participant_feature = feat_ele
            elif feat_ele.declaredName == "source":
                source_feature = feat_ele
            elif feat_ele.declaredName == "target":
                target_feature = feat_ele

    assert participant_feature is not None
    assert source_feature is not None
    assert target_feature is not None

    assert len(participant_feature.throughRedefinition) == 1
    assert len(participant_feature.reverseSubsetting) == 2

    assert source_feature in participant_feature.reverseSubsetting

    return None
