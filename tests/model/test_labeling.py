from pathlib import Path
from uuid import uuid4

import pytest

from pymbe.model import Element, Model
from pymbe.query.metamodel_navigator import get_effective_basic_name


def test_association_labels(load_kerml_library):

    """
    Test how labels present for Associations from the links library
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

    link_assoc_names = [str(bc) for bc in link_assocs]

    assert "Link «Association»" in link_assoc_names
    assert (
        "BinaryLink «Association» ([Anything «Classifier»] ←→ [Anything «Classifier»])"
        in link_assoc_names
    )
    assert (
        "SelfLink «Association» ([Anything «Classifier»] ←→ [Anything «Classifier»])"
        in link_assoc_names
    )


def test_inherited_name_labels(load_kerml_library):

    """
    Test that labels for Features include implied names
    """

    vvals_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "VectorValues"
    ][0]

    vvals_eles = vvals_ns.throughOwningMembership[0].throughOwningMembership

    three_vector_ele = None
    three_vector_dim = None

    for vvals_ele in vvals_eles:
        if vvals_ele._metatype in ("DataType"):
            if hasattr(vvals_ele, "declaredName"):
                if vvals_ele.declaredName == "ThreeVectorValue":
                    three_vector_ele = vvals_ele

    assert three_vector_ele is not None

    for three_vector_feat in three_vector_ele.throughFeatureMembership:
        if get_effective_basic_name(three_vector_feat) == "dimension":
            three_vector_dim = three_vector_feat

    assert str(three_vector_dim) == ":>>dimension «Feature»"


def test_vector_plus_invariant_labels(load_kerml_library):

    """
    Test labels for the Vector addition invariants
    """

    vfunc_ns = [
        library_model_ns
        for library_model_ns in load_kerml_library.ownedElement
        if library_model_ns.throughOwningMembership[0].declaredName == "VectorFunctions"
    ][0]

    vfunc_eles = vfunc_ns.throughOwningMembership[0].throughOwningMembership

    vector_add = None

    for vfunc_ele in vfunc_eles:
        if vfunc_ele._metatype in ("Function"):
            if hasattr(vfunc_ele, "declaredName"):
                if vfunc_ele.declaredName == "+":
                    vector_add = vfunc_ele

    assert vector_add is not None

    vector_add_labels = []

    for vector_add_feature in vector_add.throughFeatureMembership:
        vector_add_labels.append(str(vector_add_feature))

    assert "w == null or isZeroVector(w) implies u == w «Invariant»" in vector_add_labels
    assert "w != null implies u == w + v «Invariant»" in vector_add_labels
