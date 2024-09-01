import pytest

from pymbe.model import Element, Model
from pymbe.query.metamodel_navigator import (
    get_effective_basic_name,
    get_effective_lower_multiplicity,
    get_effective_upper_multiplicity,
    get_most_specific_feature_type,
)


def test_find_redefined_name(load_kerml_library):

    """
    Test that codebase can find names for features that use the modeling pattern of
    redefining a library feature as below:

    classifier MyNewClass {
        redefines someLibraryFeature;
    }

    """
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

    found_dispatch = False

    for perf_feat in performance.throughFeatureMembership:
        test_name = get_effective_basic_name(perf_feat)
        if test_name == "isDispatch":
            found_dispatch = True

    assert found_dispatch


def test_find_redefined_multiplicity(load_kerml_library):

    """
    Test that codebase can find multiplicities for features that use the modeling pattern of
    redefining a library feature as below:

    classifier MyNewClass {
        redefines someLibraryFeature;
    }

    Test example is on the dimension Feature of ThreeVectorValue which redefines the same feature in
    NumericalVectorValue while declaring neither name nor multiplicity explicitly in the KerML code

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

    assert get_effective_lower_multiplicity(three_vector_dim) == 0
    assert get_effective_upper_multiplicity(three_vector_dim) == 1


def test_find_redefined_typing(load_kerml_library):

    """
    Test that codebase can find types for features that use the modeling pattern of
    redefining a library feature as below:

    classifier MyNewClass {
        redefines someLibraryFeature;
    }

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

    dim_type = get_most_specific_feature_type(three_vector_dim)

    assert dim_type.declaredName == "Positive"
