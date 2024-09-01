from uuid import uuid4

import pytest

import pymbe.api as pm
from pymbe.interpretation.occurrences_steps import is_feature_involving_self
from pymbe.model import Element, Model
from pymbe.model_modification import (
    apply_covered_feature_pattern,
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


def test_fill_self_references(load_kerml_library):

    """
    Test creation of performances and also fill in and cover library features like portionOf.

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

    exec_package_model_data = {
        "name": "Executed Process Model",
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

    exec_package = Element.new(data=exec_package_model_data, model=empty_model)

    empty_model.reference_other_model(load_kerml_library)

    new_performance = build_from_classifier_pattern(
        owner=new_package,
        name="New Process",
        model=empty_model,
        metatype="Behavior",
        superclasses=[performance],
        specific_fields={"ownedRelationship": []},
    )

    # spot check Features

    self_feat = None
    time_enclosed_feat = None

    for new_perf_feature in new_performance.feature:
        if get_effective_basic_name(new_perf_feature) == "self":
            assert is_feature_involving_self(new_perf_feature)
            self_feat = new_perf_feature

        if get_effective_basic_name(new_perf_feature) == "timeEnclosedOccurrences":
            assert is_feature_involving_self(new_perf_feature)
            time_enclosed_feat = new_perf_feature

        if get_effective_basic_name(new_perf_feature) == "differencesOf ":
            assert not is_feature_involving_self(new_perf_feature)

    new_step_type_1 = build_from_classifier_pattern(
        owner=new_package,
        name="Sub-Process 1",
        model=empty_model,
        metatype="Behavior",
        superclasses=[performance],
        specific_fields={"ownedRelationship": []},
    )

    new_step_type_2 = build_from_classifier_pattern(
        owner=new_package,
        name="Sub-Process 2",
        model=empty_model,
        metatype="Behavior",
        superclasses=[performance],
        specific_fields={"ownedRelationship": []},
    )

    new_step_1 = build_from_feature_pattern(
        owner=new_performance,
        name="Step 1",
        model=empty_model,
        specific_fields={},
        feature_type=new_step_type_1,
        direction="",
        metatype="Step",
        connector_end=False,
    )

    new_step_2 = build_from_feature_pattern(
        owner=new_performance,
        name="Step 2",
        model=empty_model,
        specific_fields={},
        feature_type=new_step_type_2,
        direction="",
        metatype="Step",
        connector_end=False,
    )

    assert len(new_performance.throughFeatureMembership) == 2

    # apply the atom pattern to some new process

    new_performance_exec = build_from_classifier_pattern(
        owner=exec_package,
        name="New Process as Executed",
        model=empty_model,
        metatype="Behavior",
        superclasses=[new_performance],
        specific_fields={"ownedRelationship": []},
    )

    step_1_exec = build_from_classifier_pattern(
        owner=exec_package,
        name="Sub-Process 1 as Executed",
        model=empty_model,
        metatype="Behavior",
        superclasses=[new_step_type_1],
        specific_fields={"ownedRelationship": []},
    )

    step_2_exec = build_from_classifier_pattern(
        owner=exec_package,
        name="Sub-Process 2 as Executed",
        model=empty_model,
        metatype="Behavior",
        superclasses=[new_step_type_2],
        specific_fields={"ownedRelationship": []},
    )

    # cover self

    covered_self = apply_covered_feature_pattern(
        one_member_classifiers=[new_performance_exec],
        feature_to_cover=self_feat,
        type_to_apply_pattern_on=new_performance_exec,
        model=empty_model,
        new_types_owner=exec_package,
        covering_classifier_prefix="Class to Cover ",
        covering_classifier_suffix="",
        redefining_feature_prefix="",
        redefining_feature_suffix="(Closed)",
    )

    assert covered_self.reverseFeatureMembership[0] == new_performance_exec

    assert covered_self.throughFeatureTyping[0] == new_performance_exec

    # cover timeEnclosedOccurrences with steps and self

    covered_time_enclosed = apply_covered_feature_pattern(
        one_member_classifiers=[new_performance_exec, step_1_exec, step_2_exec],
        feature_to_cover=time_enclosed_feat,
        type_to_apply_pattern_on=new_performance_exec,
        model=empty_model,
        new_types_owner=exec_package,
        covering_classifier_prefix="Class to Cover ",
        covering_classifier_suffix="",
        redefining_feature_prefix="",
        redefining_feature_suffix="(Closed)",
    )

    assert len(covered_time_enclosed.throughFeatureTyping[0].throughUnioning) == 3
