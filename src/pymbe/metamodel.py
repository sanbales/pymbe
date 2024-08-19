import json
from dataclasses import field
from importlib import resources as lib_resources
from typing import Any, Dict, List

from pymbe.query.metamodel_navigator import get_more_general_types

# TODO: Is there a way to restore type hints for Element without inducing a circular dependency?


class MetaModel:
    """
    A class to hold meta-model information and perform property derivation
    """

    metamodel_hints: Dict[str, List[List[str]]] = field(default_factory=dict)
    pre_made_dicts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    relationship_metas: List[str]

    def __init__(self):
        self.pre_made_dicts = {}
        self._load_metahints()
        for metaclass in self.metamodel_hints:
            self._load_template_data(metaclass_name=metaclass)

    def _load_metahints(self):
        """Load data file to get attribute hints"""
        ecore_atts, ecore_refs = {}, {}
        with lib_resources.open_text("pymbe.static_data", "sysml_ecore_atts.json") as sysml_ecore:
            ecore_atts = json.load(sysml_ecore)
        with lib_resources.open_text(
            "pymbe.static_data", "sysml_ecore_derived_refs.json"
        ) as sysml_ecore_refs:
            ecore_refs = json.load(sysml_ecore_refs)
<<<<<<< HEAD
        # keys should be the same since they are all identified metaelements from ecore
        self.metamodel_hints = {k: ecore_atts[k] + ecore_refs[k] for k in ecore_atts}
=======

        hints_build = {}

        for att_key in ecore_atts.keys():
            inner_build = {}
            for ecore_att in ecore_atts[att_key]:
                inner_build.update({ecore_att[0]: ecore_att[1:]})
            for ecore_ref in ecore_refs[att_key]:
                inner_build.update({ecore_ref[0]: ecore_ref[1:]})
            hints_build.update({att_key: inner_build})

        # keys should be the same since they are all identified metaelements from ecore
        self.metamodel_hints = hints_build
>>>>>>> bjorn/main

    def _load_template_data(self, metaclass_name: str):
        defaults = {"Boolean": False, "String": "", "Integer": 0}

        data_template = {}
        for hint in self.metamodel_hints[metaclass_name]:
            key, _, role, dtype, *_, num_attributes = hint
            num_attributes = int(num_attributes)

<<<<<<< HEAD
            starter_field = None
            if role == "primary":
                default = defaults.get(dtype)
                # TODO: Figure out why some boolean and string attributes have 0 to -1
                # rather than 1 to 1 multiplicity
                if num_attributes > 1 or (
                    num_attributes == -1 and not (dtype in ("Boolean", "String"))
=======
        for hint_key, hint_vals in local_hints.items():
            starter_field = None
            if hint_vals[1] == "primary":
                # TODO: Figure out why some boolean and string attributes have 0 to -1
                # rather than 1 to 1 multiplicity
                if (
                    int(hint_vals[5]) > 1
                    or int(hint_vals[5]) == -1
                    and not (hint_vals[2] == "Boolean" or hint_vals[2] == "String")
>>>>>>> bjorn/main
                ):
                    starter_field = []
                else:
                    # TODO: One other janky override
<<<<<<< HEAD
                    if key == "aliasIds":
                        starter_field = []
                    elif default is not None:
                        starter_field = default
            data_template.update({key: starter_field})
        self.pre_made_dicts.update({metaclass_name: data_template})
=======
                    if hint_key == "aliasIds":
                        starter_field = []
                    elif hint_vals[2] == "Boolean":
                        starter_field = False
                    elif hint_vals[2] == "String":
                        starter_field = ""
                    elif hint_vals[2] == "Integer":
                        starter_field = 0

            data_template.update({hint_key: starter_field})

        self.pre_made_dicts.update({metaclass_name: data_template})


def list_relationship_metaclasses():
    """
    Return a list of relationship metaclass names
    """
    return [
        "FeatureTyping",
        "Membership",
        "OwningMembership",
        "FeatureMembership",
        "EndFeatureMembership",
        "Specialization",
        "Conjugation",
        "Subclassification",
        "Subsetting",
        "Redefinition",
        "FeatureValue",
        "ReferenceSubsetting",
    ]


def classifier_metas():
    return {"Classifier", "Behavior", "Structure"}


def assoc_metas():
    return {"Association"}


def connector_metas():
    return {"Connector", "Succession"}


def datatype_metas():
    return {"DataType"}


def feature_metas():
    return {"Feature", "Step"}


def derive_attribute(key: str, ele: "Element"):  # noqa: F821

    # entry point for deriving attributes within elements on demand

    if key == "type":
        return derive_type(ele)
    if "owned" in key and key not in ("ownedMember",):
        return derive_owned_x(ele, key[5:])
    if key == "ownedMember":
        return derive_owned_member(ele)
    if key == "feature":
        return derive_features(ele)

    raise NotImplementedError(f"The method to derive {key} has yet to be developed.")


def derive_type(ele: "Element"):  # noqa: F821

    if hasattr(ele, "throughFeatureTyping"):
        return ele.throughFeatureTyping

    return []


def derive_owned_member(ele: "Element"):  # noqa: F821

    found_ele = []

    for owned_rel in ele.ownedRelationship:
        if owned_rel._metatype == "OwningMembership":
            # TODO: Make this work with generalization of metatypes

            for owned_related_ele in owned_rel.ownedRelatedElement:
                found_ele.append(owned_related_ele)

    return found_ele


def derive_owned_x(ele: "Element", owned_kind: str):  # noqa: F821

    found_ele = []

    for owned_rel in ele.ownedRelationship:
        for owned_related_ele in owned_rel.ownedRelatedElement:
            if owned_related_ele._metatype == owned_kind:
                found_ele.append(owned_related_ele)

    return found_ele


def derive_inherited_featurememberships(ele: "Element"):  # noqa: F821
    """
    8.3.3.1.10 Type

    All Memberships inherited by this Type via Specialization or Conjugation.
    These are included in the derived union for the memberships of the Type
    """

    more_general = get_more_general_types(ele, 0, 100)

    try:
        fms_to_return = []
        for general_type in more_general:
            if hasattr(general_type, "ownedRelationship"):
                for inherited_fm in general_type.ownedRelationship:
                    if inherited_fm._metatype == "FeatureMembership":
                        fms_to_return.append(inherited_fm)
        return fms_to_return
    except AttributeError:
        return []


def derive_features(ele: "Element"):  # noqa: F821
    """
    8.3.3.1.10 Type

    The ownedMemberFeatures of the featureMemberships of this Type.
    """

    # TODO: Add a way to reach back to library for the inherited objects

    return [
        feature_membership.target[0]
        for feature_membership in derive_inherited_featurememberships(ele)
    ] + ele.throughFeatureMembership
>>>>>>> bjorn/main
