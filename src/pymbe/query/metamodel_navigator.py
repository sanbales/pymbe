# a collection of convenience methods to navigate the metamodel when inspecting user models
from typing import List, Tuple

from ..model import Element


def is_type_undefined_mult(type_ele: Element):
    if "throughOwningMembership" not in type_ele._derived:
        return True
    mult_range = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    return len(mult_range) == 0


def is_multiplicity_one(type_ele):
    if "throughOwningMembership" not in type_ele._derived:
        return False

    multiplicity_range, *_ = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = tuple(
        int(li.value)
        for li in multiplicity_range.throughOwningMembership
        if li["@type"] == "LiteralInteger"
    )
    if not literal_value:
        return False
    if len(literal_value) == 1:
        # TODO: Ask Bjorn: what if it is `1..*`, this would return True
        return literal_value[0] == 1
    elif literal_value == (1, ):
        return True
    return True


def is_multiplicity_specific_finite(type_ele):
    if "throughOwningMembership" not in type_ele._derived:
        return False
    multiplicity_range, *_ = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = [
        li.value
        for li in multiplicity_range.throughOwningMembership
        if li["@type"] == "LiteralInteger"
    ]
    if not literal_value:
        return False

    if len(literal_value) == 1:
        lower, *_ = literal_value
        if lower > 1:
            return True
    elif len(literal_value) == 2:
        lower, upper = literal_value
        if lower > 1 and upper == lower:
            return True
    return False


def get_finite_multiplicity_types(model):
    model_types = [
        ele for ele in model.elements.values() if ele._metatype in ("Feature", "Classifier")
    ]

    return [
        finite_type
        for finite_type in model_types
        if is_multiplicity_one(finite_type) or is_multiplicity_specific_finite(finite_type)
    ]


def get_lower_multiplicty(type_ele):
    lower_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        return lower_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    if len(multiplicity_ranges) == 1:
        literal_value = [
            li.value
            for li in multiplicity_ranges[0].throughOwningMembership
            if li["@type"] == "LiteralInteger"
        ]
    elif len(multiplicity_ranges) > 1:
        literal_value = [
            li.value
            for li in multiplicity_ranges[0].throughOwningMembership
            if li["@type"] == "LiteralInteger"
        ]

    lower_mult = int(literal_value[0])

    return lower_mult


def get_upper_multiplicty(type_ele):
    upper_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        return upper_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    if len(multiplicity_ranges) == 1:
        literal_value = [
            li.value
            for li in multiplicity_ranges[0].throughOwningMembership
            if li["@type"] == "LiteralInteger"
        ]
    elif len(multiplicity_ranges) > 1:
        literal_value = [
            li.value
            for li in multiplicity_ranges[1].throughOwningMembership
            if li["@type"] == "LiteralInteger"
        ]

    upper_mult = int(literal_value[0])

    return upper_mult


def identify_connectors_one_side(connectors: List[Element]) -> Tuple[Element]:
    return tuple(
        {
            connector
            for connector in connectors
            for end_feature in connector.throughEndFeatureMembership
            if "throughEndFeatureMembership" in connector._derived
            and "throughReferenceSubsetting" in end_feature._derived
            and is_multiplicity_one(end_feature.throughReferenceSubsetting[0])
        }
    )
