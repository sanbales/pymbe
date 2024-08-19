# a collection of convenience methods to navigate the metamodel when inspecting user models


def is_type_undefined_mult(type_ele):
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
    if literal_value == (1, 1):
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


def get_lower_multiplicity(type_ele):
    lower_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        return lower_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = []
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

    if len(literal_value) > 0:
        lower_mult = int(literal_value[0])

    return lower_mult


def get_upper_multiplicity(type_ele):
    upper_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        return upper_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = []
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

    upper_mult = int(literal_value[1])

    return upper_mult


def get_effective_lower_multiplicity(type_ele):

    """
    Get lower multiplicity on feature even if this type is redefined.
    """

    lower_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        local_general = get_more_general_types(type_ele, 1, 1)

        # if the most local general types have a finite lower multiplicity,
        # use the most restrictive

        lgm = max(
            [
                get_lower_multiplicity(lg)
                for lg in local_general
                if get_lower_multiplicity(lg) > -1
            ],
            default=-1,
        )

        if lgm > -1:
            return lgm

        # if we still have no multiplicity, try to recurse

        lower_mult = max(
            [
                get_effective_lower_multiplicity(lg)
                for lg in local_general
                if get_lower_multiplicity(lg) > -1
            ],
            default=-1,
        )

        # will either return with a lower multiplicity or if we
        # still don't find a defined lower multiplicity

        return lower_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = []
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

    if len(literal_value) > 0:
        lower_mult = int(literal_value[0])

    return lower_mult


def get_effective_upper_multiplicity(type_ele):

    """
    Get upper multiplicity on feature even if this type is redefined.
    """

    upper_mult = -1
    if "throughOwningMembership" not in type_ele._derived:
        local_general = get_more_general_types(type_ele, 1, 1)

        # if the most local general types have a finite lower multiplicity,
        # use the most restrictive

        ugm = max(
            [
                get_upper_multiplicity(lg)
                for lg in local_general
                if get_upper_multiplicity(lg) > -1
            ],
            default=-1,
        )

        if ugm > -1:
            return ugm

        # if we still have no multiplicity, try to recurse

        upper_mult = max(
            [
                get_effective_upper_multiplicity(lg)
                for lg in local_general
                if get_upper_multiplicity(lg) > -1
            ],
            default=-1,
        )

        # will either return with a lower multiplicity or if we still
        # don't find a defined lower multiplicity

        return upper_mult
    multiplicity_ranges = [
        mr for mr in type_ele.throughOwningMembership if mr["@type"] == "MultiplicityRange"
    ]
    literal_value = []
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

    if len(literal_value) > 0:
        upper_mult = int(literal_value[1])

    return upper_mult


def identify_connectors_one_side(connectors):
    one_sided = []
    for connector in connectors:
        if "throughEndFeatureMembership" in connector._derived:
            for end_feature in connector.throughEndFeatureMembership:

                if "throughReferenceSubsetting" in end_feature._derived:
                    if (
                        is_multiplicity_one(end_feature.throughReferenceSubsetting[0])
                        and connector not in one_sided
                    ):
                        one_sided.append(connector)

    # return tuple(
    #     {
    #         connector
    #         for connector in connectors
    #         for end_feature in connector.throughEndFeatureMembership
    #         if "throughEndFeatureMembership" in connector._derived
    #         and "throughReferenceSubsetting" in end_feature._derived
    #         and is_multiplicity_one(end_feature.throughReferenceSubsetting[0])
    #     }
    # )

    return one_sided


def does_behavior_have_write_features(behavior):

    if hasattr(behavior, "throughFeatureMembership"):

        candidate_features = behavior.throughFeatureMembership

        for candidate_feature in candidate_features:

            if candidate_feature._metatype == "Step":
                step = candidate_feature

                if hasattr(step, "throughFeatureTyping"):
                    candidate_types = step.throughFeatureTyping
                    for candidate_type in candidate_types:
                        if candidate_type.declaredName == "FeatureWritePerformance":
                            return True

    return False


def has_type_named(feature, type_name):
    if hasattr(feature, "throughFeatureTyping"):
        for feature_type in feature.throughFeatureTyping:
            if feature_type.basic_name == type_name:
                return True
    return False


def get_most_specific_feature_type(feature):

    if hasattr(feature, "throughFeatureTyping"):
        if len(feature.throughFeatureTyping) == 1:
            return feature.throughFeatureTyping[0]

    if hasattr(feature, "throughRedefinition"):
        if len(feature.throughRedefinition) == 1:
            redef_feature = feature.throughRedefinition[0]

            if hasattr(redef_feature, "throughFeatureTyping"):
                if len(redef_feature.throughFeatureTyping) == 1:
                    return redef_feature.throughFeatureTyping[0]

            return get_most_specific_feature_type(redef_feature)

    return None


def get_more_general_types(typ, recurse_counter, max_counter):
    """
    Recursively navigate along Specialization relationships to find all the more general
    types of the given type
    """

    local_more_general = (
        typ.throughFeatureTyping + typ.throughSubclassification + typ.throughRedefinition
    )

    # check to see if this is a library object
    # TODO: This is very hacky, need to check on library connections much better

    lib_local = []
    lib_remove = []

    for local_general in local_more_general:
        if len(typ._model._referenced_models) > 0:
            if hasattr(local_general, "isLibraryElement"):
                if local_general._data["isLibraryElement"]:
                    trial_element = typ._model._referenced_models[0].get_element(local_general._id)
                    lib_local.append(trial_element)
                    lib_remove.append(local_general)
            else:
                try:
                    trial_element = typ._model._referenced_models[0].get_element(local_general._id)
                    lib_local.append(trial_element)
                    lib_remove.append(local_general)
                except KeyError:
                    pass

    for to_remove in lib_remove:
        local_more_general.remove(to_remove)

    local_more_general = local_more_general + lib_local

    if recurse_counter >= max_counter:
        return local_more_general

    total_general = local_more_general + [
        item
        for local_general in local_more_general
        for item in get_more_general_types(local_general, recurse_counter + 1, max_counter)
    ]

    return total_general


def get_feature_bound_values(feat):
    for bound_val in feat.throughFeatureValue:
        if bound_val._metatype == "FeatureReferenceExpression":
            referred_item = bound_val.throughMembership[0]

            return [referred_item]

    return []


def get_effective_basic_name(type_ele):

    """
    Get a name for the feature even if this type is redefined.
    """

    name = ""

    if type_ele.basic_name != "":
        return type_ele.basic_name

    if "throughRedefinition" in type_ele._derived:
        local_general = type_ele.throughRedefinition

        # if the most local general types have a name, check for conflicts

        for one_general in local_general:
            if one_general.basic_name != "":
                name = one_general.basic_name

        # if we still have no name, try to recurse

        if name == "":
            for one_general in local_general:
                trial_name = get_effective_basic_name(one_general)
                if trial_name != "":
                    name = trial_name

        # will either return with a lower multiplicity or if we
        # still don't find a defined lower multiplicity

        return name

    return ""
