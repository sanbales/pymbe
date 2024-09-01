from pymbe.label import get_label_for_expression
from pymbe.model import Element
from pymbe.query.metamodel_navigator import get_effective_basic_name


def serialize_kerml_atom(atom_to_print):

    atom_string = "#atom\n"

    if hasattr(atom_to_print, "throughUnioning"):
        atom_string = ""

    if atom_to_print._metatype == "Classifier":
        atom_string = atom_string + "classifier "
    elif atom_to_print._metatype == "Behavior":
        atom_string = atom_string + "behavior "
    elif atom_to_print._metatype == "Association":
        atom_string = atom_string + "association "
    elif atom_to_print._metatype == "Succession":
        atom_string = atom_string + "succession "
    elif atom_to_print._metatype == "Structure":
        atom_string = atom_string + "struct "
    elif atom_to_print._metatype == "DataType":
        atom_string = atom_string + "datatype "

    atom_string = atom_string + atom_to_print.basic_name

    if len(atom_to_print.throughSubclassification) > 0:
        atom_string = (
            atom_string
            + " specializes "
            + ", ".join([sub.basic_name for sub in atom_to_print.throughSubclassification])
        )

    if hasattr(atom_to_print, "throughUnioning"):
        atom_string = (
            atom_string
            + " unions "
            + ", ".join([sub.basic_name for sub in atom_to_print.throughUnioning])
        )

    if len(atom_to_print.throughFeatureMembership) > 0:
        atom_string = atom_string + " {\n"
        for feature in atom_to_print.throughFeatureMembership:
            if feature._metatype == "Feature":
                atom_string = atom_string + "   feature "
            elif feature._metatype == "Step":
                atom_string = atom_string + "   step "
            elif feature._metatype == "Connector":
                atom_string = atom_string + "   connector "
            elif feature._metatype == "Succession":
                atom_string = atom_string + "   succession "

            if len(feature.throughFeatureTyping) > 0:
                atom_string = (
                    atom_string
                    + feature.basic_name
                    + " : "
                    + feature.throughFeatureTyping[0].basic_name
                )
            else:
                atom_string = atom_string + feature.basic_name

            if len(feature.throughRedefinition) > 0:
                atom_string = (
                    atom_string
                    + " redefines "
                    + ", ".join([sub.basic_name for sub in feature.throughRedefinition])
                )

            if hasattr(feature, "throughFeatureChaining"):
                atom_string = atom_string + " chains " + '.'.join(map(get_effective_basic_name,
                                                    feature.throughFeatureChaining))

            if len(feature.throughFeatureValue) > 0:
                atom_string = atom_string + " = " + str(feature.throughFeatureValue[0])

            atom_string = atom_string + ";\n"
        atom_string = atom_string + "}\n"
    elif hasattr(atom_to_print, "throughEndFeatureMembership"):
        atom_string = atom_string + " {\n"
        for feature in atom_to_print.throughEndFeatureMembership:
            if feature._metatype == "Feature":
                atom_string = atom_string + "   end feature "
            elif feature._metatype == "Step":
                atom_string = atom_string + "   step "
            elif feature._metatype == "Connector":
                atom_string = atom_string + "   connector "

            if hasattr(feature, "throughFeatureTyping"):
                atom_string = (
                    atom_string
                    + feature.basic_name
                    + " : "
                    + feature.throughFeatureTyping[0].basic_name
                )
            else:
                atom_string = atom_string + feature.basic_name

            if len(feature.throughRedefinition) > 0:
                atom_string = (
                    atom_string
                    + " redefines "
                    + ", ".join([sub.basic_name for sub in feature.throughRedefinition])
                )

            atom_string = atom_string + ";\n"

        atom_string = atom_string + "}\n"
    else:
        atom_string = atom_string + ";\n"

    return atom_string


def serialize_sysml_package(package_to_print: Element, tabs: int = 0):

    sysml_string = "\t" * tabs

    if package_to_print._metatype == "Package":
        sysml_string = sysml_string + "package "

    sysml_string = sysml_string + "'" + package_to_print.declaredName + "'"

    if len(package_to_print.throughOwningMembership) > 0:
        sysml_string = sysml_string + " {\n"
        for oe in package_to_print.throughOwningMembership:
            if oe._metatype in ("AttributeUsage", "AttributeDefinition"):
                sysml_string = sysml_string + serialize_sysml_attribute(
                    attribute_to_print=oe, tabs=tabs + 1
                )
            elif oe._metatype in ("ItemUsage", "ItemDefinition", "PartUsage", "PartDefinition"):
                sysml_string = sysml_string + serialize_sysml_item(item_to_print=oe, tabs=tabs + 1)
        sysml_string = sysml_string + "\t" * tabs + "}\n"
    else:
        sysml_string = sysml_string + ";\n"

    return sysml_string


def serialize_sysml_item(item_to_print, tabs: int = 0):
    sysml_string = "\t" * tabs

    if item_to_print._metatype == "ItemUsage":
        sysml_string = sysml_string + "item "
    elif item_to_print._metatype == "ItemDefinition":
        sysml_string = sysml_string + "item def "
    elif item_to_print._metatype == "PartUsage":
        sysml_string = sysml_string + "part "
    elif item_to_print._metatype == "PartDefinition":
        sysml_string = sysml_string + "part def "

    sysml_string = sysml_string + get_effective_basic_name(item_to_print)

    if len(item_to_print.throughFeatureTyping) > 0:
        sysml_string = (
            sysml_string
            + " : "
            + ",".join([get_effective_basic_name(ft) for ft in item_to_print.throughFeatureTyping])
        )

    if len(item_to_print.throughFeatureValue) > 0:
        sysml_string = sysml_string + " = " + serialize_feature_values(item_to_print)

    if len(item_to_print.throughFeatureMembership) > 0:
        sysml_string = sysml_string + " {\n"
        for oe in item_to_print.throughFeatureMembership:
            if oe._metatype in ("AttributeUsage", "AttributeDefinition"):
                sysml_string = sysml_string + serialize_sysml_attribute(
                    attribute_to_print=oe, tabs=tabs + 1
                )
            elif oe._metatype in ("ItemUsage", "ItemDefinition", "PartUsage", "PartDefinition"):
                sysml_string = sysml_string + serialize_sysml_item(item_to_print=oe, tabs=tabs + 1)

        sysml_string = sysml_string + "\t" * tabs + "}\n"
    else:
        sysml_string = sysml_string + ";\n"

    return sysml_string


def serialize_sysml_attribute(attribute_to_print, tabs: int = 0):
    sysml_string = "\t" * tabs

    if attribute_to_print._metatype == "AttributeUsage":
        sysml_string = sysml_string + "attribute "
    elif attribute_to_print._metatype == "AttributeDefinition":
        sysml_string = sysml_string + "attribute def "

    sysml_string = sysml_string + get_effective_basic_name(attribute_to_print)

    if len(attribute_to_print.throughFeatureTyping) > 0:
        sysml_string = (
            sysml_string
            + " : "
            + ",".join(
                [get_effective_basic_name(ft) for ft in attribute_to_print.throughFeatureTyping]
            )
        )

    if len(attribute_to_print.throughFeatureValue) > 0:
        sysml_string = sysml_string + " = " + serialize_feature_values(attribute_to_print)

    sysml_string = sysml_string + ";\n"

    return sysml_string


def serialize_feature_values(feature_to_value):
    sysml_string = ""

    for fv in feature_to_value.throughFeatureValue:
        if fv._metatype == "LiteralBoolean":
            sysml_string = sysml_string + str(fv.value)
        elif fv._metatype == "LiteralInteger":
            sysml_string = sysml_string + str(fv.value)
        elif fv._metatype == "LiteralRational":
            sysml_string = sysml_string + str(fv.value)
        elif fv._metatype == "LiteralString":
            sysml_string = sysml_string + "'" + fv.value + "'"
        elif fv._metatype == "LiteralUnlimitedNatural":
            sysml_string = sysml_string + str(fv.value)
        elif fv._metatype == "LiteralNull":
            sysml_string = sysml_string + "null"
        elif fv._metatype == "LiteralInfinity":
            sysml_string = sysml_string + "inf"
        elif fv._metatype == "LiteralUnlimited":
            sysml_string = sysml_string + "*"
        elif fv._metatype in (
            "OperatorExpression",
            "FeatureReferenceExpression",
            "FeatureChainExpression",
        ):
            sysml_string = sysml_string + serialize_expression(fv)

    return sysml_string


def serialize_expression(expr_to_print):
    sysml_string = ""

    sysml_string = sysml_string + get_label_for_expression(expr_to_print)

    return sysml_string
