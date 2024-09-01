# TODO: Refactor this whole thing and integrate it better with the new Model approach
# Module for computing useful labels and signatures for SysML v2 elements
from warnings import warn

from .model import Element, Model
from .query.metamodel_navigator import get_effective_basic_name

DEFAULT_MULTIPLICITY_LIMITS = dict(lower="0", upper="*")


def get_label(element: Element) -> str:  # pylint: disable=too-many-return-statements

    """
    Get a label for the element.
    The difference between a label and a name is that the name is meant to provide something by
    which to refer to the element in code or informally while processing a model. A label is
    meant to be part of the display of the element either in a representation (repr) or view
    of the model.
    """

    metatype = element._metatype
    if metatype.endswith("Expression"):
        return f"{get_label_for_expression(element)}" + f" «{metatype}»"
    if metatype == "Invariant" and "throughResultExpressionMembership" in element._derived:
        invar_expression = element.throughResultExpressionMembership[0]
        return f"{get_label_for_expression(invar_expression)} «{metatype}»"
    model = element._model
    data = element._data
    name = data.get("name") or data.get("effectiveName") or data.get("declaredName")

    # Get the element type(s)
    types: list = data.get("type") or []
    if isinstance(types, dict):
        types = [types]
    try:
        type_names = [model.elements[type_["@id"]]._data.get("name") for type_ in types]
        type_names = [str(type_name) for type_name in type_names if type_name]
    except KeyError:
        type_names = ["Unresolved type"]
    value = element._data.get("value")

    if name:
        if type_names:
            # TODO: look into using other types (if there are any)
            name += f": {type_names[0]}"
        return f"{name} «{metatype}»"
    if (
        metatype == "Feature"
        and hasattr(element, "owningRelationship")
        and element.owningRelationship is not None
        and "memberName" in element.owningRelationship._data
        and element.owningRelationship._metatype
        in ("ParameterMembership", "ReturnParameterMembership")
    ):
        direction = ""
        if "direction" in element._data:
            direction = element.direction + " "
        para_string = element.owningRelationship._data["memberName"]

        return f"{direction}{para_string} «{metatype}»"

    if metatype == "LiteralBoolean":
        metatype = type_names[0] if type_names else metatype.replace("Literal", "Occurred Literal")
        return f"{value} «{metatype}»"
    if value and metatype.startswith("Literal"):
        metatype = type_names[0] if type_names else metatype.replace("Literal", "Occurred Literal")
        return f"{value} «{metatype}»"
    if metatype == "MultiplicityRange":
        return get_label_for_multiplicity(multiplicity=element)

    if not name and element._metatype == "Feature":
        return f":>>{get_effective_basic_name(element)} «{metatype}»"

    if "@id" in data:
        return f"""{data["@id"]} «{metatype}»"""

    return "blank"


def get_label_for_id(element_id: str, model: Model) -> str:
    return get_label(model.elements[element_id])


def get_label_for_expression(expression: Element) -> str:
    meta = expression._metatype

    expression_label = f"<<{meta} {expression._id}>>"

    if meta == "OperatorExpression":
        # case for OperatorExpression - recurse on the parameters

        expression_label = f" {expression.operator} ".join(
            map(get_label_for_expression, expression.throughParameterMembership)
        )

    # case for the Features under an expression
    elif meta == "Feature":
        expression_label = get_label_for_expression(expression.throughFeatureValue[0])

    elif meta == "FeatureReferenceExpression":
        # case for FeatureReferenceExpression - terminal case #1
        try:
            expression_label = expression.throughMembership[0].basic_name
        except IndexError:
            # if there is no direct reference, need to recurse on parameters
            expression_label = get_label_for_expression(expression.throughFeatureMembership[0])
            # expression_label = "Empty FRE"
    elif meta == "FeatureChainExpression":
        try:
            # first item will be FRE to another feature
            expression_label = (
                expression.throughParameterMembership[0]
                .throughFeatureValue[0]
                .throughMembership[0]
                .basic_name
            )
            # check if this is a two-item feature chain or n > 2
            if (
                "throughMembership" in expression._derived
                and len(expression.throughMembership) > 0
            ):
                # if hasattr(expression, "throughMembership"):
                # this is the n = 2 case
                second_item = expression.throughMembership[0].basic_name
                expression_label += f".{second_item}"
            else:
                # this is the n > 2 case
                chains = expression.throughOwningMembership[0].throughFeatureChaining
                other_items = ".".join([chain.basic_name for chain in chains])
                expression_label += f".{other_items}"
        except IndexError as failed_link:
            expression_label = "Empty FCE"

    # covers Literal expression cases
    elif "value" in expression._data:
        # elif hasattr(expression, "value"):
        expression_label = str(expression.value)

    elif expression._metatype == "InvocationExpression":
        try:
            body = ", ".join(map(get_label_for_expression, expression.throughParameterMembership))
            expression_label = f"{expression.throughFeatureTyping[0].declaredName}({body})"
        except AttributeError:
            expression_label = "Empty InvocationExpression"

    elif expression._metatype == "NullExpression":
        expression_label = "null"
    else:
        warn(f"Cannot process {expression._metatype} elements yet!")
    return expression_label


def get_label_for_multiplicity(multiplicity: Element) -> str:
    values = {}
    data = multiplicity._data
    for limit, default in DEFAULT_MULTIPLICITY_LIMITS.items():
        literal_id = (data.get(f"{limit}Bound") or {}).get("@id")
        if literal_id is None:
            values[limit] = default
            continue
        value = multiplicity._model.elements.get(literal_id, {})._data.get("value")
        if value is None:
            value = default
        values[limit] = value
    return f"""{values["lower"]}..{values["upper"]}"""


def get_qualified_label(element: Element, parameter_name_map: dict, start: bool) -> str:
    earlier_name = "Model"

    element_data = element._data
    all_elements = element._model.elements

    if "owner" in element_data:
        if element_data["owner"] is not None:
            element_owner = all_elements[element_data["owner"]["@id"]]
            earlier_name = get_qualified_label(element_owner, parameter_name_map, False)
    else:
        name = element_data["name"]
        return f"{name}"

    printed_name = ""
    if element_data["@id"] in parameter_name_map:
        printed_name = parameter_name_map[element_data["@id"]]
    else:
        printed_name = get_label(element)

    tail = ""
    if start:
        tail = " <<" + element_data["@type"] + ">>"

    earlier_name = f"{earlier_name}::{printed_name}{tail}"

    return earlier_name
